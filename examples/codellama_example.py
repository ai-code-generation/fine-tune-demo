#!/usr/bin/env python3
"""
CodeLlama fine-tuning example for the NeMo Fine-Tuning Pipeline.

This example demonstrates:
1. Fine-tuning CodeLlama for code generation tasks
2. Code-specific data processing
3. Code-specific evaluation metrics
4. Optimized configuration for code models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainer import NeMoTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import MetricsCalculator
from deployment.deployer import ModelDeployer
from config.config_manager import ConfigManager
from data.data_processor import DataProcessor
from data.instruction_formatter import InstructionFormatter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_code_training_data():
    """Create sample code training data."""
    logger.info("Creating code training data...")
    
    code_examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Java developer specializing in test automation with SWTBot."
                },
                {
                    "role": "user",
                    "content": "Create a SWTBot test to click a button with label 'Submit' and verify a dialog appears."
                },
                {
                    "role": "assistant",
                    "content": """```java
@Test
public void testSubmitButtonClickOpensDialog() {
    // Click the Submit button
    bot.button("Submit").click();
    
    // Wait for dialog to appear
    bot.waitUntil(Conditions.shellIsActive("Confirmation Dialog"));
    
    // Verify dialog is present
    SWTBotShell dialog = bot.shell("Confirmation Dialog");
    assertTrue("Dialog should be active", dialog.isActive());
    
    // Close dialog
    dialog.close();
}
```"""
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Java developer specializing in test automation with SWTBot."
                },
                {
                    "role": "user",
                    "content": "Write a SWTBot test to fill out a form with name and email fields."
                },
                {
                    "role": "assistant",
                    "content": """```java
@Test
public void testFillForm() {
    // Fill name field
    bot.textWithLabel("Name:").setText("John Doe");
    
    // Fill email field
    bot.textWithLabel("Email:").setText("john.doe@example.com");
    
    // Verify fields are filled correctly
    assertEquals("John Doe", bot.textWithLabel("Name:").getText());
    assertEquals("john.doe@example.com", bot.textWithLabel("Email:").getText());
    
    // Submit form
    bot.button("Submit").click();
}
```"""
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Java developer specializing in test automation with SWTBot."
                },
                {
                    "role": "user",
                    "content": "Create a test to select an item from a dropdown menu and verify the selection."
                },
                {
                    "role": "assistant",
                    "content": """```java
@Test
public void testDropdownSelection() {
    // Open dropdown
    SWTBotCombo dropdown = bot.comboBoxWithLabel("Category:");
    
    // Select item
    dropdown.setSelection("Electronics");
    
    // Verify selection
    assertEquals("Electronics", dropdown.selection());
    
    // Alternative: Select by index
    dropdown.setSelection(0);
    
    // Verify first item is selected
    assertTrue("First item should be selected", dropdown.selectionIndex() == 0);
}
```"""
                }
            ]
        }
    ]
    
    # Save as YAML
    import yaml
    
    output_file = "data/code_training.yaml"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump_all(code_examples, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Created code training data: {output_file}")
    return output_file


def setup_codellama_config():
    """Setup optimized configuration for CodeLlama."""
    logger.info("Setting up CodeLlama configuration...")
    
    config_manager = ConfigManager("codellama")
    config = config_manager.load_config()
    
    # Customize for code generation
    code_overrides = {
        "model": {
            "base_model_path": "codellama/CodeLlama-7b-hf"
        },
        "data": {
            "max_seq_length": 8192,  # Longer sequences for code
            "batch_size": 1,  # Small batch due to long sequences
        },
        "training": {
            "max_epochs": 3,
            "learning_rate": 5e-5,  # Conservative LR for code
            "accumulate_grad_batches": 16,  # High accumulation
            "weight_decay": 0.05  # Higher weight decay
        },
        "lora": {
            "rank": 16,  # Lower rank for code specialization
            "alpha": 32,
            "dropout": 0.05,
            "layer_selection": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        },
        "evaluation": {
            "metrics": ["perplexity", "bleu", "rouge", "code_bleu", "exact_match", "syntax_validity"]
        }
    }
    
    config_manager.update_config(code_overrides)
    
    # Optimize for hardware
    config_manager.optimize_for_hardware(gpu_memory_gb=24, num_gpus=1)
    
    # Save custom config
    custom_config_path = "configs/training/codellama_custom.yaml"
    config_manager.save_config(custom_config_path)
    
    logger.info(f"CodeLlama configuration saved: {custom_config_path}")
    return custom_config_path


def train_codellama_model(config_path: str, train_file: str):
    """Train CodeLlama model."""
    logger.info("Training CodeLlama model...")
    
    # Initialize trainer
    trainer = NeMoTrainer(
        model_type="codellama",
        model_size="7b",
        config_path=config_path
    )
    
    # Setup model
    trainer.setup_model(
        base_model_path="codellama/CodeLlama-7b-hf"
    )
    
    # Setup data
    trainer.setup_data(
        train_file=train_file,
        val_file=train_file,  # Using same file for demo
        test_file=train_file
    )
    
    # Setup trainer
    trainer.setup_trainer(
        output_dir="checkpoints/codellama",
        log_dir="logs/codellama",
        max_epochs=3,
        gpus=1,
        precision="bf16",
        accumulate_grad_batches=16
    )
    
    # Train
    trainer.train()
    
    # Save model
    model_path = "checkpoints/codellama/codellama_swtbot.nemo"
    trainer.save_model(model_path)
    
    logger.info(f"CodeLlama model trained and saved: {model_path}")
    return model_path


def evaluate_code_model(model_path: str, test_file: str):
    """Evaluate CodeLlama model with code-specific metrics."""
    logger.info("Evaluating CodeLlama model...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator("codellama", model_path)
    evaluator.load_model(model_path)
    evaluator.load_tokenizer(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_on_file(
        test_file=test_file,
        output_file="logs/codellama/code_evaluation.json",
        max_examples=10
    )
    
    # Calculate code-specific metrics
    metrics_calculator = MetricsCalculator("codellama")
    
    # Sample code predictions and targets for demonstration
    code_predictions = [
        """```java
@Test
public void testButtonClick() {
    bot.button("Submit").click();
    bot.waitUntil(Conditions.shellIsActive("Dialog"));
}
```""",
        """```java
@Test
public void testFormFill() {
    bot.textWithLabel("Name:").setText("Test User");
    assertEquals("Test User", bot.textWithLabel("Name:").getText());
}
```"""
    ]
    
    code_targets = [
        """```java
@Test
public void testSubmitButton() {
    bot.button("Submit").click();
    bot.waitUntil(Conditions.shellIsActive("Confirmation Dialog"));
}
```""",
        """```java
@Test
public void testNameField() {
    bot.textWithLabel("Name:").setText("John Doe");
    assertEquals("John Doe", bot.textWithLabel("Name:").getText());
}
```"""
    ]
    
    code_metrics = metrics_calculator.calculate_code_metrics(
        code_predictions,
        code_targets
    )
    
    logger.info("Code evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Code-specific metrics:")
    for metric, value in code_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    return results, code_metrics


def deploy_code_model(model_path: str):
    """Deploy CodeLlama model with code-specific features."""
    logger.info("Deploying CodeLlama model...")
    
    # Deploy model
    deployer = ModelDeployer("codellama", "7b", "deploy/codellama")
    deployment_path = deployer.deploy_model(
        checkpoint_path=model_path,
        model_name="codellama_swtbot_assistant",
        copy_tokenizer=True,
        create_config=True,
        create_readme=True
    )
    
    # Create specialized inference script for code generation
    inference_script = f'''#!/usr/bin/env python3
"""
CodeLlama SWTBot Assistant Inference Script
"""

import argparse
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

def load_model(model_path):
    """Load the fine-tuned CodeLlama model."""
    print(f"Loading CodeLlama model from {{model_path}}")
    model = MegatronGPTModel.restore_from(model_path)
    return model

def generate_swtbot_code(model, task_description, max_length=1024):
    """Generate SWTBot test code for a given task."""
    
    # Format input for CodeLlama
    system_prompt = "You are an expert Java developer specializing in test automation with SWTBot."
    user_prompt = f"Task: {{task_description}}"
    
    input_text = f"System: {{system_prompt}}\\nUser: {{user_prompt}}\\nAssistant: "
    
    # Generate code
    response = model.generate(
        input_text,
        max_length=max_length,
        temperature=0.2,  # Lower temperature for more deterministic code
        top_p=0.9,
        do_sample=True
    )
    
    return response

def main():
    parser = argparse.ArgumentParser(description="CodeLlama SWTBot Assistant")
    parser.add_argument("--model-path", default="{deployment_path}/model.nemo", help="Path to model")
    parser.add_argument("--task", required=True, help="SWTBot task description")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum generation length")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Generate code
    code = generate_swtbot_code(model, args.task, args.max_length)
    
    print("Generated SWTBot Test Code:")
    print("=" * 50)
    print(code)

if __name__ == "__main__":
    main()
'''
    
    # Save specialized inference script
    inference_path = deployment_path / "swtbot_assistant.py"
    with open(inference_path, 'w', encoding='utf-8') as f:
        f.write(inference_script)
    
    # Make executable
    import os
    os.chmod(inference_path, 0o755)
    
    logger.info(f"CodeLlama model deployed: {deployment_path}")
    logger.info(f"Specialized inference script: {inference_path}")
    
    return deployment_path


def test_code_generation(deployment_path: Path):
    """Test the deployed CodeLlama model."""
    logger.info("Testing code generation...")
    
    # Sample test tasks
    test_tasks = [
        "Create a test to click a menu item 'File' -> 'New' -> 'Project'",
        "Write a test to verify that a table has at least 5 rows",
        "Create a test to select text in a text editor and copy it"
    ]
    
    logger.info("Sample code generation tasks:")
    for i, task in enumerate(test_tasks, 1):
        logger.info(f"{i}. {task}")
    
    logger.info(f"Use the inference script at {deployment_path}/swtbot_assistant.py to test these tasks")
    
    # Example usage command
    example_cmd = f"""
python {deployment_path}/swtbot_assistant.py \\
  --task "Create a test to click a button with label 'Save' and verify success message" \\
  --max-length 512
"""
    
    logger.info(f"Example usage:\n{example_cmd}")


def main():
    """Run CodeLlama fine-tuning example."""
    logger.info("Starting CodeLlama fine-tuning example...")
    
    try:
        # Create code training data
        train_file = create_code_training_data()
        
        # Setup CodeLlama configuration
        config_path = setup_codellama_config()
        
        # Train model
        model_path = train_codellama_model(config_path, train_file)
        
        # Evaluate model
        eval_results, code_metrics = evaluate_code_model(model_path, train_file)
        
        # Deploy model
        deployment_path = deploy_code_model(model_path)
        
        # Test code generation
        test_code_generation(deployment_path)
        
        logger.info("CodeLlama fine-tuning example completed successfully!")
        logger.info(f"Model: {model_path}")
        logger.info(f"Deployment: {deployment_path}")
        
    except Exception as e:
        logger.error(f"CodeLlama example failed: {e}")
        raise


if __name__ == "__main__":
    main()
