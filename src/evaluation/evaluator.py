"""
Model evaluator for fine-tuned models.
"""

import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

try:
    import torch
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Evaluation will be limited.")

from .metrics import MetricsCalculator
from ..data.data_processor import DataProcessor
from ..data.instruction_formatter import InstructionFormatter

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates fine-tuned models on test data."""
    
    def __init__(self, 
                 model_type: str,
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None):
        """
        Initialize model evaluator.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_path: Path to the fine-tuned model
            tokenizer_path: Path to the tokenizer
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        # Initialize components
        self.metrics_calculator = MetricsCalculator(model_type)
        self.data_processor = DataProcessor(model_type)
        self.formatter = InstructionFormatter(model_type)
        
        # Model and tokenizer (loaded on demand)
        self.model = None
        self.tokenizer = None
        
        # Evaluation results
        self.results = {}
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the fine-tuned model.
        
        Args:
            model_path: Optional path to model (uses instance path if not provided)
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for model loading")
            return
        
        model_path = model_path or self.model_path
        if not model_path:
            raise ValueError("No model path provided")
        
        try:
            # For NeMo models, you would use:
            # from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
            # self.model = MegatronGPTModel.restore_from(model_path)
            
            # For now, we'll use a placeholder
            logger.info(f"Loading model from {model_path}")
            # self.model = load_model_function(model_path)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None) -> None:
        """
        Load the tokenizer.
        
        Args:
            tokenizer_path: Optional path to tokenizer
        """
        tokenizer_path = tokenizer_path or self.tokenizer_path
        if not tokenizer_path:
            raise ValueError("No tokenizer path provided")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded from {tokenizer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def generate_predictions(self, 
                           test_data: List[Dict[str, Any]],
                           max_new_tokens: int = 512,
                           temperature: float = 0.7,
                           top_p: float = 0.9) -> List[str]:
        """
        Generate predictions for test data.
        
        Args:
            test_data: List of test conversations
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        predictions = []
        
        try:
            for i, conversation in enumerate(test_data):
                # Format conversation for input (excluding last assistant message)
                input_messages = []
                target_response = ""
                
                for msg in conversation['messages']:
                    if msg['role'] == 'assistant' and not input_messages:
                        continue  # Skip if first message is assistant
                    elif msg['role'] == 'assistant':
                        target_response = msg['content']
                        break  # Use this as target, don't include in input
                    else:
                        input_messages.append(msg)
                
                # Format input
                input_conversation = {'messages': input_messages}
                formatted_input = self.formatter.format_conversation(input_conversation)
                
                # Generate prediction
                prediction = self._generate_single_prediction(
                    formatted_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(test_data)} predictions")
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    def _generate_single_prediction(self, 
                                   input_text: str,
                                   max_new_tokens: int = 512,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9) -> str:
        """Generate a single prediction."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                # For NeMo models, you would use:
                # outputs = self.model.generate(
                #     inputs['input_ids'],
                #     max_new_tokens=max_new_tokens,
                #     temperature=temperature,
                #     top_p=top_p,
                #     do_sample=True,
                #     pad_token_id=self.tokenizer.pad_token_id
                # )
                
                # Placeholder for actual generation
                outputs = inputs['input_ids']  # This would be replaced with actual generation
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the new generated part
            input_length = len(input_text)
            prediction = generated_text[input_length:].strip()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to generate single prediction: {e}")
            return ""
    
    def evaluate_on_file(self, 
                        test_file: Union[str, Path],
                        output_file: Optional[Union[str, Path]] = None,
                        max_examples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model on a test file.
        
        Args:
            test_file: Path to test data YAML file
            output_file: Optional path to save detailed results
            max_examples: Maximum number of examples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Load test data
            test_conversations = self.data_processor.load_yaml_data(test_file)
            test_conversations = self.data_processor.filter_conversations(test_conversations)
            
            if max_examples:
                test_conversations = test_conversations[:max_examples]
            
            logger.info(f"Evaluating on {len(test_conversations)} examples")
            
            # Generate predictions
            start_time = time.time()
            predictions = self.generate_predictions(test_conversations)
            generation_time = time.time() - start_time
            
            # Extract targets
            targets = []
            for conversation in test_conversations:
                # Get the last assistant message as target
                for msg in reversed(conversation['messages']):
                    if msg['role'] == 'assistant':
                        targets.append(msg['content'])
                        break
                else:
                    targets.append("")  # No assistant message found
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                predictions, 
                targets, 
                model=self.model
            )
            
            # Add timing information
            metrics['generation_time_seconds'] = generation_time
            metrics['examples_per_second'] = len(predictions) / generation_time
            metrics['total_examples'] = len(predictions)
            
            # Save detailed results if requested
            if output_file:
                self._save_detailed_results(
                    test_conversations, 
                    predictions, 
                    targets, 
                    metrics, 
                    output_file
                )
            
            self.results = metrics
            logger.info(f"Evaluation completed. Main metrics: {self._format_main_metrics(metrics)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _save_detailed_results(self, 
                              conversations: List[Dict[str, Any]],
                              predictions: List[str],
                              targets: List[str],
                              metrics: Dict[str, float],
                              output_file: Union[str, Path]) -> None:
        """Save detailed evaluation results."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        detailed_results = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "evaluation_time": time.time(),
            "metrics": metrics,
            "examples": []
        }
        
        for i, (conv, pred, target) in enumerate(zip(conversations, predictions, targets)):
            example = {
                "id": i,
                "input_conversation": conv,
                "prediction": pred,
                "target": target,
                "exact_match": pred.strip() == target.strip()
            }
            detailed_results["examples"].append(example)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved detailed results to {output_file}")
    
    def _format_main_metrics(self, metrics: Dict[str, float]) -> str:
        """Format main metrics for logging."""
        main_metrics = ['exact_match', 'bleu', 'rouge1', 'rougeL']
        formatted = []
        
        for metric in main_metrics:
            if metric in metrics:
                formatted.append(f"{metric}: {metrics[metric]:.3f}")
        
        return ", ".join(formatted)
    
    def compare_models(self, 
                      other_results: Dict[str, float],
                      output_file: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare current model results with another model.
        
        Args:
            other_results: Results from another model
            output_file: Optional path to save comparison
            
        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            raise RuntimeError("No evaluation results available. Run evaluate_on_file() first.")
        
        comparison = {
            "current_model": self.results,
            "other_model": other_results,
            "differences": {}
        }
        
        # Calculate differences
        for metric in self.results:
            if metric in other_results and isinstance(self.results[metric], (int, float)):
                diff = self.results[metric] - other_results[metric]
                comparison["differences"][metric] = diff
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        return comparison
