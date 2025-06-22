#!/usr/bin/env python3
"""
Evaluation script for fine-tuned CodeLlama and Llama3 models using NeMo Framework.
This script provides various evaluation methods including perplexity, code generation, and custom metrics.
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from configs.model_configs import get_model_config, list_available_models
from data_preprocessing import load_yaml_data, extract_input_output_pairs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluation class for fine-tuned models."""
    
    def __init__(self, model_path: str, model_name: str, output_dir: str = "./evaluation_results"):
        self.model_path = model_path
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get model configuration
        self.model_config = get_model_config(model_name)
    
    def evaluate_perplexity(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate model perplexity on test data."""
        logger.info(f"Evaluating perplexity on {test_data_path}")
        
        # Create evaluation config
        eval_config = self.create_eval_config(test_data_path, "perplexity")
        
        cmd = [
            "python", "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py",
            f"--config-path={os.path.dirname(eval_config)}",
            f"--config-name={os.path.basename(eval_config)}"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse results from output
            perplexity_results = self.parse_perplexity_results(result.stdout)
            
            # Save results
            results_file = self.output_dir / f"{self.model_name}_perplexity_results.json"
            with open(results_file, 'w') as f:
                json.dump(perplexity_results, f, indent=2)
            
            logger.info(f"Perplexity evaluation completed. Results saved to {results_file}")
            return perplexity_results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            raise
    
    def evaluate_generation(self, test_data_path: str, tokens_to_generate: int = 100) -> List[Dict[str, str]]:
        """Evaluate model text generation capabilities."""
        logger.info(f"Evaluating text generation on {test_data_path}")
        
        # Create evaluation config for generation
        eval_config = self.create_eval_config(test_data_path, "generation", tokens_to_generate)
        
        cmd = [
            "python", "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py",
            f"--config-path={os.path.dirname(eval_config)}",
            f"--config-name={os.path.basename(eval_config)}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Read generated outputs
            output_file = self.output_dir / f"{self.model_name}_generation_results.jsonl"
            generations = self.parse_generation_results(str(output_file))
            
            logger.info(f"Generation evaluation completed. Results saved to {output_file}")
            return generations
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Generation evaluation failed: {e}")
            raise
    
    def create_eval_config(self, test_data_path: str, eval_type: str, tokens_to_generate: int = 100) -> str:
        """Create evaluation configuration file."""
        config = {
            "name": f"{self.model_name}_{eval_type}_eval",
            "trainer": {
                "devices": self.model_config["devices"],
                "num_nodes": 1,
                "accelerator": "gpu",
                "precision": "bf16",
                "logger": False,
                "enable_checkpointing": False,
                "use_distributed_sampler": False,
            },
            "model": {
                "restore_from_path": self.model_path,
                "tensor_model_parallel_size": self.model_config["tensor_model_parallel_size"],
                "pipeline_model_parallel_size": self.model_config["pipeline_model_parallel_size"],
                "global_batch_size": 8,
                "micro_batch_size": 1,
                "data": {
                    "test_ds": {
                        "file_names": [test_data_path],
                        "names": [f"{self.model_name}_test_set"],
                        "global_batch_size": 8,
                        "micro_batch_size": 1,
                        "tokens_to_generate": tokens_to_generate,
                        "output_file_path_prefix": str(self.output_dir / f"{self.model_name}_{eval_type}_results"),
                        "write_predictions_to_file": True,
                        "shuffle": False,
                        "num_workers": 0,
                        "pin_memory": True,
                        "max_seq_length": 2048,
                        "min_seq_length": 1,
                        "drop_last": False,
                        "label_key": "output",
                        "add_eos": True,
                        "add_sep": False,
                        "add_bos": False,
                        "truncation_field": "input",
                        "index_mapping_dir": None,
                        "data_impl": "jsonl"
                    }
                }
            },
            "inference": {
                "greedy": True,
                "top_k": 0,
                "top_p": 0.9,
                "temperature": 1.0,
                "add_BOS": False,
                "tokens_to_generate": tokens_to_generate,
                "all_probs": False,
                "repetition_penalty": 1.2,
                "min_tokens_to_generate": 1,
            }
        }
        
        # Save config
        config_file = self.output_dir / f"{self.model_name}_{eval_type}_eval_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_file)
    
    def parse_perplexity_results(self, output: str) -> Dict[str, float]:
        """Parse perplexity results from evaluation output."""
        results = {}
        
        # Look for perplexity values in the output
        lines = output.split('\n')
        for line in lines:
            if 'perplexity' in line.lower():
                # Extract numerical value
                parts = line.split()
                for i, part in enumerate(parts):
                    try:
                        value = float(part)
                        results['perplexity'] = value
                        break
                    except ValueError:
                        continue
        
        return results
    
    def parse_generation_results(self, results_file: str) -> List[Dict[str, str]]:
        """Parse generation results from JSONL file."""
        generations = []
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        generations.append(data)
        
        return generations
    
    def evaluate_code_quality(self, generations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate code quality metrics for generated code."""
        logger.info("Evaluating code quality metrics")
        
        metrics = {
            "total_samples": len(generations),
            "syntax_valid": 0,
            "avg_length": 0,
            "contains_imports": 0,
            "contains_functions": 0,
            "contains_classes": 0,
        }
        
        total_length = 0
        
        for gen in generations:
            generated_text = gen.get('generated_text', '')
            total_length += len(generated_text)
            
            # Check for syntax validity (basic heuristics)
            if self.is_likely_valid_code(generated_text):
                metrics["syntax_valid"] += 1
            
            # Check for code patterns
            if 'import ' in generated_text or 'from ' in generated_text:
                metrics["contains_imports"] += 1
            
            if 'def ' in generated_text:
                metrics["contains_functions"] += 1
            
            if 'class ' in generated_text:
                metrics["contains_classes"] += 1
        
        if metrics["total_samples"] > 0:
            metrics["avg_length"] = total_length / metrics["total_samples"]
            metrics["syntax_valid_rate"] = metrics["syntax_valid"] / metrics["total_samples"]
            metrics["imports_rate"] = metrics["contains_imports"] / metrics["total_samples"]
            metrics["functions_rate"] = metrics["contains_functions"] / metrics["total_samples"]
            metrics["classes_rate"] = metrics["contains_classes"] / metrics["total_samples"]
        
        return metrics
    
    def is_likely_valid_code(self, code: str) -> bool:
        """Basic heuristic to check if generated text looks like valid code."""
        # Simple checks for code-like patterns
        code_indicators = [
            '=', '(', ')', '{', '}', '[', ']', ';', 
            'def ', 'class ', 'if ', 'for ', 'while ', 'import ', 'from '
        ]
        
        indicator_count = sum(1 for indicator in code_indicators if indicator in code)
        return indicator_count >= 3  # Arbitrary threshold
    
    def run_comprehensive_evaluation(self, test_data_path: str, tokens_to_generate: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation including perplexity and generation quality."""
        logger.info("Starting comprehensive evaluation")
        
        results = {}
        
        try:
            # Evaluate perplexity
            perplexity_results = self.evaluate_perplexity(test_data_path)
            results["perplexity"] = perplexity_results
            
            # Evaluate generation
            generations = self.evaluate_generation(test_data_path, tokens_to_generate)
            results["generation_samples"] = len(generations)
            
            # Evaluate code quality
            code_quality = self.evaluate_code_quality(generations)
            results["code_quality"] = code_quality
            
            # Save comprehensive results
            results_file = self.output_dir / f"{self.model_name}_comprehensive_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Comprehensive evaluation completed. Results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CodeLlama/Llama3 models")
    parser.add_argument("--model-path", "-m", required=True, help="Path to the fine-tuned model (.nemo file)")
    parser.add_argument("--model-name", "-n", required=True, choices=list_available_models(),
                       help="Model name/type")
    parser.add_argument("--test-data", "-d", required=True, help="Path to test data (JSONL format)")
    parser.add_argument("--output-dir", "-o", default="./evaluation_results", help="Output directory")
    parser.add_argument("--eval-type", choices=["perplexity", "generation", "comprehensive"], 
                       default="comprehensive", help="Type of evaluation to run")
    parser.add_argument("--tokens-to-generate", type=int, default=100, 
                       help="Number of tokens to generate for generation evaluation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model file {args.model_path} does not exist")
        return 1
    
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file {args.test_data} does not exist")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_path, args.model_name, args.output_dir)
        
        # Run evaluation
        if args.eval_type == "perplexity":
            results = evaluator.evaluate_perplexity(args.test_data)
        elif args.eval_type == "generation":
            results = evaluator.evaluate_generation(args.test_data, args.tokens_to_generate)
        else:  # comprehensive
            results = evaluator.run_comprehensive_evaluation(args.test_data, args.tokens_to_generate)
        
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        if isinstance(results, dict):
            print("\nSummary:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
