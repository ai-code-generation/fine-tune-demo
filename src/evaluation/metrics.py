"""
Evaluation metrics for fine-tuned models.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import logging

try:
    from rouge_score import rouge_scorer
    from bleu import sentence_bleu, SmoothingFunction
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Some evaluation metrics not available. Install rouge-score, bleu, and evaluate packages.")

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates various evaluation metrics for model outputs."""
    
    def __init__(self, model_type: str = "llama2"):
        """
        Initialize metrics calculator.
        
        Args:
            model_type: Type of model for specialized metrics
        """
        self.model_type = model_type.lower()
        
        # Initialize metric calculators
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            self.bleu_smoother = SmoothingFunction().method1
        
        # Code-specific metrics for CodeLlama
        self.is_code_model = self.model_type == "codellama"
    
    def calculate_perplexity(self, 
                           predictions: List[str], 
                           targets: List[str],
                           model=None) -> float:
        """
        Calculate perplexity score.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            model: Model instance for loss calculation
            
        Returns:
            Perplexity score
        """
        if model is None:
            logger.warning("Model not provided, cannot calculate true perplexity")
            return float('inf')
        
        try:
            # This would require actual model inference
            # For now, return a placeholder
            total_loss = 0.0
            total_tokens = 0
            
            # In a real implementation, you would:
            # 1. Tokenize targets
            # 2. Calculate model loss on targets
            # 3. Compute perplexity = exp(loss)
            
            # Placeholder calculation
            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = math.exp(avg_loss)
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Failed to calculate perplexity: {e}")
            return float('inf')
    
    def calculate_bleu(self, 
                      predictions: List[str], 
                      targets: List[str]) -> Dict[str, float]:
        """
        Calculate BLEU scores.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            
        Returns:
            Dictionary with BLEU scores
        """
        if not METRICS_AVAILABLE:
            logger.warning("BLEU calculation not available")
            return {"bleu": 0.0}
        
        try:
            bleu_scores = []
            
            for pred, target in zip(predictions, targets):
                # Tokenize
                pred_tokens = pred.split()
                target_tokens = [target.split()]  # List of reference lists
                
                # Calculate BLEU
                bleu = sentence_bleu(
                    target_tokens, 
                    pred_tokens, 
                    smoothing_function=self.bleu_smoother
                )
                bleu_scores.append(bleu)
            
            avg_bleu = np.mean(bleu_scores)
            
            return {
                "bleu": avg_bleu,
                "bleu_std": np.std(bleu_scores)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate BLEU: {e}")
            return {"bleu": 0.0}
    
    def calculate_rouge(self, 
                       predictions: List[str], 
                       targets: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not METRICS_AVAILABLE:
            logger.warning("ROUGE calculation not available")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            
            for pred, target in zip(predictions, targets):
                scores = self.rouge_scorer.score(target, pred)
                
                rouge_scores["rouge1"].append(scores['rouge1'].fmeasure)
                rouge_scores["rouge2"].append(scores['rouge2'].fmeasure)
                rouge_scores["rougeL"].append(scores['rougeL'].fmeasure)
            
            # Calculate averages
            result = {}
            for metric, scores in rouge_scores.items():
                result[metric] = np.mean(scores)
                result[f"{metric}_std"] = np.std(scores)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate ROUGE: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_exact_match(self, 
                             predictions: List[str], 
                             targets: List[str]) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            
        Returns:
            Exact match accuracy
        """
        try:
            exact_matches = 0
            total = len(predictions)
            
            for pred, target in zip(predictions, targets):
                if pred.strip() == target.strip():
                    exact_matches += 1
            
            accuracy = exact_matches / total if total > 0 else 0.0
            return accuracy
            
        except Exception as e:
            logger.error(f"Failed to calculate exact match: {e}")
            return 0.0
    
    def calculate_code_metrics(self, 
                              predictions: List[str], 
                              targets: List[str]) -> Dict[str, float]:
        """
        Calculate code-specific metrics (for CodeLlama).
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            
        Returns:
            Dictionary with code metrics
        """
        if not self.is_code_model:
            return {}
        
        try:
            metrics = {}
            
            # Code BLEU (simplified version)
            code_bleu_scores = []
            for pred, target in zip(predictions, targets):
                # Extract code blocks
                pred_code = self._extract_code_blocks(pred)
                target_code = self._extract_code_blocks(target)
                
                if pred_code and target_code:
                    # Calculate token-level similarity
                    pred_tokens = self._tokenize_code(pred_code[0])
                    target_tokens = self._tokenize_code(target_code[0])
                    
                    # Simple overlap-based metric
                    overlap = len(set(pred_tokens) & set(target_tokens))
                    total = len(set(pred_tokens) | set(target_tokens))
                    score = overlap / total if total > 0 else 0.0
                    code_bleu_scores.append(score)
                else:
                    code_bleu_scores.append(0.0)
            
            metrics["code_bleu"] = np.mean(code_bleu_scores)
            
            # Syntax validity (simplified check)
            syntax_valid = []
            for pred in predictions:
                code_blocks = self._extract_code_blocks(pred)
                if code_blocks:
                    is_valid = self._check_syntax_validity(code_blocks[0])
                    syntax_valid.append(is_valid)
                else:
                    syntax_valid.append(False)
            
            metrics["syntax_validity"] = np.mean(syntax_valid)
            
            # Function/method extraction accuracy
            function_accuracy = []
            for pred, target in zip(predictions, targets):
                pred_functions = self._extract_functions(pred)
                target_functions = self._extract_functions(target)
                
                if target_functions:
                    overlap = len(set(pred_functions) & set(target_functions))
                    accuracy = overlap / len(target_functions)
                    function_accuracy.append(accuracy)
                else:
                    function_accuracy.append(1.0 if not pred_functions else 0.0)
            
            metrics["function_accuracy"] = np.mean(function_accuracy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate code metrics: {e}")
            return {}
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        # Find code blocks marked with ```
        pattern = r'```(?:\w+)?\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Simple code tokenization."""
        # Split on common code delimiters
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [token.lower() for token in tokens if token.strip()]
    
    def _check_syntax_validity(self, code: str) -> bool:
        """Check if code has basic syntax validity."""
        try:
            # Simple checks for common syntax elements
            brackets = {'(': ')', '[': ']', '{': '}'}
            stack = []
            
            for char in code:
                if char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return False
                    last = stack.pop()
                    if brackets[last] != char:
                        return False
            
            return len(stack) == 0
            
        except Exception:
            return False
    
    def _extract_functions(self, text: str) -> List[str]:
        """Extract function/method names from code."""
        # Simple regex for function definitions
        patterns = [
            r'def\s+(\w+)\s*\(',  # Python
            r'function\s+(\w+)\s*\(',  # JavaScript
            r'public\s+\w+\s+(\w+)\s*\(',  # Java
            r'(\w+)\s*\([^)]*\)\s*{',  # C/C++
        ]
        
        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            functions.extend(matches)
        
        return list(set(functions))
    
    def calculate_all_metrics(self, 
                             predictions: List[str], 
                             targets: List[str],
                             model=None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            model: Optional model for perplexity calculation
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics["exact_match"] = self.calculate_exact_match(predictions, targets)
            
            # BLEU scores
            bleu_scores = self.calculate_bleu(predictions, targets)
            metrics.update(bleu_scores)
            
            # ROUGE scores
            rouge_scores = self.calculate_rouge(predictions, targets)
            metrics.update(rouge_scores)
            
            # Perplexity (if model provided)
            if model is not None:
                metrics["perplexity"] = self.calculate_perplexity(predictions, targets, model)
            
            # Code-specific metrics
            if self.is_code_model:
                code_metrics = self.calculate_code_metrics(predictions, targets)
                metrics.update(code_metrics)
            
            logger.info(f"Calculated {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {"error": str(e)}
