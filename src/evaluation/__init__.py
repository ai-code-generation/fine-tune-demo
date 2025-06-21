try:
    # Try relative imports first (when used as a package)
    from .metrics import MetricsCalculator
    from .evaluator import ModelEvaluator
except ImportError:
    # Fall back to absolute imports (when run directly)
    from evaluation.metrics import MetricsCalculator
    from evaluation.evaluator import ModelEvaluator

__all__ = ["MetricsCalculator", "ModelEvaluator"]
