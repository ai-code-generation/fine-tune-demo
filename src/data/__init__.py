try:
    # Try relative imports first (when used as a package)
    from .data_processor import DataProcessor
    from .instruction_formatter import InstructionFormatter
    from .dataset_builder import DatasetBuilder
except ImportError:
    # Fall back to absolute imports (when run directly)
    from data.data_processor import DataProcessor
    from data.instruction_formatter import InstructionFormatter
    from data.dataset_builder import DatasetBuilder

__all__ = ["DataProcessor", "InstructionFormatter", "DatasetBuilder"]
