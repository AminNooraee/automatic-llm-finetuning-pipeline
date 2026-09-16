"""Public API and default registry for dataset adaptation."""

from .alpaca_adapter import AlpacaAdapter
from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetAdapterError,
    DatasetConversionResult,
    DatasetFormatDetectionError,
    DatasetValidationError,
    DetectionCandidate,
    DetectionResult,
    OptionalDependencyError,
    UnsupportedDatasetTaskError,
)
from .chatml_adapter import ChatMLAdapter, ChatMLTextSourceAdapter
from .csv_adapter import CsvAdapter
from .dpo_adapter import DPOAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .instruction_response_adapter import InstructionResponseAdapter
from .json_adapter import JsonArrayAdapter
from .jsonl_adapter import JsonlAdapter
from .openai_chat_adapter import OpenAIChatAdapter
from .parquet_adapter import ParquetAdapter
from .prompt_completion_adapter import PromptCompletionAdapter
from .qa_adapter import QAAdapter
from .rag_qa_adapter import RAGQAAdapter
from .registry import AdapterRegistry, DatasetAdapterFramework
from .sharegpt_adapter import ShareGPTAdapter


def create_default_framework() -> DatasetAdapterFramework:
    source_registry = AdapterRegistry(AdapterStage.SOURCE)
    for adapter in (
        JsonArrayAdapter(),
        JsonlAdapter(),
        CsvAdapter(),
        ParquetAdapter(),
        ChatMLTextSourceAdapter(),
        HuggingFaceAdapter(),
    ):
        source_registry.register(adapter)

    schema_registry = AdapterRegistry(AdapterStage.SCHEMA)
    for adapter in (
        AlpacaAdapter(),
        InstructionResponseAdapter(),
        PromptCompletionAdapter(),
        QAAdapter(),
        RAGQAAdapter(),
        OpenAIChatAdapter(),
        ShareGPTAdapter(),
        ChatMLAdapter(),
        DPOAdapter(),
    ):
        schema_registry.register(adapter)

    return DatasetAdapterFramework(source_registry, schema_registry)


DEFAULT_FRAMEWORK = create_default_framework()


__all__ = [
    "AdapterContext",
    "AdapterRegistry",
    "AdapterStage",
    "BaseDatasetAdapter",
    "DEFAULT_FRAMEWORK",
    "DatasetAdapterError",
    "DatasetAdapterFramework",
    "DatasetConversionResult",
    "DatasetFormatDetectionError",
    "DatasetValidationError",
    "DetectionCandidate",
    "DetectionResult",
    "OptionalDependencyError",
    "UnsupportedDatasetTaskError",
    "create_default_framework",
]
