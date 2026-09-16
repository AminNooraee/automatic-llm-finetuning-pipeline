"""Context/question/answer schema adapter for RAG-oriented data."""

from .base import FieldMappingAdapter


class RAGQAAdapter(FieldMappingAdapter):
    name = "rag_qa"
    aliases = ("context_qa",)
    required_fields = ("context", "question", "answer")
    output_mapping = {
        "instruction": "question",
        "input": "context",
        "output": "answer",
    }
    detection_confidence = 1.0
