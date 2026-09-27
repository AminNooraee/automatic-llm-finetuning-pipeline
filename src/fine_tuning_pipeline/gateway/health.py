"""Gateway endpoint verification reuses the OpenAI-compatible contract."""

from ..serving.health import OpenAIEndpointVerifier

__all__ = ["OpenAIEndpointVerifier"]
