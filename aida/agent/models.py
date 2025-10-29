"""Model registry and initialisation helpers for A.I.D.A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentModel:
    """Simple container describing the selected reasoning model."""

    name: str
    provider: str
    mode: str
    metadata: Dict[str, Any]


_AVAILABLE_MODELS: Dict[str, AgentModel] = {
    "llama3-8b": AgentModel(
        name="llama3-8b",
        provider="transformers",
        mode="LLM",
        metadata={"description": "Meta Llama 3 8B Instruct"},
    ),
    "llama3-70b": AgentModel(
        name="llama3-70b",
        provider="transformers",
        mode="LLM",
        metadata={"description": "Meta Llama 3 70B Instruct"},
    ),
    "mistral-7b": AgentModel(
        name="mistral-7b",
        provider="transformers",
        mode="LLM",
        metadata={"description": "Mistral 7B Instruct"},
    ),
    "mistral-8x22b": AgentModel(
        name="mistral-8x22b",
        provider="transformers",
        mode="LLM",
        metadata={"description": "Mistral Large (8x22B)"},
    ),
    "phi3-mini": AgentModel(
        name="phi3-mini",
        provider="transformers",
        mode="SLM",
        metadata={"description": "Phi-3 Mini 3.8B"},
    ),
    "phi3-medium": AgentModel(
        name="phi3-medium",
        provider="transformers",
        mode="SLM",
        metadata={"description": "Phi-3 Medium"},
    ),
    "gemma-2b": AgentModel(
        name="gemma-2b",
        provider="transformers",
        mode="SLM",
        metadata={"description": "Gemma 2B"},
    ),
    "gemma-7b": AgentModel(
        name="gemma-7b",
        provider="transformers",
        mode="SLM",
        metadata={"description": "Gemma 7B"},
    ),
}


class UnknownModelError(ValueError):
    """Raised when attempting to use an unsupported model name."""


def initialize_llm(model_name: str, **kwargs: Any) -> AgentModel:
    """Return the model configuration for ``model_name``.

    The function does not instantiate heavy weight model objects. Instead, it returns a
    lightweight :class:`AgentModel` describing how the agent should configure itself.
    """

    try:
        model = _AVAILABLE_MODELS[model_name]
    except KeyError as exc:  # pragma: no cover - defensive message formatting
        raise UnknownModelError(f"Unknown model: {model_name}") from exc

    # merge additional keyword arguments into the metadata so downstream components have
    # access to custom parameters (e.g. temperature, max_tokens, API host, ...)
    if kwargs:
        merged_metadata = dict(model.metadata)
        merged_metadata.update(kwargs)
        return AgentModel(name=model.name, provider=model.provider, mode=model.mode, metadata=merged_metadata)
    return model


def available_models() -> Dict[str, AgentModel]:
    """Return a copy of the registered models."""

    return dict(_AVAILABLE_MODELS)
