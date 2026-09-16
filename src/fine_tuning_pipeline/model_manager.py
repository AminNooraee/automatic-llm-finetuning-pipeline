"""Model-family detection and LLaMA-Factory template compatibility."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


class ModelCompatibilityError(ValueError):
    """Raised when a model cannot be mapped safely to a supported template."""


@dataclass(frozen=True)
class ModelCompatibility:
    model_name: str
    family: str
    variant: str
    template: str
    detection_source: str
    warnings: tuple[str, ...] = ()
    config_model_type: str | None = None


@dataclass(frozen=True)
class _ModelMatch:
    family: str
    variant: str
    template: str | None
    warnings: tuple[str, ...] = ()


_SUPPORTED_TEMPLATES = {
    "qwen": {"qwen"},
    "llama": {"llama2", "llama3"},
    "mistral": {"mistral"},
    "gemma": {"gemma", "gemma2", "gemma3"},
}


@lru_cache(maxsize=32)
def _load_huggingface_config(model_name: str) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise ModelCompatibilityError(
            "Model inspection requires the 'transformers' package used by "
            "LLaMA-Factory. Install the project training dependencies first."
        ) from error
    return AutoConfig.from_pretrained(model_name)


def resolve_model_compatibility(
    model_name: str,
    *,
    requested_template: str | None = None,
    config_loader: Callable[[str], Any] | None = None,
) -> ModelCompatibility:
    """Detect a supported model family and resolve its training template."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelCompatibilityError("model.name must be a non-empty string")

    model_name = model_name.strip()
    name_match = _detect_from_name(model_name)
    config_match: _ModelMatch | None = None
    config_model_type: str | None = None
    config_error: Exception | None = None
    config_loaded = False
    loader = config_loader or _load_huggingface_config

    try:
        model_config = loader(model_name)
        config_loaded = True
        config_model_type = _config_value(model_config, "model_type")
        config_match = _detect_from_config(model_config, model_name)
    except ModelCompatibilityError:
        raise
    except Exception as error:
        config_error = error

    warnings: list[str] = []
    if name_match is not None and config_match is not None:
        if name_match.family != config_match.family:
            raise ModelCompatibilityError(
                f"Model name {model_name!r} looks like {name_match.family}, but its "
                f"HuggingFace config reports {config_match.family} "
                f"(model_type={config_model_type!r}). Refusing to select a template."
            )
        if (
            name_match.template is not None
            and config_match.template is not None
            and name_match.template != config_match.template
        ):
            raise ModelCompatibilityError(
                f"Model name {model_name!r} implies template {name_match.template!r}, "
                f"but its HuggingFace config implies {config_match.template!r}."
            )
        selected = _ModelMatch(
            family=config_match.family,
            variant=(
                name_match.variant
                if name_match.variant != name_match.family
                else config_match.variant
            ),
            template=name_match.template or config_match.template,
            warnings=(*name_match.warnings, *config_match.warnings),
        )
        detection_source = "name+config"
    elif config_match is not None:
        selected = config_match
        detection_source = "config"
    elif name_match is not None and config_loaded:
        raise ModelCompatibilityError(
            f"Model name {model_name!r} looks like {name_match.family}, but its "
            f"HuggingFace config has unsupported model_type={config_model_type!r}."
        )
    elif name_match is not None:
        selected = name_match
        detection_source = "name"
        if config_error is not None:
            warnings.append(
                "HuggingFace config inspection was unavailable; model compatibility "
                f"was resolved from the model name only ({_short_error(config_error)})."
            )
    else:
        if config_error is not None:
            raise ModelCompatibilityError(
                f"Unable to recognize model family from {model_name!r}, and its "
                f"HuggingFace config could not be loaded: {_short_error(config_error)}"
            ) from config_error
        raise ModelCompatibilityError(
            f"Unsupported model {model_name!r} (model_type={config_model_type!r}). "
            "Supported families: Qwen/Qwen2/Qwen2.5, Llama 2/3, Mistral, and Gemma."
        )

    if selected.template is None:
        raise ModelCompatibilityError(
            f"Detected the {selected.family} family for {model_name!r}, but could not "
            "determine a compatible model generation/template."
        )

    _validate_template(selected.family, selected.template, model_name)
    if requested_template is not None:
        if not isinstance(requested_template, str) or not requested_template.strip():
            raise ModelCompatibilityError("model.template must be a non-empty string")
        normalized_template = requested_template.strip().lower().replace("-", "_")
        if normalized_template != selected.template:
            raise ModelCompatibilityError(
                f"Model {model_name!r} resolves to the {selected.template!r} template, "
                f"but config requested {requested_template!r}. Remove model.template "
                "to use automatic selection."
            )

    warnings.extend(selected.warnings)
    if not _looks_instruction_tuned(model_name):
        warnings.append(
            f"Model {model_name!r} does not look like an Instruct/Chat checkpoint. "
            "SFT is allowed, but confirm that a base checkpoint is intentional."
        )

    return ModelCompatibility(
        model_name=model_name,
        family=selected.family,
        variant=selected.variant,
        template=selected.template,
        detection_source=detection_source,
        warnings=tuple(dict.fromkeys(warnings)),
        config_model_type=config_model_type,
    )


def _detect_from_name(model_name: str) -> _ModelMatch | None:
    normalized = model_name.lower().replace("_", "-")

    if "qwen" in normalized:
        if re.search(r"qwen[- ]?2[.-]?5", normalized):
            variant = "qwen2.5"
        elif re.search(r"qwen[- ]?2", normalized):
            variant = "qwen2"
        else:
            variant = "qwen"
        return _ModelMatch("qwen", variant, "qwen")

    if "llama" in normalized:
        if re.search(r"llama[- .]?2(?:[- ./]|$)", normalized):
            return _ModelMatch("llama", "llama2", "llama2")
        if re.search(r"llama[- .]?3(?:[.-]\d+)?(?:[- ./]|$)", normalized):
            return _ModelMatch("llama", "llama3", "llama3")
        return _ModelMatch("llama", "llama", None)

    if "mistral" in normalized and "mixtral" not in normalized:
        return _ModelMatch("mistral", "mistral", "mistral")

    if "gemma" in normalized:
        if "gemma-3-" in normalized:
            return _ModelMatch("gemma", "gemma3", "gemma3")
        if "gemma-2-" in normalized:
            return _ModelMatch("gemma", "gemma2", "gemma2")
        return _ModelMatch("gemma", "gemma", "gemma")

    return None


def _detect_from_config(model_config: Any, model_name: str) -> _ModelMatch | None:
    model_type_value = _config_value(model_config, "model_type")
    model_type = str(model_type_value).lower() if model_type_value is not None else ""

    if model_type in {"qwen", "qwen2", "qwen2_moe"}:
        name_match = _detect_from_name(model_name)
        variant = name_match.variant if name_match and name_match.family == "qwen" else model_type
        return _ModelMatch("qwen", variant, "qwen")

    if model_type == "llama":
        generation = _detect_llama_generation(model_config, model_name)
        return _ModelMatch("llama", generation[0], generation[1], generation[2])

    if model_type == "mistral":
        return _ModelMatch("mistral", "mistral", "mistral")

    gemma_types = {
        "gemma": ("gemma", "gemma"),
        "gemma2": ("gemma2", "gemma2"),
        "gemma3": ("gemma3", "gemma3"),
        "gemma3_text": ("gemma3", "gemma3"),
    }
    if model_type in gemma_types:
        variant, template = gemma_types[model_type]
        return _ModelMatch("gemma", variant, template)

    architectures = _config_value(model_config, "architectures") or []
    if isinstance(architectures, (list, tuple)):
        architecture_match = _detect_from_name(" ".join(str(item) for item in architectures))
        if architecture_match is not None and architecture_match.template is not None:
            return architecture_match

    return None


def _detect_llama_generation(
    model_config: Any, model_name: str
) -> tuple[str, str, tuple[str, ...]]:
    combined_name = " ".join(
        part
        for part in (
            model_name,
            str(_config_value(model_config, "_name_or_path") or ""),
        )
        if part
    )
    name_match = _detect_from_name(combined_name)
    if name_match is not None and name_match.template is not None:
        return name_match.variant, name_match.template, ()

    vocab_size = _config_value(model_config, "vocab_size")
    if isinstance(vocab_size, int) and vocab_size >= 100_000:
        return (
            "llama3",
            "llama3",
            ("Llama generation was inferred from its HuggingFace config vocabulary size.",),
        )
    if isinstance(vocab_size, int) and 30_000 <= vocab_size <= 40_000:
        return (
            "llama2",
            "llama2",
            ("Llama generation was inferred from its HuggingFace config vocabulary size.",),
        )

    raise ModelCompatibilityError(
        f"HuggingFace config identifies {model_name!r} as Llama, but does not "
        "distinguish Llama 2 from Llama 3 safely. Use an official model identifier."
    )


def _validate_template(family: str, template: str, model_name: str) -> None:
    allowed = _SUPPORTED_TEMPLATES.get(family, set())
    if template not in allowed:
        raise ModelCompatibilityError(
            f"Template {template!r} is incompatible with detected {family} model "
            f"{model_name!r}. Allowed templates: {sorted(allowed)}"
        )


def _config_value(model_config: Any, key: str) -> Any:
    if isinstance(model_config, dict):
        return model_config.get(key)
    return getattr(model_config, key, None)


def _looks_instruction_tuned(model_name: str) -> bool:
    normalized = model_name.lower()
    markers = ("instruct", "chat", "-it", "_it", "assistant")
    return any(marker in normalized for marker in markers)


def _short_error(error: Exception) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return message[:240]
