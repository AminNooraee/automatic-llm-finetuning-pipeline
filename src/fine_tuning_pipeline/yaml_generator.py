"""Generate validated LLaMA-Factory training YAML files."""

from pathlib import Path

import yaml

from .training_config import TrainingConfig, resolve_training_config


def build_llamafactory_config(
    config,
    dataset_dir=None,
    output_dir=None,
    training_config=None,
):
    """Build a LLaMA-Factory configuration from validated public settings."""
    if dataset_dir is None:
        raise ValueError("dataset_dir must point to the current run's dataset directory")
    if output_dir is None:
        raise ValueError("output_dir must point to the current run's model directory")

    resolved_template = config["model"].get("template")
    if not isinstance(resolved_template, str) or not resolved_template.strip():
        raise ValueError(
            "model.template must be resolved by the model compatibility layer "
            "before generating training YAML"
        )

    resolved_training = training_config or resolve_training_config(config.get("training"))
    if not isinstance(resolved_training, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig instance")

    llama_config = {
        "model_name_or_path": config["model"]["name"],
        # These describe this pipeline's fixed SFT operation, not user-tunable
        # hyperparameters.
        "stage": "sft",
        "do_train": True,
        "dataset": config["dataset"]["name"],
        "dataset_dir": str(dataset_dir),
        "template": resolved_template,
        "output_dir": str(output_dir),
    }
    requested_revision = config["model"].get("revision")
    if requested_revision is not None:
        llama_config["model_revision"] = requested_revision
    llama_config.update(resolved_training.to_llamafactory_args())
    # LLaMA-Factory's model loader owns checkpointing and enables it by default.
    # The generic Trainer flag alone does not disable that loader behavior.
    llama_config.pop("gradient_checkpointing")
    llama_config["disable_gradient_checkpointing"] = not resolved_training.gradient_checkpointing
    return llama_config


def generate_training_yaml(
    config,
    dataset_dir=None,
    output_dir=None,
    output_file=None,
    training_config=None,
):
    """Write a validated LLaMA-Factory training configuration."""
    if output_file is None:
        raise ValueError("output_file must belong to the current run's config directory")

    llama_config = build_llamafactory_config(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        training_config=training_config,
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(llama_config, file, sort_keys=False)

    print(f"LLaMA-Factory config created: {output_file}")
    return str(output_file)


def generate_lora_yaml(
    config,
    dataset_dir=None,
    output_dir=None,
    output_file=None,
    training_config=None,
):
    """Backward-compatible alias for callers using the original function name."""
    return generate_training_yaml(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        output_file=output_file,
        training_config=training_config,
    )
