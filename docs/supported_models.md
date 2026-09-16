# Supported models

Support means safe mapping to a documented LLaMA-Factory family/template, not
unlimited coverage of all HuggingFace models or independently validated training
of every checkpoint variant. The backend and chosen hardware must support the
actual architecture, precision, and tokenizer.

| Family/generation | Template | Representative identifier |
| --- | --- | --- |
| Qwen / Qwen2 / Qwen2.5 | `qwen` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Llama 2 | `llama2` | `meta-llama/Llama-2-7b-chat-hf` |
| Llama 3 | `llama3` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral | `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma | `gemma` | `google/gemma-2b-it` |
| Gemma 2 | `gemma2` | `google/gemma-2-2b-it` |
| Gemma 3 | `gemma3` | `google/gemma-3-1b-it` |

The user normally supplies only the model name:

```yaml
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
```

## Detection and compatibility

The compatibility layer inspects both model name and `AutoConfig` fields,
including model type/architecture. Llama generation can be inferred from a
recognizable identifier or vocabulary-size heuristic; uncertain generations
fail. Conflicting name/config evidence and incompatible manual templates fail
before training. Recognizable names can be used with a warning if config access
is unavailable; that fallback does not bypass gated weight authorization.

Names that do not look like Instruct/Chat checkpoints produce a warning: base
checkpoint SFT is permitted but should be intentional. Family matching is not
a substitute for checking upstream checkpoint/model-card compatibility and terms.

## Validation scope

Qwen2.5-0.5B-Instruct was really trained on CPU with LoRA and successfully loaded
for adapter inference. Qwen, Llama 2, Llama 3, Mistral, and Gemma tokenizers encoded
user/assistant examples through their resolved LLaMA-Factory templates.

Llama 2/3, Mistral, and Gemma generated training YAML parsed with the backend,
and incompatible `qwen` overrides were rejected. Official gated Meta/Google
configs returned authorization errors; public NousResearch Llama and unsloth Gemma
mirrors supplied real configs/tokenizers for the remaining smoke checks.

No large-model weights were downloaded for these smoke tests. Gemma 2/3 mapping
is implemented but those generations were not independently training-qualified.
The inspected backend emitted a Llama 3 new-token/resize warning during tokenizer
testing; weight-level vocabulary-resize behavior was not validated for that family.

## Operational prerequisites

- Obtain gated repository access and satisfy model license terms independently.
- Keep authentication outside committed YAML/source files.
- Select compatible torch/driver builds and supported precision for the host.
- Do not equate configuration smoke results with full GPU or large-model training.
- Preserve the matching base model/revision when publishing or using a LoRA adapter.

See [acceptance scope](acceptance_report.md) and [future family-extension guidance](development.md).
