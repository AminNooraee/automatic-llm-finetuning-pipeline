# Supported datasets

The framework separates source storage from record schema. Any supported source
can contain any supported compatible schema; a new JSONL schema does not need a
new JSONL loader. Automatic detection is the normal workflow:

```yaml
dataset:
  path: ../examples/datasets/alpaca_demo.json
  name: demo_dataset
  source_format: auto
  format: auto
```

## Source adapters

| `source_format` | Input |
| --- | --- |
| `json` | Top-level JSON array of records |
| `jsonl` / `ndjson` | One JSON record per nonempty line |
| `csv` | Header/columns and text fields; dialect detection |
| `parquet` | Parquet records; lazy PyArrow import |
| `huggingface` / `hf` | `datasets.load_dataset()` dataset identifier; optional subset/split/load kwargs |
| `chatml_text` | Raw ChatML text, such as `.txt` or `.chatml` |

`auto` uses extension, file structure/magic/header, identifier characteristics,
and then schema structure/keys. Missing local file paths are not treated as Hub
names. HuggingFace/PyArrow dependencies are lazy at the adapter level, though
the complete training backend installs datasets as its own dependency.

## Schema adapters

| `format` | Required shape | Unified conversion |
| --- | --- | --- |
| `alpaca` | `instruction`, `output`; optional `input` | Direct fields; input defaults to empty |
| `instruction_response` | `instruction`, `response` | Response -> output; input empty |
| `prompt_completion` | `prompt`, `completion` | Prompt -> instruction; completion -> output |
| `qa_json` | `question`, `answer` (legacy `prompt`, `response` also supported) | Question -> instruction; answer -> output |
| `rag_qa` | `context`, `question`, `answer` | Context -> input; question -> instruction; answer -> output |
| `openai_chat` | `messages` with string `role`/`content` | Per-assistant-turn SFT expansion |
| `sharegpt` | `conversations` with `from`/`value`, e.g. human/gpt | Shared conversation validation and SFT expansion |
| `chatml` | ChatML strings or structured `text` values | Parse roles, validate, expand turns |
| `dpo` | `prompt`, `chosen`, `rejected` | Preference interface/detection only; SFT training rejects it |

Unified SFT JSON rows are:

```json
{"instruction": "Question or request", "input": "Optional context", "output": "Target answer"}
```

The strings must satisfy schema/unified validation. Chat accepts text roles
`system`, `user`, `assistant`, alternating requests/replies and leading system
messages. Earlier conversation is flattened into labeled text in `input`, not
preserved as native role arrays or separate loss masks. Tool/function calls,
multimodal content arrays, arbitrary nested answers, and unsupported schemas
are not covered by this text-only SFT contract.

## Detection, manual overrides, and diagnostics

The registry scores candidates using heuristic evidence, including sampled keys
and message structure. Full validation follows selection. Closely scored matches
are rejected as ambiguous; errors show detected fields and candidate explanations.
Scores are not calibrated probabilities of correctness for arbitrary data.

Set `dataset.format: sharegpt`, for example, to force that schema. Forced selection
does not skip validation. For nonstandard columns use a canonical-to-source map:

```yaml
dataset:
  path: ../datasets/custom.csv
  name: custom_data
  format: auto
  columns:
    instruction: prompt_text
    input: context_text
    output: answer_text
```

Mappings rename canonical fields for detection/conversion; they do not perform
arbitrary nested transformations. Use a supported schema or a future adapter for
structures beyond this mapping capability.

## HuggingFace subsets

See [the validated HF example](../configs/huggingface_dataset_example.yaml).
Optional `subset`, `split`, and `load_kwargs` configure the dataset loader. This
path generates normalized JSON and run-local `dataset_info.json` automatically;
manual upstream registry edits are not required.

Only the selected normalized records are stored alongside a Hub source/options
descriptor. Revision pinning, raw export retention, and data licensing/privacy
requirements should be reviewed before external distribution.

## Size and publication constraints

Records are currently materialized in memory; neither streaming nor 100k/1M-row
stress qualification was part of acceptance. Small JSON and Hub datasets really
trained; other sources/schemas have regression coverage. Public examples are
synthetic. Keep real/private corpora and generated normalized data out of Git.

See [examples](../examples/README.md), [configuration](configuration.md), and
[acceptance](acceptance_report.md).
