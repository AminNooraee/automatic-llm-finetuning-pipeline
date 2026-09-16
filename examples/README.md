# Public examples

These small synthetic examples illustrate supported schemas; they are not
quality training datasets. No model weights, private datasets, or raw logs belong
in this directory. The relocated [Alpaca demo](datasets/alpaca_demo.json)
is the public two-row dataset used by the default config.

| Example | Automatic schema | Unified mapping |
| --- | --- | --- |
| [alpaca_demo.json](datasets/alpaca_demo.json) | `alpaca` | Instruction, optional input, output |
| [rag_qa.json](datasets/rag_qa.json) | `rag_qa` | Question -> instruction, context -> input, answer -> output |
| [openai_chat.json](datasets/openai_chat.json) | `openai_chat` | Each assistant response becomes a row; prior conversation becomes input |

From a config in root `configs/`, use `../examples/datasets/rag_qa.json` or
`../examples/datasets/openai_chat.json` as `dataset.path`. The default
`configs/config.yaml` uses `../examples/datasets/alpaca_demo.json`.
Paths resolve relative to the config file.

Keep `dataset.format: auto` and a nonempty `dataset.name`. Instructions for the
named [model/dataset configs](../configs/) are in [usage](../docs/usage.md).
