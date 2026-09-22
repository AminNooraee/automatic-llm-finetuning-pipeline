# Console observability

The training subprocess has two independent output consumers:

```text
LLaMA-Factory stdout + stderr
          |
          +----> logs/train.log (complete raw stream)
          |
          +----> streaming console formatter
                       |
                 quiet / concise / full
```

Configure the formatter in the input YAML. Existing configurations without this
section use `concise` automatically.

```yaml
observability:
  console_verbosity: concise
```

## Modes

- `quiet` shows run creation, the training model/dataset summary, start/completion
  or failure, final status, and important errors. Backend progress and INFO noise
  stay out of the terminal.
- `concise` is the default. It additionally recognizes training setup, throttled
  Trainer progress and per-step metric dictionaries, plus warnings, errors, and
  tracebacks. Preprocessing/loading progress and the backend's final metrics table
  stay hidden; the structured completion summary reports those final metrics once.
  Its matching is model-family-neutral.
- `full` reproduces the unfiltered live backend stream while still writing it to
  the log.

Example concise output:

```text
=== Training Started ===
Model: Qwen/Qwen2.5-0.5B-Instruct
Method: LORA
Dataset: smoke_demo
Samples: 1200
Epochs: 3
Batch size: 1

Progress: 75/150 (50%)
Step 75/150 | Epoch 0.5 | Loss 1.234 | GradNorm 0.91 | LR 0.0001

=== Training Complete ===
Train loss: 0.75
Runtime: 288.1 s
Samples/sec: 4.16
Steps/sec: 0.52
Status: SUCCESS
```

Only values actually emitted by the backend or extracted from Trainer artifacts
are displayed. Training loss is never presented as accuracy. Per-step
`log_history` remains in `trainer_state.json` rather than being duplicated.

## Authoritative logging and limitations

`logs/train.log` always contains the complete combined raw stdout/stderr stream,
including INFO lines, model/configuration dumps, training examples, and progress
updates hidden from the terminal. Raw carriage returns and newlines are retained
where UTF-8 decoding permits.

Concise parsing follows common Transformers/LLaMA-Factory text formats rather
than a stable machine-readable event protocol. Unknown future formats remain in
`train.log` but may not appear in concise mode until the recognizers are updated.
The formatter buffers only the current unterminated record, not the complete log.

Console presentation does not implement serving, inference, vLLM, LiteLLM,
endpoint generation, adapter merging, Project #2 integration, QLoRA, or any new
training algorithm.
