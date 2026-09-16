# Project #1 Acceptance Report

Validation date: September 16, 2026

Conclusion: PASS for the supported supervised fine-tuning/LoRA workflow.
Project #1 is ready to supply validated outputs to Project #2 (Benchmark Pipeline).
Original code is licensed under [Apache License 2.0](../LICENSE). Clean-target
reproduction remains pending and is not certified by source publication.

This is a portable summary of the completed acceptance validation, not a new
training run or universal certification of all HuggingFace models/hardware.
Raw logs, private machine paths, weights, and runtime evidence remain local and
are intentionally excluded from publication.

Docker support was added after this acceptance: the existing host suite plus
four static Docker contract tests passes 72 tests. CPU/CUDA base image digests
were verified, but no Docker executable/daemon is available on this host.
Image-build, container training/inference, and GPU acceptance remain pending;
see [Docker deployment](docker.md). These checks do not replace the original
real training/inference evidence or claim fresh server reproduction.

## Post-relocation verification

The approved source organization was validated separately from real training:

| Check | Result / evidence |
| --- | --- |
| Full relocated test suite | PASS: 68 tests in 11.432 s, exit code 0; 64 original + 4 layout checks |
| Package installation/imports | PASS: editable install without dependency changes; all 28 submodules import; `pip check` passes |
| Config loading/path resolution | PASS: default and all three named configs validate; default loads from another directory without `PYTHONPATH` |
| Real cached JSON preparation | PASS: 2 rows, Alpaca confidence 0.99, Qwen config resolution, run-local dataset registration/logs/YAML |
| Real cached HuggingFace preparation | PASS: `lhoestq/demo1` subset, 1 row, Alpaca confidence 0.99, automatic dataset registration |
| Generated backend YAML | PASS: both preparation recipes parsed by installed LLaMA-Factory `get_train_args()` |
| Existing actual adapters | PASS: both previously successful LoRA outputs rechecked by the relocated artifact validator |
| Retained runtime state | PASS: all 186 recorded runtime/evidence files retain original sizes and modification timestamps; none moved |
| Protected implementation | PASS: all 18 adapter files, model manager, and training configuration are byte-identical |

No real optimization or inference job was restarted for this relocation. Real
training and inference evidence below remains the original completed acceptance.
Temporary smoke runs were test-owned and cleaned by their temporary-directory
contexts, not written to the retained `runs/` directory. Historical run metadata
keeps its original source paths; source snapshots and normalized data remain in
those runs. The repository root was not initialized, committed, or pushed.

## Completed workflow

Input YAML led to automatic model/template resolution, validated training settings,
source/schema adaptation, normalized JSON, generated dataset registration/training
YAML, real LLaMA-Factory LoRA training, and run-scoped output verification. Metadata
reached `success` only after required model artifacts were checked.

Supported source/schema adapters and compatibility/configuration layers have
regression coverage. JSON and HuggingFace paths were additionally actually trained.
The saved Qwen adapter was loaded with its base model and saved tokenizer for generation.

## Execution environment

| Component | Recorded acceptance value |
| --- | --- |
| Platform | Windows 10, CPU only |
| Python | 3.12.7 |
| PyTorch | `2.14.0+cpu`; CUDA unavailable, zero GPUs |
| Transformers | 5.8.0 |
| PEFT | 0.18.1 |
| datasets | 4.0.0 |
| safetensors | 0.8.0 |
| LLaMA-Factory | `0.9.6.dev0` |
| CPU resource setting | Two threads; one training job at a time |

The inspected upstream source revision is
`97b32d3133b501432141a82949d5c7bc4d94f23a`; see [installation](installation.md)
for the pinned dependency recipe and reproduction caveats.

## Real training results

| Test | Dataset | Settings | Result |
| --- | --- | --- | --- |
| Original Qwen run | Synthetic local JSON, 2 Alpaca rows | Qwen2.5-0.5B-Instruct, LoRA rank 8, batch 1, epoch 1, 2 steps | PASS; 792.07 s training; loss 2.7600 |
| HuggingFace path | `lhoestq/demo1`, `train[4:5]`, 1 row | Same base, LoRA rank 2, batch 1, epoch 1, 1 step | PASS; 310.05 s training; loss 6.5016 |

The original machine shutdown interrupted the first run before optimization.
Recovery inspected process state, stale metadata, the existing log, and the empty
model output; no usable checkpoint existed. Completed inspection/preparation
was not restarted. Only the unfinished training stage was relaunched against the
same prepared run; recovery was recorded and it ultimately succeeded.

The Hub dataset was downloaded through `datasets.load_dataset()` and reused from
cache during training. YAML-only column mapping selected `package_name` as
instruction and `review` as output. Automatic detection selected Alpaca at 0.99.
`dataset_info.json` was generated without manual upstream edits. This is a tiny
mechanics dataset, not a recommended training corpus.

Both successful runs had input/resolved/training configs, normalized dataset,
dataset registration/source snapshot, `logs/train.log`, metadata, and saved model
output within their isolated run directories. LoRA artifact validation passed:

| Run | Adapter configuration | Adapter safetensors |
| --- | --- | --- |
| Local JSON | 1,100 bytes | 17,640,136 bytes |
| HuggingFace | 1,099 bytes | 4,442,408 bytes |

## Feature validation

| Feature | Status | Evidence / scope |
| --- | --- | --- |
| Dataset flexibility | PASS | Source/schema regression tests; real JSON/HF training; [public configs](../configs/) |
| Automatic detection and errors | PASS | Scored candidates, ambiguity, invalid rows, overrides, optional-dependency cases tested |
| Model compatibility | PASS: smoke scope | Real config/tokenizer checks for Qwen, Llama 2/3, Mistral, Gemma; YAML parsing and incompatible-override rejection |
| Training automation | PASS | Two real LoRA jobs completed using generated run-local configs/registration |
| Run isolation | PASS | Distinct actual run outputs; timestamp collision/no-overwrite unit tests |
| Logging and metadata | PASS | Logs persisted; model/dataset/training/output recorded; both actual statuses `success` |
| Artifact validation | PASS | Both actual adapters verified; missing-output failure cases tested |
| Failure reporting | PASS: handled-error scope | Simulated process/artifact failures marked runs failed with error logs |
| Adapter inference | PASS | Saved tokenizer + recorded base + trained adapter loaded and generated text |
| Original regression suite | PASS | 64 tests; exit code 0 |
| Post-relocation regression suite | PASS | 68 tests; exit code 0; see verification above |
| Full fine-tuning | PARTIAL | Configuration/artifact-unit coverage; no real full-method run |
| GPU/H100 deployment | NOT TESTED | CPU-only host |

Model/tokenizer smoke checks resolved `qwen`, `llama2`, `llama3`, `mistral`, and
`gemma` templates and encoded a simple user/assistant pair. Official Meta/Google
configs were gated; public NousResearch Llama and unsloth Gemma mirrors supplied
real config/tokenizer coverage. No large-model weights were downloaded. Llama 3
produced an upstream new-token/resize warning; its weight-level vocabulary-resize
behavior was not validated. Gemma 2/3 are implemented mappings, not additional
independently validated full training runs.

Relevant regression modules are in
[tests/](../tests/), including
[run lifecycle/artifact tests](../tests/test_run_management.py),
[training configuration tests](../tests/test_training_config.py),
and [model tests](../tests/test_model_manager.py).

## Inference smoke

Loaded the successful original run's adapter, its saved tokenizer, and
`Qwen/Qwen2.5-0.5B-Instruct` base revision
`7ae557604adf67be50417f59c2c2f167def9a775` on CPU in float32.

Prompt: `What is artificial intelligence?`

Recorded generated output, deliberately capped at 32 new tokens:

> Artificial Intelligence (AI) refers to the field of research and development
> that aims at creating intelligent machines capable of performing tasks that
> typically require human intelligence. AI can

The adapter loaded and generated a nonempty response. The token cap explains the
unfinished final phrase; this was a usability smoke test, not quality evaluation.
Inference is not an automatic step of each training execution.

## Minimum fix made during acceptance

The recovered run requested `gradient_checkpointing: false`, but the backend
model loader enabled checkpointing by default. The generic Trainer flag did not
control that loader. YAML generation was minimally corrected to use inverse
`disable_gradient_checkpointing`; regression assertions/true-false mapping
coverage and a README note were updated. The subsequent real HF run used
`disable_gradient_checkpointing: true` and did not enable checkpointing.
No dataset/model/training-layer redesign was performed.

The regression suite passed outside restrictive Windows sandbox policies. One
transcript-capture attempt hit temporary-folder permissions; the permitted rerun
passed 64 tests. Test-owned empty folders from that attempt were cleaned and raw
evidence preserved locally. This documentation pass makes no new functional fix.

## Remaining limitations

### Functional blockers within the validated scope

None identified for supported-family SFT/LoRA. Publication is a separate scope:
a clean environment must reproduce installation before a release claims
deployment reproducibility. Apache-2.0 licensing was selected during publication
preparation, without changing this functional acceptance evidence.

### Additional qualification / future improvements

- GPU/H100/BF16/FP16 and real larger-family/full-method training validation.
- Complete target-specific dependency locks and retrievable wheel/source provenance.
- Model/dataset revision tracking and licensing/privacy review for real inputs.
- Large-dataset stress testing; current materialization is not bounded-memory streaming.
- Automated power-loss status reconciliation/resumption; stale statuses are possible.
- Stronger full-model shard/integrity checks; existence/size checks are not full verification.
- QLoRA and DPO training are intentionally not implemented.
- LoRA output requires the matching base model; no automatic merged standalone model.
- Minimal runs do not establish training quality or universal model compatibility.

## Final acceptance checklist

- [x] Completed phases preserved; interrupted training assessed before recovery.
- [x] Original prepared run's unfinished training completed successfully.
- [x] Real one-epoch local JSON LoRA test passed.
- [x] Real HF subset passed adaptation, registration, and training.
- [x] Model/template and training YAML generated without manual upstream edits.
- [x] Run artifacts, logs, metadata, and both real LoRA outputs verified.
- [x] Model-family configuration/tokenizer smoke checks passed.
- [x] Invalid-template and missing-artifact error paths tested.
- [x] Base-model-plus-adapter inference generated a response.
- [x] 64 regression tests passed.
- [x] Untested hardware/method/scale boundaries documented.

Project #1 can proceed to Project #2 for benchmarking; publication/release gates
remain described in [the repository preparation plan](repository_structure.md).
