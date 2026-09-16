# Developer guide

This describes extension points for future approved development. No adapter,
model-family, training-method feature, or core refactor is added by the current
repository organization pass.

## Layout and tests

Source: `src/fine_tuning_pipeline/`. Regression tests: `tests/test_*.py`.
Install with `python -m pip install -r requirements.txt` from the repository root
before imports/tests. See [repository classification](repository_structure.md).

With your virtual environment active:

```bash
python -m unittest discover -s tests -v
```

Use targeted modules when appropriate, for example:

```bash
python -m unittest discover -s tests -p "test_dataset_schema_adapters.py" -v
```

The original acceptance suite had 64 tests; relocation added four layout checks.
The current suite has 72 tests, including four static Docker deployment checks.
Mocked failures deliberately exercise failed
metadata/logging paths; error messages in their transcript do not by themselves
indicate a failed suite. Read its final result and exit code.

## Add a dataset adapter

First decide whether the gap is storage or record meaning. Existing JSON/JSONL/
CSV/Parquet/HF loaders can already feed new record schemas; do not duplicate them.
Prefer `dataset.columns` mapping for simple renamed fields before writing code.

For genuinely new formats:

1. Add a focused module under `src/fine_tuning_pipeline/dataset_adapters/` implementing
   `BaseDatasetAdapter`. Declare a unique `name`, optional `aliases`, and
   `stage = AdapterStage.SOURCE` or `AdapterStage.SCHEMA`.
2. Implement `detect(data, context) -> DetectionResult`, returning confidence
   within `[0, 1]`, a match flag, and explainable evidence without mutating input.
3. Implement `validate(data, context) -> None`, raising a precise framework error
   with fields/row/turn information. Validation must still run for manual overrides.
4. Implement `convert(data, context)`, returning loaded raw records for source
   adapters or compatible SFT rows for schema adapters.
5. Reuse `FieldMappingAdapter` in `base.py` for named text fields; configure
   required fields, output mapping/defaults, and detection evidence. Reuse
   `ConversationAdapter` in `conversation.py` for shared conversation validation
   and turn expansion. Do not duplicate these lifecycles.
6. Import/register the instance in `create_default_framework()` in
   `src/fine_tuning_pipeline/dataset_adapters/__init__.py`, in the appropriate source/schema registry.
   There is no automatic Python-plugin discovery to assume.
7. Add detection, conversion, invalid-input, ambiguity/override, and registry
   tests. For loaders, include optional-import, path, and malformed-source cases.
8. Update dataset/config docs and a small synthetic fixture/example. Keep
   preference/multimodal tasks out of SFT unless separately approved.

Registries enforce unique format names and stage compatibility; scoring does
not replace complete validation. Imports for optional source dependencies should
stay lazy and failures should explain what is needed.

## Add a model family

The compatibility layer is `src/fine_tuning_pipeline/model_manager.py`. Extend it only through an
approved change with checkpoint and backend-template evidence:

1. Confirm the template is registered in the pinned LLaMA-Factory version.
2. Add deterministic name and HuggingFace config detection, including generation
   distinctions where needed; do not rely on an arbitrary name substring alone.
3. Update compatible family/template mappings and useful warnings. Preserve
   rejection of name/config conflicts, unsupported architectures, uncertain
   generations, and incompatible explicit overrides.
4. Unit-test using the existing injectable `config_loader` to avoid network or
   gated access requirements in the regression suite.
5. Smoke-test real config/tokenizer behavior and generated backend YAML. Obtain
   appropriate hardware/access for separate weight-level training qualification.
6. Update supported-model docs with precise validation scope; do not infer full
   support merely from one successful template lookup.

Do not hardcode a new model/template in the YAML generator or bypass compatibility
validation in the pipeline entry point.

## Extend training configuration

The public schema and typed values live in `src/fine_tuning_pipeline/training_config.py`;
final run/YAML assembly lives in `src/fine_tuning_pipeline/yaml_generator.py`.

1. Confirm the actual argument name/semantics in the pinned backend.
2. Add the field to the supported public schema and typed configuration, decide
   whether it is required, and validate type/range/method/precision constraints.
3. Add explicit translation rather than passing unknown YAML keys through.
   Note that checkpointing uses the backend's inverse loader flag in final YAML.
4. For methods, extend method-specific configuration deliberately and include
   output-artifact verification and lifecycle tests; do not silently treat QLoRA
   as ordinary LoRA or preference data as SFT.
5. Test config propagation, invalid/missing fields, incompatible combinations,
   valid serialized YAML, and backend argument parsing. Preserve fixed SFT
   invariants and the run-specific dataset/output paths.
6. Document settings/default semantics and hardware prerequisites. Execute a
   proportional real acceptance check for changes affecting training behavior.

## Run and dependency contract

Always let run management allocate isolated artifacts. Never write generated
dataset registration into the upstream checkout, reuse a previous model output
directory, or declare success solely from the subprocess exit code.

Use small, synthetic fixtures in tests. Keep credentials, corpora, weights,
environment directories, raw machine logs, and temporary files out of Git.
Pin the upstream source and verify the chosen environment after dependency
changes. Public release requires an approved [license](../LICENSE) and the
[release/layout gates](repository_structure.md), not just a passing test suite.

## Import and path conventions

Use package-relative sibling imports inside `fine_tuning_pipeline`, and
`from fine_tuning_pipeline.<module> import ...` for tests and external callers.
Mock targets must use the same qualified module path. Run the entry point as
`python -m fine_tuning_pipeline.train_pipeline`, not as a file under `src/`.
The default YAML is `configs/config.yaml`; explicit config/dataset/output paths
keep their existing config-relative semantics. Test-owned temporary directories
belong under `tests/`, never in legacy artifacts or an existing run.
