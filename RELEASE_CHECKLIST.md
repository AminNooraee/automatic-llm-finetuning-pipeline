# v1.0.0 release checklist

Prepared September 16, 2026. Status: **source publication authorized; preparing
GitHub publication**. Docker and clean-target deployment qualification remain pending.
Completed checks refer to this source tree/host, not a public tag or clean server.
Record evidence, reviewer, environment, and date when closing each remaining gate.

## Before publishing

- [x] Tests passed — 72 host regression tests; no real optimization restarted.
- [x] Documentation complete — required topics, current commands/paths, validation
  scope, limitations, and relative links reviewed.
- [ ] Docker tested — build CPU and CUDA variants, verify image dependencies/tests,
  then minimally train/infer in the CPU container with persistent output mounts.
- [ ] Clean installation tested — fresh target environment installs the documented
  stack without depending on this host's venv, vendor checkout, or existing cache.
- [x] Staged source reviewed — 71 curated text files, no forbidden tracked paths,
  no oversized files, public synthetic fixtures/configs inspected, and selected
  credential/private-key patterns absent. This initial repository has no prior
  history. No automated scan guarantees absence of every possible secret.
- [x] License approved — owner requested Apache License 2.0; full text and package/
  documentation references updated. Third-party model/data terms remain separate.
- [x] Version aligned — package `1.0.0`, release notes/changelog `v1.0.0`.
- [x] Owner authorizes the initial release commit, `v1.0.0` tag, and public GitHub
  publication under `AminNooraee`; this does not claim deployment qualification.

## Files that must never be committed in this source release

- `runs/`, `models/`, `checkpoints/`, and `checkpoint-*/` directories, including
  normalized private data, metadata snapshots, trainer state, logs, and model weights.
- HuggingFace/download caches: `.cache/`, `.huggingface/`, `hf_cache/`,
  `huggingface_cache/`, cache volumes, Hub tokens, and user-home cache directories.
- Virtual environments (`venv/`, `.venv/`, other machine-specific environments),
  Python caches, build/dist/egg metadata, and test temporary directories.
- Secrets/credentials: `.env` files with real values, private keys/certificates,
  credential JSON/YAML, `.netrc`, `.pypirc`, `secrets/`, `credentials/`, and
  tokens embedded in source/YAML/URLs/logs. Review `.env.example` too, if introduced.
- Private/full external `datasets/`, generated normalized data or training YAML,
  raw acceptance logs/evidence, and legacy `fine_tuning_pipeline/` runtime remnants.
- The local `LLaMA-Factory/` Git checkout, unrelated benchmark/serving/evaluation
  outputs/code, editor/machine files, and absolute-path environment exports.

Only documented synthetic examples belong under `examples/datasets/`.
Unknown names/content can escape ignore patterns: review the actual files.
Git ignore rules do not remove already tracked files or prevent force-adds;
see [Git's ignore documentation](https://git-scm.com/docs/gitignore).

## Docker acceptance evidence

On a Docker-enabled target, follow [Docker deployment](docs/docker.md):

1. Build `automatic-llm-finetuner:1.0.0` and `:1.0.0-cuda` from the release tree.
2. Verify build-time tests, `pip check`, imports, and config loading without a GPU.
3. Run the documented tiny LoRA recipe with read-only inputs and writable
   persistent `runs/`/cache mounts; verify metadata success, logs, and artifacts.
4. Load matching base + saved adapter/tokenizer and generate a response in-container.
5. Record image digest/ID, `/opt/deployment-requirements.txt`, platform, input
   provenance, run evidence, and mount/user permissions. Keep private evidence local.
6. GPU runtime checks are additionally required before claiming GPU/H100
   qualification. A CUDA build alone is not such evidence.

## Clean installation evidence

On a fresh target, follow [installation](docs/installation.md), select a matched
PyTorch set, and install `requirements.txt`. Verify `pip check`, all tests, default/
named configs, a minimal local JSON/HuggingFace run, and adapter loading. Record
resolved dependencies plus model/data provenance. Do not relabel an existing-venv
editable reinstall as clean reproduction. Conda and wheel distribution require
separate checks if included in the release claims.

## Manual Git preparation and review

The project root has its own newly initialized `main` repository. The upstream
checkout remains a separate, ignored repository with pre-existing data edits.
Publication preparation does not alter or copy its Git history.

The owner has authorized publication. Stage only
the intended source release: root manifests/README/LICENSE/release documents,
`src/`, `tests/`, `configs/`, `examples/`, `docs/`, and `docker/`. Never blindly
stage the complete workspace or force-add generated artifacts. Run your approved
secret scanner on the candidate tree and relevant history.

Read-only review commands, once a project-root repository/index exists:

```bash
git status --short --ignored
git diff --cached --name-only
git diff --cached --check
git ls-files --cached
git ls-files --cached --ignored --exclude-standard
```

The final command should show no forbidden tracked paths; review file content
even if it is empty. Inspect configuration/example licenses and ensure no private
paths or secrets are included. Treat missing Git history/index as an unperformed
review, not a clean result. [Git file-listing reference](https://git-scm.com/docs/git-ls-files).

The owner requested the release commit and `v1.0.0` tag and publication to GitHub.
Record their actual hashes/remote evidence after verification. A hosted GitHub
Release object or Docker image upload is not implied. Docker and clean-install
checks stay open; do not advertise those deployments as qualified.
