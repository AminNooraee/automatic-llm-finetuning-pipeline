# Installation and reproducibility

The existing Windows/CPU environment is validated. Relocation installed only the
editable application package, without changing its training dependencies.
Instructions below prepare a new environment; a complete clean install has not
been executed in this pass. Do not replace the validated environment without verification.

## Prerequisites

- Python 3.12.7 is the recorded acceptance version; inspected upstream metadata
  requires Python 3.11 or newer. Other Python versions were not acceptance-tested.
- Git and access to the package/Hub/upstream repositories for first installation.
- Enough RAM/disk for the chosen model, packages, normalized data, and runs.
- For gated models/datasets, approved access and credentials supplied outside Git.
- A separately validated PyTorch/driver combination for GPU execution. H100/GPU
  operation has not been tested in Project #1 acceptance.

## Virtual environment

From the repository root:

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS shell
source .venv/bin/activate
```

Activation is optional if every command uses the environment's explicit Python
path. On this validated Windows workspace that path is `venv/Scripts/python.exe`.
Keep environments out of version control.

## Select PyTorch for the host

Install torch, torchvision, and torchaudio as a compatible set from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
choosing your OS and CPU/CUDA platform. Do not blindly reuse CPU builds for GPU.
The application manifest intentionally does not choose a CUDA runtime or wheel
index; upstream training requires PyTorch and its related packages.

Recorded installed CPU builds were:

| Package | Installed version |
| --- | --- |
| torch | `2.14.0+cpu` |
| torchvision | `0.29.0+cpu` |
| torchaudio | `2.11.0+cpu` |

These are observations from the validated host, not a claim that every platform's
public wheel index provides the same builds. Confirm availability and compatibility
on the fresh target; archive approved wheels or a complete target lock if exact
long-term binary reproducibility is required.

## Install the application/backend dependencies

After selecting/installing PyTorch:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import yaml, transformers, peft, llamafactory; print('Imports OK')"
```

[requirements.txt](../requirements.txt) pins the inspected application integration
versions and installs LLaMA-Factory from immutable source revision
`97b32d3133b501432141a82949d5c7bc4d94f23a`. The local backend reports
`0.9.6.dev0`. Do not substitute the moving upstream `main` branch and claim the
same acceptance evidence. The manifest also installs this checkout in editable
mode (`-e .`); [pyproject.toml](../pyproject.toml) discovers only the package under
`src/` and pins the tested setuptools build backend.

The installed acceptance versions of transformers, peft, datasets, accelerate,
trl, safetensors, and PyYAML are reflected in the manifest. PyArrow is pinned from
the inspected current environment. Upstream transitive dependencies still resolve
normally: this is not a complete hash-locked environment export. Installing the
requirements may affect versions in its target environment; use a new one.

The existing `LLaMA-Factory/` folder is a separate checkout with legacy data edits,
not code to copy into this repository or a prerequisite path at runtime. The
launcher imports the installed `llamafactory` package. Preparation generates each
run's own dataset registry; no upstream data edits are needed.

## Optional adapter-only use

Standalone local source normalization does not require importing the training
backend. With your environment active, install the editable package and its PyYAML dependency;
enable source-specific dependencies only as needed:

```bash
python -m pip install -e .
# Only for HuggingFace loading:
python -m pip install datasets==4.0.0
# Only for Parquet loading:
python -m pip install pyarrow==25.0.1
```

The adapter framework lazily imports datasets/PyArrow and reports missing packages
when those sources are requested. The complete LLaMA-Factory training environment
itself requires datasets, so full installation includes it even for JSON training.

## Optional Conda bootstrap

```bash
conda env create -f environment.yml
conda activate automatic-llm-finetuning
```

Then select/install PyTorch and run `python -m pip install -r requirements.txt` as
above. [environment.yml](../environment.yml) bootstraps Python/pip only; it is not
a CUDA environment specification or a full solved dependency lock. Conda creation
was not tested during this pass.

## Tests and verification

With the environment active:

```bash
python -m unittest discover -s tests -v
```

The current suite contains 72 tests (64 original, four relocation checks, and
four static Docker checks).
Windows sandbox policies can block test
temporary directories/cache locks even when the source is readable. A permission
failure is not a reason to change adapters/training logic; use a permitted test
environment and inspect the actual errors. Passing negative-path tests intentionally
log simulated failures.

Only use offline mode after required files are cached. It cannot supply missing
weights or replace authorization for gated repositories. See [usage](usage.md).

## Publication/reproduction gates

Before publishing a reproducible release, confirm the pinned upstream revision and
package wheels are retrievable from the release environment, solve/lock its full
dependency set, run `pip check` and regression tests, and repeat the minimal training
and inference acceptance on that target. No complete fresh training-environment install, dependency upgrade, container
build, or GPU deployment was performed here. Editable installation was verified
in the existing environment. See [layout/release gates](repository_structure.md).

## Distribution scope

[Docker deployment](docker.md) now provides CPU and selectable CUDA images using
the same editable source checkout. Build with
`docker build -f docker/Dockerfile -t automatic-llm-finetuner .` from the root.
No Docker daemon is available on the current validation host, so this is provided
support, not a claim of completed image-build/training acceptance.

This pass supports editable installation from a source checkout. Default configs
and synthetic examples remain in that checkout, outside the Python package.
A standalone wheel with bundled configs is not claimed or validated. Do not use
a copied wheel as a substitute for the documented editable installation.
