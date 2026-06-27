# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## What LION is

**LION** (Learned Iterative Optimization Networks) is the University of Cambridge
CIA group's PyTorch toolbox for **learned image reconstruction**, primarily
**Computed Tomography (CT)**. It bundles:

- **Operators** — forward/adjoint linear operators (CT projection via
  `tomosipo`/`astra`, plus wavelet, Walsh–Hadamard, photocurrent-mapping, and
  composite operators).
- **Classical algorithms** — FDK, SIRT, CG, TV-min, FISTA, SPGL1.
- **Models** — data-driven reconstruction networks grouped by taxonomy:
  post-processing, iterative-unrolled, learned-regularizer, learned-FBP,
  plug-and-play (PnP), CNNs.
- **Optimizers/solvers** — supervised, self-supervised (Noise2Inverse,
  Noisier2Inverse, etc.), adversarial-regularizer training loops.
- **Data loaders** — `2DeteCT`, `LIDC-IDRI`, `LUNA16`, `walnuts` datasets, with
  download + preprocessing scripts.
- **Experiments / reconstructors / metrics / losses** — reproducible experiment
  definitions and evaluation tools.

It is explicitly **early-stage research software** (the README warns "Building in
progress... Many things are bound to fail and many more are bound to change").

## Repository layout

```
LION/                  the Python package (note: nested under repo root of the same name)
├── CTtools/           CT geometry + operator construction (ct_geometry, ct_utils)
├── operators/         linear operators; Operator base class; CTProjectionOp, Wavelet2D, ...
├── classical_algorithms/  fdk, sirt, conjugate_gradient, tv_min, fista, spgl1_torch
├── models/            reconstruction networks; LIONmodel.py is the base class
├── optimizers/        training solvers (LIONsolver base + supervised/self-supervised)
├── reconstructors/    LIONreconstructor, PnP
├── data_loaders/      dataset Dataset classes + download/preprocess scripts
├── experiments/       predefined experiment configurations
├── losses/, metrics/  evaluation tooling
├── utils/             parameter base class, paths, math, normaliser
└── exceptions/        custom exception types
demos/                 tutorial scripts (d00..d04) + one notebook
scripts/               data generation, example, hackathon, and paper-reproduction scripts
tests/                 pytest suite (operators, classical_algorithms, utils only)
```

The base classes worth knowing:
- `LION/models/LIONmodel.py` — `LIONmodel(nn.Module, ABC)`; all models inherit it
  for `save()`/`load()`/`save_checkpoint()`/`default_parameters()` with git-hash
  provenance for reproducibility.
- `LION/operators/Operator.py` — `Operator` base (forward `__call__` + `adjoint`).
- `LION/utils/parameter.py` — `LIONParameter`, the config object passed everywhere.

## Hard environment constraints (read before trying to run anything)

1. **GPU required for the CT stack.** `tomosipo` + `astra-toolbox` use CUDA GPU
   projectors. On a CPU-only host they *import* fine but **abort with a
   floating-point exception (SIGFPE) at runtime** the moment a projection runs.
   19 modules import `tomosipo` at module top, so anything touching CT needs a GPU.
2. **`tomosipo` and `ts_algorithms` are not on PyPI.** They are git-only
   dependencies (`github.com/ahendriksen/...`) pinned in `pyproject.toml`. Their
   legacy `setup.py` cannot build a wheel under a modern (non-conda) setuptools.
3. **`astra-toolbox` is conda-first.** The README installs it via conda; a PyPI
   wheel exists (`astra-toolbox`) and imports without a GPU.
4. **`LION/utils/paths.py` raises at import time if `LION_DATA_PATH` is unset.**
   Any module that transitively imports it will crash without that env var. Set
   `export LION_DATA_PATH=/path/to/Data` before using data loaders.
5. Supported Python is `>=3.9, <3.13` (numpy 1.x constraint).

## Installing / running

### Official path (conda + GPU) — see README.md

```
git submodule update --init --recursive
conda env create --file=env_base.yml --name=lion
conda activate lion
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

### Conda-free, CPU-only path (verified to import + pass the non-CT tests)

Useful on a machine with no conda and no GPU. The CT operators will still not
*run* (no GPU), but everything else imports and the non-CT test subset passes.

1. `pip install torch torchvision` (PyPI build) and `pip install astra-toolbox`.
2. `tomosipo`/`ts_algorithms`: fetch the source tarballs
   (`https://github.com/ahendriksen/tomosipo/archive/refs/tags/v0.7.0.tar.gz`,
   `.../ts_algorithms/archive/refs/heads/master.tar.gz`) and copy the
   pure-Python `tomosipo/` and `ts_algorithms/` package directories into
   site-packages. (A normal `pip install git+...` works only with conda's
   setuptools; the legacy `setup.py` fails to build a wheel otherwise.)
3. Install the remaining PyPI deps:
   `deepinv h5py imageio kornia lightning matplotlib numpy opencv-python-headless
   pandas pillow pydicom pylidc ptwt PyWavelets pytorch-wavelets scikit-image scipy
   spgl1 spyrit tifffile torchdiffeq torchmetrics tqdm wandb jaxtyping` plus
   `pytest pytest-xdist pytest-cov coverage`.
4. Put LION on the path without re-resolving the git deps:
   `pip install -e . --no-deps --no-build-isolation`.

## Testing

```
# Full suite (assumes the cuda marker + GPU):
pytest

# CPU-only: skip cuda-marked AND name-based *_cuda tests, plus the CT tests
# (which need a GPU but are NOT marked cuda):
pytest -o addopts="" -p no:cacheprovider -m "not cuda" -k "not cuda" \
  --ignore=tests/operators/test_ct_op.py \
  --deselect "tests/utils/test_operator_norm.py::test_ct_operator_norm_torch"
```

- `pytest` config lives in `pyproject.toml` (`filterwarnings = ["error", ...]`,
  `addopts = "-n auto --dist loadfile"` for xdist parallelism, and a `cuda` marker).
- The suite only covers `operators`, `classical_algorithms`, and `utils`. There
  are **no tests** for `models`, `data_loaders`, `optimizers`, `reconstructors`,
  `experiments`, `losses`, or `metrics`.

## CI / code style

- CI (`.github/workflows/pre-commit.yml`) runs **only Black** (via pre-commit,
  pinned to an old `black 22.10.0`). It does **not** run the test suite, ruff, or
  mypy — even though `pyproject.toml` has extensive ruff + mypy configuration.
- So formatting in CI ≠ the linting the project documents. Run `ruff` and `mypy`
  manually if you care about those (they will report many pre-existing findings).

## Conventions when editing

- New models inherit `LIONmodel` and implement `default_parameters()`; they must
  use the provided `save()`/`load()` so commit hashes and parameters are stored.
- Configuration flows through `LIONParameter` objects, not loose kwargs.
- Add a demo under `demos/` when adding user-facing functionality (per
  `developers.md`).
- Commit/push only when asked. The active development branch for this work is
  `claude/getting-it-running-233lp1`.

## Known issues to be aware of

See `IMPROVEMENTS.md` for the full, prioritized list. A few that will bite you
immediately:
- `LION/models/LIONmodel.py`: `ModelInputType` enum defines both `NOISY_RECON = 1`
  and `IMAGE = 1` (duplicate value — `IMAGE` aliases `NOISY_RECON`).
- `LION/operators/__init__.py` eagerly imports `CTProjectionOp`, so
  `import LION.operators` pulls in `tomosipo` even for non-CT use.
- `LION/data_loaders/2deteCT/` is not an importable module name (starts with a digit).
- `dowload.py` (LUNA16) is misspelled.
- `developers.md` is stale (refers to the old "AItomotools" name and wrong paths).
- Two conda env files (`env_base.yml` minimal, `env.yml` a 314-line pinned export)
  that disagree (python 3.12 vs 3.10).
