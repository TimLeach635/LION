# IMPROVEMENTS.md

A prioritized, modular catalogue of potential improvements to the LION
repository. Intended as a **source of truth** to be worked through one item at a
time with a human in the loop.

**How this is organised**
- Items are grouped by **size**: **Small** (a focused change, minutes–an hour),
  **Medium** (a contained but non-trivial piece of work), **Large** (a project /
  multi-PR effort).
- Each item has a stable **ID** (`S-01`, `M-03`, `L-02`, ...) so it can be
  referenced in commits/PRs/issues.
- Each item lists **What / Why / Where / Suggested action**, and where useful a
  rough note on risk or dependencies.
- IDs are stable; when an item is done, mark it rather than renumbering.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` won't do.

> Observations were gathered by static inspection plus actually installing and
> running the test suite in a CPU-only, conda-free environment. Some line
> numbers may drift as the code changes — treat file paths as the anchor.

---

## Small improvements

### Correctness / bugs

- [ ] **S-01 — Fix duplicated `ModelInputType` enum value.**
  - **What:** `ModelInputType` defines `NOISY_RECON = 1` and `IMAGE = 1`. In
    Python, `IMAGE` becomes an alias of `NOISY_RECON`, so the two input types are
    indistinguishable.
  - **Why:** Any logic branching on `IMAGE` vs `NOISY_RECON` (e.g. normalisation
    decisions in `set_normalisation`) is silently wrong.
  - **Where:** `LION/models/LIONmodel.py` (`class ModelInputType`).
  - **Action:** Give `IMAGE` a distinct value (`= 2`). Audit all call sites and
    saved checkpoints that store the integer value; add a regression test.
  - **Risk:** Medium — changing the integer may affect serialized models. Consider
    keeping `NOISY_RECON = 1`, `IMAGE = 2` and a migration note.

- [ ] **S-02 — Remove the no-op `__metaclass__` assignment.**
  - **What:** `LIONmodel.__init__` sets a local variable `__metaclass__ = ABCMeta`.
    This is the Python-2 idiom and does nothing in Python 3 (abstractness already
    comes from inheriting `ABC`).
  - **Where:** `LION/models/LIONmodel.py`, in `__init__`.
  - **Action:** Delete the line; rely on `ABC`.

- [ ] **S-03 — Audit the 3 bare `except:` clauses.**
  - **What:** Three bare `except:` blocks swallow everything (including
    `KeyboardInterrupt`/`SystemExit`).
  - **Where:** `grep -rn "except:" LION` (3 hits).
  - **Action:** Narrow to specific exception types, or at least `except Exception`.

### Naming / typos

- [ ] **S-04 — Rename misspelled `dowload.py`.**
  - **Where:** `LION/data_loaders/LUNA16/dowload.py` → `download.py`. Update imports
    and any script references.

- [ ] **S-05 — Fix stale "AItomotools" references.**
  - **What:** 13 references to the project's old name across `.py`/`.md`.
  - **Where:** notably `developers.md`, code comments.
  - **Action:** Replace with "LION" / correct paths.

- [ ] **S-06 — Tidy comment typos & stale aliases.**
  - **What:** e.g. "definest", "optinal", "tomogprahic" in `LIONmodel.py`;
    `import LION.utils.utils as ai_utils` uses an old alias.
  - **Action:** Cheap readability cleanup; do alongside other edits to the file.

### Docs (small)

- [ ] **S-07 — Rewrite `developers.md`.**
  - **What:** It is marked "WIP", uses the old name, and points to
    `./AItomotools/models/LIONmodel.py` (wrong path) with a tree that omits half
    the current packages (operators, optimizers, reconstructors, experiments,
    losses, exceptions).
  - **Action:** Regenerate the structure section from the real tree; link to the
    correct base-class path; reference `CLAUDE.md`.

- [ ] **S-08 — Reconcile the two conda env files.**
  - **What:** `env_base.yml` (minimal, python 3.12) and `env.yml` (a 314-line fully
    pinned export, python 3.10, channel `aahendriksen`) disagree. The README uses
    `env_base.yml`; `env.yml` looks like a stale frozen lockfile.
  - **Action:** Decide which is canonical. Either delete `env.yml`, or clearly
    label it "frozen reference environment" and align its Python/astra versions.

- [ ] **S-09 — Document `LION_DATA_PATH` requirement prominently.**
  - **What:** `paths.py` raises `RuntimeError` at import if `LION_DATA_PATH` is
    unset. New users importing a data loader hit a hard crash.
  - **Action:** Add a clear note in README/Datasets and in the error message link
    to docs. (Behaviour change tracked separately in M-05.)

### Packaging (small)

- [ ] **S-10 — Remove or reconcile the legacy `setup.py`.**
  - **What:** `setup.py` hardcodes `version="0.2"`, `author`, `license="BSD"`,
    `url="-"`, and a commented `os.system("pip install ...")`. It conflicts with
    `pyproject.toml`'s dynamic git-based versioning and metadata.
  - **Action:** Delete `setup.py` (pyproject is authoritative) or reduce it to a
    `setuptools.setup()` shim. Confirm builds still work.

- [ ] **S-11 — Make `import LION` expose a version.**
  - **What:** `LION/__init__.py` is empty; there is no `LION.__version__`.
  - **Action:** Expose `__version__` via `importlib.metadata.version("LION")`.

- [ ] **S-12 — Pin/refresh the pre-commit Black version.**
  - **What:** `.pre-commit-config.yaml` pins `black 22.10.0` (years old); the CI
    workflow comments about Python 3.14 incompatibility.
  - **Action:** Bump Black to a current release and update the hook rev; re-run on
    the codebase (a formatting-only commit).

---

## Medium improvements

### Testing & CI

- [ ] **M-01 — Run the test suite in CI.**
  - **What:** CI only runs Black. `pytest` never runs on PRs, so regressions in
    operators/algorithms are not caught.
  - **Where:** `.github/workflows/`.
  - **Action:** Add a workflow that installs the CPU-runnable subset and runs
    `pytest -m "not cuda"` (excluding the GPU-bound CT tests — see M-02). Consider
    a separate self-hosted/GPU job for the cuda-marked tests.
  - **Depends on:** M-02 (so the CPU job doesn't SIGFPE), L-04 (clean CPU install).

- [ ] **M-02 — Mark GPU-only tests with the `cuda` marker.**
  - **What:** The `cuda` pytest marker exists, but several GPU-requiring tests are
    not marked: `tests/operators/test_ct_op.py`,
    `tests/utils/test_operator_norm.py::test_ct_operator_norm_torch`, and
    `tests/classical_algorithms/test_fista.py::test_fista_l1_cuda` (named `_cuda`
    but unmarked). The CT ones hard-crash the whole pytest process (SIGFPE) on a
    CPU-only host rather than failing gracefully.
  - **Action:** Add `@pytest.mark.cuda` to all tests that need a GPU so
    `-m "not cuda"` cleanly selects the CPU subset. Optionally add an autouse
    fixture that `pytest.skip`s cuda tests when `torch.cuda.is_available()` is
    False.

- [ ] **M-03 — Add a smoke-test layer for untested subpackages.**
  - **What:** `models`, `data_loaders`, `optimizers`, `reconstructors`,
    `experiments`, `losses`, `metrics` have **zero** tests.
  - **Action (incremental):** Start with import smoke tests and tiny
    forward-pass/shape tests for a couple of representative models (e.g. a UNet
    post-processing model on a random tensor). This is naturally modular — one
    model/loader per PR. (Larger coverage push is L-01.)

- [ ] **M-04 — Enforce ruff (and optionally mypy) as a check.**
  - **What:** `pyproject.toml` has a large ruff ruleset and mypy config, but
    neither runs in CI or pre-commit. There are ~206 `print()` calls (ruff `T20`),
    plus many other selected rules that are currently unenforced.
  - **Action:** Add ruff to pre-commit/CI. Because the existing code will produce
    many findings, do this in stages: enable a small rule subset first (or use
    `--add-noqa`/baseline), then ratchet up. Replace library `print()`s with
    `logging` over time.

### Architecture / import hygiene

- [ ] **M-05 — Make `LION_DATA_PATH` lazy instead of import-time fatal.**
  - **What:** `LION/utils/paths.py` raises `RuntimeError` at import when the env
    var is missing. This couples *importing* a data loader to *having data
    configured*, and makes the package hard to test/inspect.
  - **Where:** `LION/utils/paths.py` (imported by 9 data-loader modules).
  - **Action:** Defer the check — expose the paths via functions/properties that
    validate on first access, or return `None`/raise only when a loader is
    actually instantiated. Keep a clear error message.

- [ ] **M-06 — Decouple operator package import from the GPU/CT stack.**
  - **What:** `LION/operators/__init__.py` eagerly imports `CTProjectionOp`, which
    imports `tomosipo`. So `import LION.operators` (even just to use `Wavelet2D` or
    `WalshHadamard2D`) drags in the whole astra/tomosipo dependency. Same pattern
    in `classical_algorithms/__init__.py` (imports `fdk`/`sirt` → tomosipo).
  - **Action:** Make CT-dependent imports lazy (import inside the class/function
    that needs them), or guard them so non-CT operators are usable without
    tomosipo installed. Add a clear error if a CT op is used without the stack.

- [ ] **M-07 — Reduce module-top `tomosipo` imports (19 files).**
  - **What:** 19 modules import `tomosipo` at module top, making large parts of the
    package unusable/uninspectable without a GPU stack.
  - **Action:** Same lazy-import strategy as M-06, applied package-wide. Pairs well
    with a documented "CT extra" (see L-04).

- [ ] **M-08 — Fix the `2deteCT` package directory name.**
  - **What:** `LION/data_loaders/2deteCT/` starts with a digit, so it is **not an
    importable Python module** (`import LION.data_loaders.2deteCT` is a syntax
    error). It can only be reached as a path, not as a package.
  - **Action:** Rename to a valid identifier (e.g. `detect_2d`/`two_detect`) or
    fold its `download.py`/`pre_process_2deteCT.py` into the existing
    `data_loaders/deteCT.py` module (see M-09).

- [ ] **M-09 — Resolve `deteCT.py` vs `2deteCT/` duplication.**
  - **What:** There is both a top-level `data_loaders/deteCT.py` (the `Dataset`
    class) and a `data_loaders/2deteCT/` folder (download + preprocess). Naming and
    responsibilities overlap confusingly.
  - **Action:** Adopt one consistent module per dataset containing
    loader+download+preprocess (or clearly separate `download`/`preprocess`
    submodules per dataset, applied uniformly across 2DeteCT/LIDC/LUNA16/walnuts).

### Docs (medium)

- [ ] **M-10 — Provide the Sphinx docs the packaging promises.**
  - **What:** `pyproject.toml` has a full `docs` extra (sphinx, myst-nb, RTD theme,
    etc.) and `[tool.ruff] exclude = ["docs/**"]`, but there is **no `docs/`
    directory**. The docs build is referenced but does not exist.
  - **Action:** Scaffold `docs/` (sphinx-quickstart + autodoc of the public API)
    so the configured extra actually builds, and wire a docs CI job. (API-doc
    quality is iterative — start with the module tree.)

- [ ] **M-11 — Per-subpackage README/landing coverage.**
  - **What:** Some folders have READMEs (`data_loaders`, `models`, `operators`),
    many do not (`optimizers`, `reconstructors`, `experiments`, `losses`,
    `metrics`). The README links to `LION/operators/_README.rst` styling that is
    inconsistent (`.rst` vs `.md`).
  - **Action:** Add short READMEs explaining each subpackage's role and entry
    points; standardise on one format.

---

## Large improvements

- [ ] **L-01 — Build out a real test suite with coverage targets.**
  - **What:** Coverage is limited to operators/classical_algorithms/utils. The bulk
    of the library (models, solvers, data loaders, experiments, reconstructors) is
    untested.
  - **Action:** Define a coverage strategy: unit tests for tensor-shape/contract
    behaviour of every model `forward`, save/load round-trips for `LIONmodel`,
    solver step correctness on tiny problems, dataset loaders against small
    fixtures. Wire `pytest-cov` + Codecov (already a declared dependency) and set a
    ratcheting coverage floor. Naturally decomposed into many M-03-style PRs.
  - **Depends on:** M-01, M-02.

- [ ] **L-02 — Decouple LION from a hard CUDA/GPU requirement (CPU fallback).**
  - **What:** The CT operators rely on astra's CUDA projectors and **abort with
    SIGFPE** on CPU. There is no CPU code path, so CI, local development, and
    onboarding all require a GPU.
  - **Action:** Investigate a CPU execution path for at least small-geometry CT
    operators (astra has CPU 2D algorithms; tomosipo is GPU-centric — may require a
    CPU shim or an alternative projector for tests). Goal: the operator either runs
    on CPU for small sizes or raises a clear, catchable error instead of crashing
    the process.
  - **Risk:** High — touches the core numerical backend; may be partially
    achievable (CPU for 2D test-sized problems only).

- [ ] **L-03 — Modernise and clarify dependency management.**
  - **What:** Dependencies are split awkwardly: conda for astra, two git-only deps
    (`tomosipo`, `ts_algorithms`) pinned by URL in `pyproject.toml`, a minimal
    `env_base.yml`, a giant pinned `env.yml`, a legacy `setup.py`, and submodules
    (`pyradiomics`, `msd_pytorch`) for "legacy" features.
  - **Action:** Define a coherent story: which deps are required vs optional
    extras, vendor/replace the git-only deps if possible, document GPU vs CPU
    installs, and converge env files. Consider a lockfile (e.g. `pip-tools`/`uv`)
    for reproducible installs. Decide the fate of the submodules (keep, vendor, or
    drop MS-D / pyradiomics).
  - **Depends on/feeds:** S-08, S-10, L-04.

- [ ] **L-04 — First-class conda-free / CPU install path + optional extras.**
  - **What:** The only documented install is conda+GPU. A reproducible
    conda-free CPU install is possible (see `CLAUDE.md`) but currently requires
    manual tarball juggling because `tomosipo`/`ts_algorithms` won't `pip install`
    under modern setuptools.
  - **Action:** Provide installable extras, e.g. `LION[cpu]` / `LION[ct]`, and a
    documented/scripted conda-free setup that works out of the box (fix or vendor
    the two git deps so `pip install` succeeds; gate CT-only deps behind an extra).
    This is what makes M-01 (CI) and onboarding painless.
  - **Depends on:** M-06, M-07, L-03.

- [ ] **L-05 — Logging, configuration, and reproducibility framework.**
  - **What:** ~206 `print()` calls, ad-hoc `warnings`, and `wandb` usage scattered
    through training code; reproducibility currently leans on `LIONmodel` saving
    commit hashes.
  - **Action:** Introduce a proper `logging` configuration, make experiment
    tracking (wandb) optional/configurable rather than implicit, and centralise
    seeding/determinism utilities. Document the reproducibility guarantees.

- [ ] **L-06 — Public API surface & stability pass.**
  - **What:** `__init__.py` files are mostly empty (`LION`, `models`,
    `data_loaders`) or eagerly import heavy modules (`operators`,
    `classical_algorithms`). There is no curated public API, no `__all__` at the
    top level, and import side-effects vary by subpackage.
  - **Action:** Define the intended public API, expose it consistently from
    `LION/__init__.py` (lazily where heavy), add `__version__`, and document
    stability expectations given the "early stage" status.
  - **Relates to:** M-06, M-07, S-11.

- [ ] **L-07 — Type-checking and static-analysis cleanup.**
  - **What:** mypy is configured (`check_untyped_defs`, strict-ish error codes) but
    never run; ruff selects ANN/type rules that are unenforced. Type coverage is
    uneven across the codebase.
  - **Action:** Stand up `mypy` in CI with a baseline, then incrementally add
    annotations and drive the error count down module by module. Pairs with M-04.
  - **Depends on:** M-04.

---

## Suggested sequencing (a starting point)

1. **Quick wins / unblockers:** S-01, S-02, S-04, S-05, M-02 (stops the SIGFPE),
   S-10/S-11.
2. **Make the project testable in CI:** M-06/M-07 (lazy imports) + L-04 (CPU
   install) → M-01 (CI runs tests) → M-04 (ruff).
3. **Harden:** M-03 → L-01 (tests), M-05 (lazy data path), M-08/M-09 (data-loader
   naming), M-10/M-11 (docs).
4. **Strategic:** L-02 (CPU fallback), L-03 (deps), L-05 (logging/repro),
   L-06 (public API), L-07 (types).

> Keep this file updated as items are completed — flip `[ ]` to `[x]` and add a
> one-line note/PR link, rather than deleting entries, so the history stays useful.
