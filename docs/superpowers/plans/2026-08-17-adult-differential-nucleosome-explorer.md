# Adult Differential Nucleosome Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an adult-only, lineage-aware differential nucleosome-positioning caller and a notebook that interrogates its calls across cell types.

**Architecture:** A pure NumPy/Polars scoring module implements cohort validation and differential-positioning statistics. A CLI layer streams candidate windows from cell-type BigWigs and writes Parquet, BED, JSON, and NPZ artifacts; a separate notebook reads those artifacts and never performs the expensive genome-wide call.

**Tech Stack:** Python 3.11, Polars, NumPy, SciPy, pyBigWig, Matplotlib, ipywidgets, pytest, Jupyter.

## Global Constraints

- The primary comparison is adult target versus leave-one-out lineage consensus.
- Global adult one-vs-rest is a secondary context score.
- Fetal and cell-line rows are rejected explicitly from metadata, never inferred silently.
- The analysis tests positioning shape after sum normalization, not total occupancy.
- Existing 20-cell pilot artifacts are not overwritten.
- Real corpus execution remains separate from synthetic test execution.

---

### Task 1: Pure cohort and profile statistics

**Files:**
- Create: `tests/test_differential_nucleosomes.py`
- Create: `chrombpnet/multitask/differential_nucleosomes.py`

**Interfaces:**
- Produces: `validate_adult_cohort(manifest: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame`
- Produces: `normalize_profiles(raw: np.ndarray, sigma: float) -> np.ndarray`
- Produces: `score_profiles(raw: np.ndarray, cell_types: list[str], lineages: list[str], config: CallConfig) -> tuple[pl.DataFrame, dict[str, np.ndarray]]`
- Produces: `estimate_shift(target: np.ndarray, consensus: np.ndarray, max_shift: int) -> int`

- [ ] **Step 1: Write failing cohort tests**

  Cover metadata joins, adult filtering, cell-line filtering, duplicate cell types,
  missing metadata, and lineage comparator counts.

- [ ] **Step 2: Run the cohort tests and confirm failure**

  Run: `uv run pytest -q tests/test_differential_nucleosomes.py`

- [ ] **Step 3: Implement cohort validation minimally**

  Join by `cell_type`, validate unique keys and required columns, require boolean
  `is_adult`, reject `is_cell_line` when present, and preserve manifest order.

- [ ] **Step 4: Write and run failing profile-statistic tests**

  Use deterministic impulses with known shifts and Poisson count profiles. Assert
  normalization, leave-one-out consensus behavior, shift sign, robust effects, depth
  gating, and deterministic split-half estimates.

- [ ] **Step 5: Implement profile statistics and make tests pass**

  Use Gaussian sum-preserving smoothing, L1 profile normalization, median/MAD robust
  z-scores, bounded cross-correlation, and seeded binomial thinning.

### Task 2: Manifest, BigWig, candidate, and artifact runner

**Files:**
- Modify: `tests/test_differential_nucleosomes.py`
- Modify: `chrombpnet/multitask/differential_nucleosomes.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Task 1 scoring interfaces.
- Produces: `read_manifest(path: Path, root: Path | None) -> pl.DataFrame`
- Produces: `build_candidate_loci(cohort: pl.DataFrame, chroms: set[str], per_cell_limit: int | None) -> pl.DataFrame`
- Produces: `call_differential_nucleosomes(...) -> CallArtifacts`
- Produces CLI entry point: `chrombpnet-call-diff-nucleosomes`

- [ ] **Step 1: Add failing tiny-BigWig and serialization tests**

  Build temporary BigWigs with known chromosome sizes and assert candidate deduplication,
  boundary filtering, Parquet schema, BED coordinates, summary fields, and NPZ arrays.

- [ ] **Step 2: Run focused tests and confirm expected failures**

  Run: `uv run pytest -q tests/test_differential_nucleosomes.py`

- [ ] **Step 3: Implement the I/O runner and CLI**

  Resolve manifest paths, deduplicate candidate summits on genomic coordinates, read
  bounded windows from all cohort nucleosome tracks, score one locus at a time, retain
  called and configurable top-ranked loci, and write artifacts atomically.

- [ ] **Step 4: Register the CLI and analysis dependencies**

  Add `chrombpnet-call-diff-nucleosomes` under `[project.scripts]` and an `analysis`
  optional extra containing JupyterLab and ipywidgets.

- [ ] **Step 5: Run focused tests until green**

  Run: `uv run pytest -q tests/test_differential_nucleosomes.py`

### Task 3: Adult differential-nucleosome notebook

**Files:**
- Create: `tests/test_adult_diffnuc_notebook.py`
- Create: `notebooks/adult_differential_nucleosomes.ipynb`
- Create: `docs/adult_differential_nucleosomes.md`

**Interfaces:**
- Consumes: Task 2 Parquet, JSON, BED, and NPZ artifacts.
- Produces: an executable notebook with configurable `RESULT_DIR`, `TARGET_CELL`, and `LOCUS_RANK`.

- [ ] **Step 1: Write a failing notebook contract test**

  Parse notebook JSON and assert sections for configuration, cohort QC, ranked calls,
  profile overlay, deviation, heatmap, and distributions; assert no hard-coded NFS path.

- [ ] **Step 2: Run the notebook test and confirm failure**

  Run: `uv run pytest -q tests/test_adult_diffnuc_notebook.py`

- [ ] **Step 3: Create the notebook**

  Use Polars for result filtering and Matplotlib for all plots. Add ipywidgets dropdowns
  when available and a plain variable-based fallback. Load only selected profiles from
  NPZ and show target, lineage consensus, global consensus, deviations, cross-cell heatmap,
  effect distributions, and a copyable genomic locus string.

- [ ] **Step 4: Document real-corpus execution**

  Provide the adult manifest, metadata, reference root, output directory, default thresholds,
  runtime notes, and notebook launch command. State that the 20-cell pilot included fetal
  and cell-line entries and is not the adult result.

- [ ] **Step 5: Run notebook contract and syntax checks**

  Run: `uv run pytest -q tests/test_adult_diffnuc_notebook.py`

### Task 4: Verification and handoff

**Files:**
- Modify only files from Tasks 1-3 if verification exposes defects.

**Interfaces:**
- Consumes: all preceding deliverables.
- Produces: verified CLI help, focused tests, repository regression results, and a real-run command.

- [ ] **Step 1: Run focused tests**

  Run: `uv run pytest -q tests/test_differential_nucleosomes.py tests/test_adult_diffnuc_notebook.py`

- [ ] **Step 2: Run existing multitask smoke tests**

  Run: `uv run pytest -q tests/smoke_test_multitask_torch.py tests/smoke_test_pipeline_torch.py`

- [ ] **Step 3: Verify CLI help**

  Run: `uv run chrombpnet-call-diff-nucleosomes --help`

- [ ] **Step 4: Inspect the final diff and worktree scope**

  Confirm that only the new adult-analysis files plus intentional project registration
  changed and that pre-existing training edits remain untouched.

- [ ] **Step 5: Report the blocked real-data run honestly**

  Include the exact production command and state that it has not run until the NFS or GCS
  data source is available.
