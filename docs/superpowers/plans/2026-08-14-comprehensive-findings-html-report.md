# Comprehensive Findings HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `docs/unified_project_report.html` into a comprehensive, self-contained report that presents the project's existing findings through plots, tables, scientific analysis, and a reproducibility appendix.

**Architecture:** Keep one standalone HTML artifact with inline CSS and SVG. Organize it into a progressive executive, scientific, experimental, and technical reading order; derive every plotted value from repository documents or completed run artifacts, and preserve adjacent tabular values for auditability.

**Tech Stack:** HTML5, CSS3, inline SVG, Python standard-library HTML parser, shell verification with `rg`, Git.

## Global Constraints

- The report remains a single private self-contained HTML file with no network dependency.
- Existing results must be labeled as established, encouraging, or open according to evidentiary strength.
- Missing measurements must be marked as not measured and never inferred.
- Current GM12878 values must match `runs/GM12878_multitask_full/model.log` and `runs/GM12878_multitask_full/eval_test.json`.
- Historical values must match the repository's phase, corpus, and diagnostic documents.
- The report must work for leadership readers and technical collaborators.
- The implementation must not modify unrelated dirty-worktree files.

---

### Task 1: Establish the Source-Fidelity Contract

**Files:**
- Modify: `docs/unified_project_report.html`
- Reference: `runs/GM12878_multitask_full/model.log`
- Reference: `runs/GM12878_multitask_full/eval_test.json`
- Reference: `docs/nucleosome_model_phase1_results.md`
- Reference: `docs/nucleosome_model_phase2_3_results.md`
- Reference: `docs/corpus_nucleosome_diagnostic.md`
- Reference: `docs/dataset_curation.md`

**Interfaces:**
- Consumes: numeric and categorical findings recorded in the reference files.
- Produces: HTML elements with stable section IDs and literal values that later verification can query.

- [ ] **Step 1: Record the required finding set before editing**

Use these exact result groups in the report:

```text
GM12878 loss curve: epochs 1-12, train_loss and val_loss from model.log
GM12878 best checkpoint: epoch 7, step 18942, val_loss 1367.51955389
Accessibility: count Pearson 0.6179098651; Spearman 0.5614560981;
               profile JSD 0.4511845177 vs flat 0.6437919587;
               profile Pearson 0.4866831811
Nucleosome: count Pearson 0.5989942017; Spearman 0.5027967063;
             profile JSD 0.7679896093 vs flat 0.8390724101;
             profile Pearson 0.1851320320
Classification: peak/nonpeak AUROC 0.8467328984
Reverse complement count Pearson: accessibility 0.9578812591; nucleosome 0.9640916021
cfDNA phase 1: held-out profile correlation approximately 0.679; count correlation 0.624
Receptive-field ablation: 4-mer 0.213; L1 0.632; L2 0.634; L4 0.638; L8 0.674
Corpus: 724.6 GiB; 68 pseudobulk samples; 26 studies; 23/26 insert-length viable
Multicell diagnostic: absolute profile r 0.369; split-half ceiling 0.770;
                      841 differential sites; deviation r 0.030 overall and 0.217 at called sites
```

- [ ] **Step 2: Verify the current-run source files exist**

Run:

```bash
test -s runs/GM12878_multitask_full/model.log
test -s runs/GM12878_multitask_full/eval_test.json
```

Expected: both commands exit 0.

- [ ] **Step 3: Confirm source values before implementation**

Run:

```bash
sed -n '1,20p' runs/GM12878_multitask_full/model.log
sed -n '1,240p' runs/GM12878_multitask_full/eval_test.json
```

Expected: values agree with the finding set above.

### Task 2: Rebuild the Report Reading Structure

**Files:**
- Modify: `docs/unified_project_report.html`

**Interfaces:**
- Consumes: the finding set from Task 1.
- Produces: stable sections `executive-summary`, `project-arc`, `data`, `labels`, `model`, `training`, `evaluation`, `multicell`, `roadmap`, and `appendix`.

- [ ] **Step 1: Add the semantic report shell**

Implement this hierarchy inside the existing standalone document:

```html
<nav aria-label="Report contents">
  <a href="#executive-summary">Executive Summary</a>
  <a href="#evaluation">Held-out Evaluation</a>
  <a href="#appendix">Technical Appendix</a>
</nav>
<main>
  <section id="executive-summary"><h2>Executive Summary</h2></section>
  <section id="project-arc"><h2>Scientific Project Arc</h2></section>
  <section id="data"><h2>Dataset Landscape</h2></section>
  <section id="labels"><h2>Label Construction</h2></section>
  <section id="model"><h2>Model and Training Method</h2></section>
  <section id="training"><h2>GM12878 Training Record</h2></section>
  <section id="evaluation"><h2>Held-out Evaluation</h2></section>
  <section id="multicell"><h2>Multicell Evidence</h2></section>
  <section id="roadmap"><h2>Decision Roadmap</h2></section>
  <section id="appendix"><h2>Reproducibility Appendix</h2></section>
</main>
```

- [ ] **Step 2: Add evidence labels and plot primitives**

Define reusable CSS classes for `.evidence-established`, `.evidence-encouraging`,
`.evidence-open`, `.plot`, `.plot-legend`, `.finding`, `.method-note`, `.command`, and
responsive `.table-wrap`. Include print rules that hide navigation and avoid splitting figures.

- [ ] **Step 3: Add the executive status and milestone narrative**

State the present decision explicitly: the single-cell-type PyTorch benchmark is operational;
accessibility is healthy, nucleosome counts are promising, and nucleosome profile shape is the
main improvement target before broad multicell scaling.

- [ ] **Step 4: Verify section coverage**

Run:

```bash
rg -n 'id="(executive-summary|project-arc|data|labels|model|training|evaluation|multicell|roadmap|appendix)"' docs/unified_project_report.html
```

Expected: all ten IDs are present exactly once.

### Task 3: Add Existing Findings Plots and Analyses

**Files:**
- Modify: `docs/unified_project_report.html`

**Interfaces:**
- Consumes: stable report sections from Task 2 and the verified finding set from Task 1.
- Produces: inline figures with `role="img"`, descriptive `aria-label` text, adjacent source tables, and written interpretation.

- [ ] **Step 1: Add the project-arc and corpus figures**

Create:

```text
Figure 1: milestone timeline from cfDNA prior to GM12878 held-out evaluation
Figure 2: corpus composition summary, including 68 samples/26 studies and 23/26 Gate 0 pass
Figure 3: selected study-size comparison for Camiel2023, Domcke2020, Li2023a,
          Terekhanova2023, Liang2023, Kanemaru2023, Pierce2021
```

Explain that archive size is a depth proxy rather than a biological-quality metric.

- [ ] **Step 2: Add the prior-model figures**

Create a horizontal comparison plot for the 4-mer floor and L1/L2/L4/L8 receptive-field
models. Annotate the +0.46 gap between the end-motif floor and L8 as evidence that the model
learned positioning beyond local cleavage preference; preserve the limitation that this does
not establish cell-type-specific positioning.

- [ ] **Step 3: Add the GM12878 training figure**

Plot all 12 training and validation losses as two polylines in an inline SVG. Mark epoch 7 as
the selected checkpoint and visually distinguish the later early-stopping window. Include the
full epoch table under the figure.

- [ ] **Step 4: Add the held-out evaluation figures**

Create:

```text
Figure: count-correlation comparison for accessibility and nucleosome heads
Figure: profile-JSD comparison against each head's flat baseline, with lower-is-better labeling
Figure: profile-Pearson comparison showing the nucleosome profile gap
Figure: reverse-complement count-correlation comparison
```

Every figure must include exact values in labels or the adjacent table.

- [ ] **Step 5: Add the multicell diagnostic figure**

Plot absolute profile correlation, split-half ceiling, overall deviation correlation, and
called-site deviation correlation on a common 0-1 scale. Explain why the 0.369 absolute result
and 0.030 overall deviation result answer different questions.

- [ ] **Step 6: Add interpretation blocks**

For every figure, include four compact fields: question, observed result, interpretation, and
limitation. Use evidence labels consistently.

### Task 4: Add Methods, Reproducibility, and Decision Roadmap

**Files:**
- Modify: `docs/unified_project_report.html`

**Interfaces:**
- Consumes: report and figures from Tasks 2-3.
- Produces: complete technical methods and artifact appendix.

- [ ] **Step 1: Document fragment-to-label construction**

Describe cell-type-specific Tn5 cutsites and mono-nucleosomal dyad centers, main-chromosome
filtering, Polars sparse aggregation, bigWig output, and MACS3 peak/nonpeak preparation.

- [ ] **Step 2: Document the active PyTorch architecture and run lifecycle**

Describe the shared trunk, two profile/count heads, reverse-complement augmentation,
chromosome folds, 16 workers, GPU training, early stopping, checkpoint selection, and W&B
backfill/evaluation logging. State that the owned workflow has no TensorFlow training dependency.

- [ ] **Step 3: Add reproducibility commands and artifact paths**

Include completed commands for label building, peak calling, GM12878 training, and held-out
evaluation using the repository entry points and project-local paths. Include both W&B run URLs,
checkpoint, logs, evaluation JSON, tracks, peaks, genome, and fold paths.

- [ ] **Step 4: Add the staged roadmap**

For each next stage, state its experiment, success criterion, and unlocked decision: track export
and locus inspection; K562/MCF7 matched baselines; profile-head/label tuning; shared-holdout
multicell training; orthogonal validation; corpus expansion.

- [ ] **Step 5: Add source map and glossary**

Map each historical claim group to its repository document and define dyad, cutsite, JSD,
profile Pearson, count Pearson, reverse-complement consistency, chromosome holdout, and
cell-type deviation.

### Task 5: Verify the Final Standalone Report

**Files:**
- Verify: `docs/unified_project_report.html`

**Interfaces:**
- Consumes: completed report.
- Produces: evidence that the HTML parses, required findings are present, source values match,
and no external dependencies or unresolved placeholders remain.

- [ ] **Step 1: Parse the HTML**

Run:

```bash
python3 -c "from html.parser import HTMLParser; from pathlib import Path; p=Path('docs/unified_project_report.html'); HTMLParser().feed(p.read_text()); print('html_parse_ok', p.stat().st_size)"
```

Expected: prints `html_parse_ok` and a nonzero byte count.

- [ ] **Step 2: Check required findings and plots**

Run:

```bash
rg -n "1367\.519|0\.6179|0\.5989|0\.4512|0\.7679|0\.213|0\.674|724\.6|23/26|0\.369|0\.030|0\.217" docs/unified_project_report.html
rg -c '<figure' docs/unified_project_report.html
```

Expected: every value is found and the figure count is at least eight.

- [ ] **Step 3: Check standalone and placeholder constraints**

Run:

```bash
rg -n '<script[^>]+src=|<link[^>]+href=|https://[^" ]+\.(css|js)' docs/unified_project_report.html
rg -n 'T[B]D|T[O]DO|F[I]XME|PLACEHOLD[E]R' docs/unified_project_report.html
```

Expected: both searches return no matches.

- [ ] **Step 4: Confirm only intended files changed during implementation**

Run:

```bash
git diff -- docs/unified_project_report.html
git status --short
```

Expected: report changes are visible; unrelated pre-existing changes remain unmodified.

- [ ] **Step 5: Commit the plan and report intentionally**

Run:

```bash
git add docs/superpowers/plans/2026-08-14-comprehensive-findings-html-report.md docs/unified_project_report.html
git commit -m "Expand unified nucleosome findings report"
```

Expected: one commit containing only the implementation plan and comprehensive HTML report.
