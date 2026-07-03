# cfDNA Cell-Type Deconvolution via Nucleosome Positioning

**Feasibility assessment and concrete methodology**

*Author: research note for the ChromBPNet multitask (accessibility + nucleosome-dyad) project*
*Date: 2026-06-29*

---

## 0. Executive summary

Cell-free DNA (cfDNA) in plasma is, to first order, a pool of nucleosome-protected
DNA fragments produced when dying cells' chromatin is cleaved in the inter-nucleosomal
linker regions. Because nucleosome positioning is cell-type-specific, the **fragmentation
pattern of cfDNA carries a footprint of the cell types of origin**. This is established
biology (Snyder et al. 2016, *Cell*). The question is whether we can turn a cfDNA-derived
track into a **quantitative mixture decomposition** `y ≈ R w` that recovers per-cell-type
fractions, using our per-cell-type nucleosome-dyad / accessibility reference tracks as the
columns of `R`.

**Verdict: feasible, with caveats.** The reference-based linear-mixture framework is exactly
the framework that already works for methylation-based cfDNA deconvolution (Houseman 2012;
MethAtlas/Moss 2018; CelFiE 2018; Loyfer atlas 2023; MetDecode 2024), and the fragmentation/
nucleosome signal has been shown to encode tissue-of-origin (Snyder 2016) and to support
tumor subtyping (Griffin, Doebley 2022) and cancer detection (DELFI, Cristiano 2019). The
main risks are (a) collinearity between related cell types, (b) low sensitivity to minor
fractions, and (c) the fact that nucleosome-dyad signal is somewhat lower-information than
methylation per locus, so it likely needs to be combined with accessibility dips and
fragment-length features. Our ChromBPNet-style model is a real asset here: it lets us
**generate clean, depth-matched, cell-type-specific reference tracks** (including for cell
types where we lack deep data), which directly addresses the biggest weakness of every
reference-based method — reference quality.

The single recommended approach: **region-selected, non-negative, sum-to-one constrained
least squares (NNLS / quadratic programming) on a multi-feature reference matrix**, with the
reference columns built from (and denoised/imputed by) our model. Start with an in-silico
proof-of-concept on existing tracks before touching real plasma.

---

## 1. Biological basis

### 1.1 Why cfDNA encodes nucleosome positioning

The dominant source of plasma cfDNA in healthy people is hematopoietic cell turnover
(apoptosis of white blood cells). During apoptosis, endonucleases (and downstream serum
nucleases such as DNASE1L3) cleave chromatin preferentially in the **linker DNA between
nucleosomes**, because the ~147 bp wrapped around each histone octamer is sterically
protected. The result is a characteristic ladder of fragments at multiples of the
nucleosome repeat length, with a dominant mode at **~167 bp** (147 bp core + ~20 bp linker /
one chromatosome). Where a nucleosome sits, the DNA survives; where the linker is, it is
cut. Therefore the **positions of fragment boundaries (and fragment midpoints) report
nucleosome positions in the cell of origin at the moment of death**.

Because different cell types organize their chromatin differently — different genes
expressed, different regulatory elements open, different transcription-factor occupancy —
their nucleosome positioning differs, especially around promoters, enhancers, and TF
binding sites. This is the crux: **nucleosome positioning is a cell-type-specific epigenetic
fingerprint, and cfDNA fragmentation samples it non-invasively.**

### 1.2 Key prior work (real references)

**Foundational — nucleosome footprint / tissue-of-origin in cfDNA**

- **Snyder, Kircher, Hill, Daza, Shendure (2016), *Cell* 164:57–68**,
  "Cell-free DNA Comprises an In Vivo Nucleosome Footprint that Informs Its Tissues-Of-Origin."
  ([DOI](https://doi.org/10.1016/j.cell.2015.11.050)). The canonical paper. Introduced the
  **Windowed Protection Score (WPS)**: for a window, count fragments that fully span it minus
  fragments with an endpoint inside it. Peaks of WPS mark protected (nucleosome-occupied)
  positions. They showed cfDNA nucleosome spacing correlates with lymphoid/myeloid epigenomes
  in healthy donors (consistent with hematopoietic origin) and that the footprint shifts in
  cancer — i.e. tissue-of-origin is recoverable **without** relying on genetic differences.

- **Ulz et al. (2016), *Nature Genetics* 48:1273–1278**, "Inferring expressed genes by
  whole-genome sequencing of plasma DNA" ([link](https://www.nature.com/articles/ng.3648)).
  Read-depth coverage dips at transcription start sites (nucleosome depletion of active
  promoters) let them infer gene expression / cell-of-origin from plasma WGS. Establishes
  that **coverage patterns at TSS/accessible sites are cell-type-discriminative in cfDNA**.

- **Ulz et al. (2019), *Nature Communications* 10:4666**, "Inference of transcription factor
  binding from cell-free DNA enables tumor subtype prediction and early detection"
  ([link](https://www.nature.com/articles/s41467-019-12714-4)). Nucleosome footprints around
  TF binding sites in cfDNA reflect TF activity, enabling subtype prediction.

**Nucleosome / accessibility profiling frameworks**

- **Griffin (Doebley et al., 2022), *Nature Communications* 13:7475**, "A framework for
  clinical cancer subtyping from nucleosome profiling of cell-free DNA"
  ([link](https://www.nature.com/articles/s41467-022-35076-w)). Aggregates cfDNA coverage
  ("nucleosome profiling") around sets of sites (TFBS, accessibility/ATAC peaks) with a
  **fragment-size-aware GC-correction**; the central coverage dip depth reflects accessibility
  in the cells of origin. Works at coverage as low as ~0.1x. ER-subtyping AUC 0.89–0.96. This
  is essentially the cfDNA analog of reading our accessibility head, and its GC-correction
  recipe is directly reusable.

**Fragmentation / "fragmentomics" for detection**

- **DELFI (Cristiano et al., 2019), *Nature* 570:385–389**, "Genome-wide cell-free DNA
  fragmentation in patients with cancer"
  ([preprint/PDF](https://rscharpf.github.io/files/Cristiano_et_al_Nature_2019.pdf)).
  Short/long fragment ratios in 5 Mb bins; healthy profiles are "dominated by the nucleosomal
  patterns of white blood cells." Establishes fragment-length as a tissue/disease signal and
  the WBC-dominated baseline.
- Reviews: **Qi et al. 2023 *IJMS*** ([DOI](https://doi.org/10.3390/ijms24021503));
  **Ding & Lo 2022 *Diagnostics*** ([DOI](https://doi.org/10.3390/diagnostics12040978));
  **Hu, Ding & Jiang 2022** ([DOI](https://doi.org/10.20517/evcna.2022.34)). These catalog the
  fragmentomic feature menu: fragment size, end motifs, jagged ends, preferred ends,
  nucleosome footprints, open-chromatin/expression inference.

**Methylation-based deconvolution — the analogous matrix-decomposition framework**

These are the methods whose *math* we are borrowing; the only change is swapping methylation
beta-values for nucleosome/accessibility features.

- **Houseman et al. (2012), *BMC Bioinformatics* 13:86** — constrained
  projection / quadratic programming to estimate cell-mixture fractions from methylation. The
  original reference-based deconvolution recipe.
- **EpiDISH (Teschendorff et al.)** — R/Bioconductor package bundling Houseman QP/CP, Robust
  Partial Correlation (RPC), and CIBERSORT-style SVR for methylation deconvolution.
- **MethAtlas / Moss et al. (2018), *Nature Communications* 9:5068** — NNLS deconvolution of
  plasma cfDNA against a reference methylation atlas of human cell types.
- **CelFiE (Caggiano et al., 2021), *Nature Communications*** — EM-based cfDNA methylation
  deconvolution that additionally estimates contributions from **unknown** cell types not in
  the reference (relevant to our incomplete-reference problem).
- **Loyfer et al. (2023), *Nature* 613:355–364**, "A DNA methylation atlas of normal human
  cell types" ([link](https://www.nature.com/articles/s41586-022-05580-6)) — the high-quality,
  fragment-level reference atlas the field now uses; the model for what a good `R` looks like.
- **MetDecode (2024), *Bioinformatics* 40:btae522**
  ([link](https://academic.oup.com/bioinformatics/article/40/9/btae522/7739698)) and
  **cfDecon (2025, bioRxiv)** — recent supervised/deep methylation deconvolution; cfDecon
  jointly estimates proportions *and* refines cell-type signatures.
- **Comprehensive tissue deconvolution of cfDNA by deep learning (PNAS 2023)**
  ([link](https://www.pnas.org/doi/10.1073/pnas.2305236120)) — a learned (rather than linear)
  deconvolution of cfDNA for disease diagnosis/monitoring; precedent for the end-to-end variant.

**Accessibility-based deconvolution (directly analogous to our second head)**

- **DECA (2025)** — transformer that deconvolves bulk chromatin-accessibility into cell-type
  fractions using scATAC references ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11847511/)).
- **DeconPeaker** — NNLS/regression deconvolution of ATAC-seq mixtures by cell type.

**Takeaway:** every ingredient exists in published, validated form. No one (to our knowledge)
has combined a *sequence-to-nucleosome-dyad model* with reference-based cfDNA deconvolution —
that is the novel contribution our assets enable.

---

## 2. Signal extraction from cfDNA

Goal: convert a BAM of paired-end cfDNA fragments into **tracks on the same footing as our
per-cell-type reference tracks** (nucleosome-dyad profile and accessibility cut-site profile).
We want the cfDNA-derived track and the reference-track columns of `R` to be the same kind of
quantity over the same coordinates so the linear model is meaningful.

### 2.1 Pre-processing (shared)
- Paired-end, properly mapped fragments only; infer fragment intervals `[start, end)` and
  length `L`. Keep mono-nucleosomal range (~100–250 bp) for nucleosome work; optionally keep
  sub-nucleosomal (<150 bp, TF-protected / open-chromatin) separately.
- **GC and mappability correction** following Griffin's fragment-size-aware GC bias model — this
  is essential; cfDNA coverage is strongly GC-biased and biases differ across regions.

### 2.2 Track 1 — nucleosome dyad / occupancy (matches our dyad head)
Two equivalent constructions; pick the one that matches how our dyad track is defined:
- **Fragment-midpoint density.** Place each mono-nucleosomal fragment's midpoint; the midpoint
  approximates the nucleosome **dyad**. Smooth (e.g. Gaussian, ~30–50 bp) → a dyad-occupancy
  track directly comparable to our predicted/observed dyad bigWigs. (This is the cleanest
  match to a "nucleosome-dyad" head.)
- **Windowed Protection Score (WPS), Snyder 2016.** For each position, `WPS = (#fragments fully
  spanning a k-bp window) − (#fragments with an endpoint inside it)`; use a long window
  (~120 bp) for nucleosome-scale protection. WPS peaks = protected dyad-ish positions. Convert
  to occupancy by peak-calling/smoothing.

### 2.3 Track 2 — accessibility / cut-site (matches our accessibility head)
- **Coverage-dip / Griffin-style profile.** Mean GC-corrected fragment coverage around
  accessibility anchor sites; the **central dip depth** is the accessibility readout. Aggregate
  per region or keep per-bp around peaks.
- **Fragment-endpoint ("cut-site") density.** cfDNA fragment ends cluster at nucleosome
  boundaries / nuclease-accessible positions; the endpoint pile-up is the closest cfDNA analog
  to an ATAC cut-site profile. This is the natural counterpart to our cut-site head.

### 2.4 Track 3 — fragment-length features (auxiliary, strongly recommended)
- Per-region short:long ratio (DELFI-style, e.g. 100–150 vs 151–220 bp) and/or a per-region
  fragment-length histogram summary. Cheap, robust at low depth, and adds tissue-of-origin
  information orthogonal to positioning. Use as extra rows in `y` / `R` (Section 3.5).

### 2.5 Output for deconvolution
For a chosen set of **informative regions/anchors** (Section 3.4), reduce each track to a
feature vector: e.g. dyad occupancy summarized per region, accessibility dip depth per region,
short:long ratio per region. Stack into a single cfDNA feature vector `y`. Build the reference
columns `R[:,k]` by applying the *same* reduction to each cell type's reference track. Same
coordinates, same summary statistic, same normalization → the linear model is well-posed.

> Practical tooling note: WPS / fragment profiles can be computed with existing packages
> (Griffin, finaletoolkit/finaletools, cfDNA WPS pipelines) rather than from scratch; reuse the
> GC-correction from Griffin verbatim.

---

## 3. Deconvolution formulation

### 3.1 The mixture model

Let there be `n` informative features (regions × track-types) and `m` candidate cell types.

- `y ∈ ℝ^n` — the cfDNA feature vector (Section 2.5).
- `R ∈ ℝ^{n×m}` — the **reference (signature) matrix**; column `k` is cell type `k`'s
  expected feature vector (its nucleosome/accessibility/length signature).
- `w ∈ ℝ^m` — unknown cell-type fraction vector.

Linear-mixture assumption (the same one methylation deconvolution makes): the cfDNA pool is a
fraction-weighted sum of the per-cell-type contributions, so

```
            m
  y_i  ≈   Σ   R_{i,k} · w_k        for each feature i,        i.e.   y ≈ R w
           k=1
```

with biological constraints

```
  w_k ≥ 0   (non-negativity)        and        Σ_k w_k = 1   (fractions sum to one).
```

### 3.2 Estimation

**Constrained least squares (recommended default).** Solve

```
  ŵ = argmin_w  || y − R w ||_2^2     s.t.   w ≥ 0,   1ᵀ w = 1.
```

This is a small convex **quadratic program**:

```
  minimize   wᵀ (RᵀR) w − 2 (Rᵀy)ᵀ w
  s.t.       w ≥ 0,  1ᵀ w = 1.
```

Solve with NNLS (then renormalize) or, better, a proper QP solver that enforces the simplex
constraint directly (`scipy.optimize.nnls`, `cvxpy`, `quadprog`, or `scipy.optimize.lsq_linear`
+ projection). This is exactly Houseman's constrained-projection recipe and MethAtlas's NNLS.

**Robust / weighted variants.**
- **Weighted LS:** weight each feature by inverse variance (depth-dependent) — important
  because cfDNA features are heteroscedastic (low-depth regions are noisy):
  `minimize Σ_i ω_i (y_i − (Rw)_i)^2`.
- **Robust regression (RPC, EpiDISH):** Huber loss to resist outlier regions / copy-number
  artifacts.
- **Regularization:** small L2 on `w` if `R` is ill-conditioned; or a sparsity prior
  (most cell types absent) if appropriate.

### 3.3 When references are incomplete — NMF and unknown components

If we lack reference tracks for some contributing cell types (very likely early on), pure
reference-based fitting will mis-attribute their signal. Two remedies:

- **Augmented reference with an "unknown" component** (CelFiE-style): add extra free columns
  to `R` whose signatures are estimated jointly with `w`, capturing tissues not in the atlas.
- **(Semi-)NMF.** Factor the *cohort* matrix `Y ∈ ℝ^{n×s}` (`s` samples) as `Y ≈ R W`,
  `R ≥ 0, W ≥ 0`, learning both signatures `R` and loadings `W`. Use **semi-supervised NMF**:
  fix the columns we *do* have references for, learn the rest. Caveat: unsupervised factors are
  only identifiable up to scaling/rotation and need post-hoc annotation — prefer reference-based
  whenever a trustworthy `R` exists.

Comparison of the three:

| Approach | Needs reference `R`? | Identifiable fractions? | Handles unknown cell types | Best when |
|---|---|---|---|---|
| **Reference NNLS / QP** | Yes (full) | Yes (with constraints + good `R`) | No (mis-attributes) | We have trusted per-cell-type tracks (our case) |
| **NMF / semi-NMF** | No / partial | Up to rotation; needs annotation | Yes (data-driven factors) | Reference incomplete; many samples |
| **Learned / end-to-end** | Yes (training sims) | Yes if trained well | Yes (can model residual) | Lots of (sim) training data; want noise/nonlinearity handled |

### 3.4 Feature / region selection — making `R` well-conditioned

The single biggest lever on accuracy. Do **not** use the whole genome; use a curated panel of
**cell-type-discriminative regions** so `R` has well-separated columns.

- Compute, across our reference cell types, **differential nucleosome positioning** and
  **differential accessibility** regions (one-vs-rest t-statistic / F-statistic / effect size
  per region). Pick top-`t` markers per cell type (the Loyfer/MethAtlas marker-selection idea,
  applied to dyad/accessibility instead of methylation).
- Favor regions where the target cell type is an outlier vs all others (high specificity), not
  merely variable. Around TSS, enhancers, and TFBS the nucleosome signal is strongest and most
  cell-type-specific (Snyder, Ulz, Griffin).
- Quantify conditioning of the resulting `R`: condition number `κ(R)`, pairwise column
  correlations, and the minimum singular value. Prune/merge columns that are near-collinear.

### 3.5 Multi-feature stacking

Concatenate feature types into one `y`/`R` (block rows), each block z-scored within type so no
single feature type dominates:

```
       ┌ dyad-occupancy features         ┐         ┌ R_dyad ┐
  y =  │ accessibility-dip features       │ ,  R =  │ R_acc  │ ,   y ≈ R w.
       └ fragment-length (short:long)     ┘         └ R_len  ┘
```

Empirically (methylation + fragmentomics literature) combining feature types improves
minor-fraction sensitivity and conditioning relative to any single feature.

### 3.6 Identifiability & collinearity

Closely related cell types (e.g. CD4 vs CD8 T cells; subtypes of the same lineage) have similar
nucleosome landscapes → near-collinear columns → unstable individual `w_k` even when the *group*
fraction is well estimated. Mitigations: (i) marker selection that maximizes column separation;
(ii) **hierarchical deconvolution** — estimate coarse lineage fractions first, then split within
lineage only if conditioning allows; (iii) report **confidence intervals** on `w` (bootstrap
over regions, or the QP's covariance) and collapse indistinguishable types into a reported group.

---

## 4. How our model and assets plug in

We have two distinct, complementary roles for the ChromBPNet-style multitask model and its data.

### 4.1 Role (a): observed per-cell-type tracks → columns of `R`
Our per-cell-type **observed nucleosome-dyad bigWigs** (GM12878 done; many more coming from the
GCS corpus) are exactly the per-cell-type signatures the deconvolution needs. Reduce each to the
selected-region feature vector (Sections 2.5, 3.4) → that becomes `R[:,k]`. The accessibility
head's observed tracks give the accessibility block `R_acc`. This is the most direct asset: **we
already have, or are producing, the reference matrix.**

### 4.2 Role (b): the sequence model generates/denoises/imputes references
This is where our project is differentiated from methylation atlases.

- **Imputation for data-poor cell types.** For a cell type with shallow or no ATAC data, the
  model can *predict* its cell-type-specific nucleosome-dyad and accessibility tracks genome-wide
  from sequence (conditioned on the cell-type embedding/head). That predicted track becomes a
  reference column we otherwise couldn't build — directly attacking the incomplete-reference
  problem that limits CelFiE/MethAtlas.
- **Denoising / depth-normalization.** Model predictions are smooth, depth-independent
  expectations. Using *predicted* (rather than raw, depth-variable) reference tracks yields a
  cleaner, better-conditioned `R` and removes per-sample depth artifacts from the references.
- **Region selection by the model.** Use the model to find regions where predicted dyad/
  accessibility tracks **differ most across cell types** — a principled, genome-wide
  differential-signal screen for the marker panel, independent of which cell types we happened to
  sequence deeply.
- **Bias matching.** The model can produce reference tracks under the same midpoint/WPS reduction
  we apply to cfDNA, keeping `y` and `R` on identical footing.

### 4.3 Optional: learned / end-to-end deconvolution
Instead of solving a fixed QP, train a network that maps a cfDNA feature vector to `w` directly,
trained on **in-silico mixtures** generated from our reference tracks (Section 5). It can (i)
absorb nonlinearities (saturation, nuclease-bias, platform effects), (ii) learn a residual
"unknown tissue" component, and (iii) be made interpretable by structuring it around the linear
model (a learned reweighting of regions feeding a constrained final layer; cf. DECA's transformer,
cfDecon, and the PNAS 2023 deep cfDNA deconvolver). Recommended as a **phase-3 enhancement**, not
the starting point — the linear QP is the interpretable, debuggable baseline and the thing to beat.

---

## 5. Validation plan

### 5.1 In-silico mixtures (primary, do this first)
1. Take `m` per-cell-type reference tracks (start with the handful we have, including GM12878).
2. Draw a known fraction vector `w*` (Dirichlet over cell types; include realistic regimes:
   one dominant WBC-like type at 80–95%, minor tissue at 1–20%, and a hard "trace" regime at
   0.1–1% to probe detection limits).
3. Synthesize `y = R w* + noise`. Make noise realistic:
   - **Poisson/depth sampling** at target coverages (e.g. 0.1x, 1x, 5x, 30x) — cfDNA is often
     shallow; sensitivity-vs-depth is a key deliverable.
   - GC/mappability perturbations; optional fragment-length resampling.
   - A held-out cell type *not* in `R` to test robustness to incomplete references.
4. Run the deconvolver; compare `ŵ` to `w*`.

**Metrics:**
- Pearson/Spearman correlation and **RMSE / MAE** between `ŵ` and `w*` (overall and per cell type).
- **Minor-fraction detection limit:** smallest true `w_k` reliably distinguished from 0
  (ROC/AUC for "present vs absent" per cell type at each depth).
- Calibration: predicted vs true fraction slope; bias at low fractions.
- Conditioning diagnostics: `κ(R)`, sensitivity of `ŵ` to region-subset bootstrap.

### 5.2 Ablations
- Single feature vs stacked features (dyad only / accessibility only / +length).
- Observed vs model-predicted references (does Role (b) help conditioning and trace detection?).
- Marker-panel size and selection method.
- NNLS/QP vs robust (RPC) vs (semi-)NMF vs learned.

### 5.3 Real-data benchmarks (secondary)
- **Healthy plasma** should deconvolve to a predominantly hematopoietic (lymphoid/myeloid)
  composition — the Snyder 2016 sanity check; a hard requirement before trusting anything.
- **Cohorts with a known ground-truth fraction:**
  - Tumor-fraction cohorts (ctDNA fraction estimable by orthogonal means: ichorCNA from CNAs,
    or mutant-allele fraction) — does estimated tumor/epithelial fraction track it?
  - **Pregnancy** plasma (fetal/placental fraction has orthogonal estimators) — classic
    deconvolution benchmark.
  - **Transplant** plasma (donor fraction by donor-specific SNPs) — orthogonal ground truth for a
    specific-organ fraction.
  - Tissue-injury / known-perturbation cohorts where a specific cell type should rise.
- Compare against methylation deconvolution on matched samples where available (concordance of
  fractions is strong external validation).

**Success criteria (suggested):** in-silico Pearson `r ≥ 0.9` and RMSE within target at ≥1x for
the major components; detection of a 5% minor fraction at ≥5x with AUC ≥ 0.9; on healthy real
plasma, ≥ ~80–90% of mass assigned to hematopoietic lineages; on a ground-truth cohort,
estimated fraction correlates with the orthogonal estimate (`r ≥ 0.7`).

---

## 6. Risks and limitations

- **Collinearity / ill-conditioning** among related cell types (Section 3.6). The dominant
  practical risk: individual fractions of similar lineages are unstable. Mitigate with marker
  selection, hierarchical estimation, grouping, and reported uncertainty.
- **Minor-fraction sensitivity.** A 0.1–1% tumor or specific-tissue signal sits under an
  80–95% WBC background. Positioning signal per locus is weaker than methylation; trace detection
  may require high depth and the multi-feature stack. Be explicit about the detection limit per
  depth rather than over-claiming.
- **Depth requirements.** Nucleosome-position resolution needs more reads than coverage-dip or
  fragment-length features. Griffin works at ~0.1x for aggregate profiles but per-region dyad
  positioning is hungrier. Quantify the trade-off in 5.1.
- **GC / mappability bias.** Strong in cfDNA and region-dependent; must apply Griffin-style
  fragment-size-aware GC correction to both `y` and `R`, or fractions will be biased.
- **Batch / platform mismatch (the deepest concern).** Our references come from **scATAC
  pseudobulk** (a Tn5-based assay); cfDNA fragmentation is generated by **apoptotic/serum
  nucleases**. The mapping from "ATAC accessibility / dyad" to "cfDNA protection" is not
  guaranteed identical. This domain gap can bias `R` relative to `y`. Mitigations: (i) calibrate
  on the healthy-plasma sanity check; (ii) learn an affine/monotone transfer between ATAC-derived
  and cfDNA-derived signatures from paired or healthy data; (iii) prefer features known to
  transfer (TSS coverage dips, TFBS footprints — validated in cfDNA by Ulz/Griffin) over raw
  per-bp dyad amplitude; (iv) the learned variant (4.3) can absorb part of this gap.
- **Is dyad alone discriminative enough?** Likely **not** on its own for fine cell types.
  Recommendation: treat nucleosome-dyad as one channel and combine with accessibility dips,
  fragment-length, and (where available) **methylation** — methylation deconvolution is the
  most mature signal and the natural fusion partner; our contribution is adding the
  model-imputed positioning channel and showing incremental lift.
- **Reference completeness.** Missing cell types cause mis-attribution; use unknown-component
  modeling (CelFiE-style) and expand the atlas via Role (b).
- **Non-apoptotic / active release, long cfDNA, NETs** violate the clean nucleosome-ladder
  assumption in some conditions; size-gating and robust loss reduce but do not eliminate this.

---

## 7. Recommended pipeline (step by step) and phased plan

### 7.1 The pipeline (target end state)

1. **Build references `R`.** For each cell type: take observed dyad + accessibility tracks
   (Role a) and/or model-predicted tracks (Role b); apply GC/mappability correction; reduce to
   per-region features (midpoint/WPS dyad occupancy, accessibility dip depth, short:long ratio).
2. **Select markers.** Differential nucleosome-positioning + differential-accessibility regions
   across cell types (one-vs-rest); keep top markers per type; check `κ(R)` and prune collinear
   columns. Use the model to screen genome-wide for maximally cell-type-discriminative regions.
3. **Process cfDNA → `y`.** Same GC correction, same track constructions, same region reduction,
   same normalization as the references.
4. **Deconvolve.** Solve the constrained QP (`w ≥ 0`, `1ᵀw = 1`), weighted by depth, robust loss;
   add an unknown component if the reference is incomplete.
5. **Report.** Fractions with bootstrap CIs; collapse indistinguishable types into groups;
   QC against the healthy-hematopoietic prior.

### 7.2 Phased plan

- **Phase 0 — In-silico proof-of-concept (start here, uses only existing tracks).**
  Build `R` from the cell-type tracks we already have. Generate in-silico mixtures (5.1). Run
  NNLS/QP. Establish baseline recovery, detection limits vs depth, and conditioning. **Decision
  gate:** can we recover `w*` for well-separated cell types at realistic depth? This needs no
  cfDNA and no new model work — do it first.

- **Phase 1 — Reference-based NNLS/QP, expanded references.** Scale `R` to many cell types from
  the GCS corpus (observed tracks). Add marker selection, multi-feature stacking, robust/weighted
  loss, unknown component. Re-run in-silico benchmarks; add ablations (6 / 5.2).

- **Phase 2 — Model-augmented references.** Use the sequence model to impute references for
  data-poor cell types, to denoise references, and to drive region selection (Role b). Quantify
  the lift in conditioning and trace-fraction detection vs observed-only references.

- **Phase 3 — Real cfDNA + optional learned deconvolution.** Process real plasma; run the
  healthy-hematopoietic sanity check; then ground-truth cohorts (pregnancy / transplant /
  tumor-fraction). Optionally train the end-to-end learned deconvolver on in-silico mixtures and
  compare to the QP. Consider fusing methylation features if available.

### 7.3 First concrete experiment
Build `R` from the existing per-cell-type dyad tracks (GM12878 + whatever else is ready),
synthesize Poisson-downsampled in-silico mixtures at known `w*` across depths (0.1x/1x/5x/30x),
solve the constrained QP, and report correlation/RMSE of `ŵ` vs `w*` plus the minor-fraction
detection limit. This validates the whole premise with zero new data and tells us immediately
whether nucleosome-dyad signal is discriminative enough or whether we must stack accessibility +
fragment-length from the outset.

---

## 8. Phase 0 implementation (`chrombpnet.cfdna`, `chrombpnet-cfdna-poc`)

Phase 0 is implemented and validated at real scale (**146 cell types**, not just GM12878).
Package `chrombpnet/cfdna/`:

- `reference.py` — build a marker-region panel (union of each cell type's strongest peak
  summits, de-duplicated onto a genomic grid), reduce each cell type's accessibility + dyad
  tracks to per-region signal-sum feature vectors, one-vs-rest **marker selection**, and
  column-normalization into the reference matrix `R` (each cell-type column a distribution
  over regions, so `R w` with `sum(w)=1` is itself a distribution).
- `deconvolve.py` — finite-depth **multinomial** mixture simulation (depth = fragment count,
  so low depth = shot noise), simplex-constrained least squares by **projection-onto-simplex
  accelerated gradient descent** (NNLS-warm-started; no cvxpy/quadprog dependency), and
  recovery / detection / conditioning metrics.
- `run_poc.py` (`chrombpnet-cfdna-poc`) — driver: build `R`, sweep feature sets
  (acc / nuc / acc+nuc) × depths × mixture scenarios (Dirichlet with `n_active` types;
  a WBC-dominant 85%-plus-minor regime), report recovery + detection-limit tables + JSON.

**Validation (smoke config, 146 cells, 300 regions):** the loop is correct — **noiseless
recovery is exact** (Pearson 1.0, RMSE 0) across all feature sets, confirming `R` is
invertible when well-conditioned. At finite depth (1e4 fragments) fraction recovery is
**Pearson 0.90–0.99**. **Feature stacking helps conditioning materially, as predicted:**
`acc+nuc` cond(R)≈230 vs acc-only ≈4100, nuc-only ≈3060. Low-depth **precision** is the
weak point (0.2–0.6): the solver leaks a little mass onto collinear absent cell types, and
minor components under an 85% dominant fall below the detection limit at 1e4 fragments —
i.e. Phase 0 already yields a concrete **detection-limit-vs-depth** curve, the key
deliverable. Next: richer per-region positioning features (not just windowed sums),
fragment-length block, robust/regularized solve, and model-imputed references (Role b).

---

## 9. Real cfDNA test — Tao 2023 (GSE186573), first attempt (2026-07-02)

First contact with **real plasma cfDNA WGS**: Tao et al. 2023 *Cell Reports Medicine*
(GSE186573), staged at `gs://prima-mente-sequencing-public/tao_2023_cell_reports_medicine/`.
86 samples: **25 normal (NC), 26 colorectal (CRC), 35 gastric (STAD)**; per-sample merged
proper-pair fragment parquets (`processed_data_inhouse/DNA-seq/fragments/`, MAPQ≥50, columns
chromosome/start/end/strand/**sequence**). Ingestion code: `chrombpnet/cfdna/ingest.py`,
`chrombpnet/cfdna/deconvolve_real.py` (`chrombpnet-cfdna-deconvolve`).

**Data QC (NC-PKU-mix15, the healthy test sample) — all good:**
- **hg38** confirmed (chrX max-end 155.70M: above hg19 155.27M, below hg38 156.04M). Matches `R`.
- **Genuine cfDNA fragmentation:** median 170 bp, modal 165 bp (the ~167 bp mono-nucleosome
  peak), 94.5% mono-nucleosomal, 2.5% sub-nucleosomal.
- Coverage **0.28x** (~5.1M fragments) — low, typical cfDNA WGS.
- My extraction is correct: cfDNA endpoint vs midpoint per-region counts correlate 0.956.

**The deconvolution FAILED — four diagnostics, all pointing the same way:**

| # | Test | Result | Rules out |
|---|---|---|---|
| 1 | Per-region deconvolution | recon r≈0, max corr any cell type **−0.009** | direct linear signal at this resolution |
| 2 | Composite at per-study peak summits | dip sign flips **by source study** | raw summit centering (convention artifact) |
| 3 | Coverage composite at consensus sites | central **peak** not dip (−0.107) | "just aggregate more" |
| 4 | **GC-corrected** consensus composite | peak *stronger* (−0.240) | GC bias as the cause |

**Three root causes (in severity order):**
1. **Reference mismatch (was #1; now largely fixed).** The original 146-cell corpus was
   fetal/tissue and lacked adult blood — the dominant healthy-cfDNA source. **Fixed 2026-07-02
   by scaling the corpus to 315 cell types** with Granja/Lareau/Satpathy/Mimitou/Buenrostro
   blood/immune (lymphoid, monocyte, macrophage, erythroid, megakaryocyte, HSC/GMP/CLP).
   **Still no mature neutrophils/granulocytes** (PBMC excludes them; scATAC drops them) — the
   single largest healthy-cfDNA lineage, so keep the CelFiE-style "unknown" component.
2. **Depth (0.28x → 1.6 fragments/region).** Per-region resolution is Poisson noise; Phase 0
   predicted collapse here. Requires aggregate features, not per-region.
3. **Feature engineering / domain gap — the real remaining blocker.** cfDNA fragment density is
   *anti-correlated* with accessibility (Ulz coverage-dip effect; corr −0.18), so "count
   fragments in an ATAC peak" measures the inverse of what we want, buried under GC/mappability.
   The 4 diagnostics fail on the **extraction side**, independent of which cell types are in `R`.

**Verdict / next step (Griffin pipeline — "Plan A", not yet built):** the naive linear
deconvolution of ATAC references from raw cfDNA per-region density does **not** work. Need the
literature-standard aggregate approach: **GC-corrected composite coverage / WPS** over
cell-type-specific marker sites (per-fragment GC is computable from the parquet `sequence`
column — a self-vs-genome GC-bias reweighting was prototyped). **The decisive gate before
investing further** is a positive control: reproduce the canonical published **TSS coverage
dip** (Ulz/Griffin) with proper WPS — needs a GENCODE TSS set, no ATAC reference. If the TSS
dip reproduces, the method is alive and the blocker is purely feature engineering; if not,
per-sample deconvolution is not viable at 0.28x. **Start here tomorrow.**

## Appendix: notation

| Symbol | Meaning |
|---|---|
| `y ∈ ℝ^n` | cfDNA feature vector over `n` informative regions/features |
| `R ∈ ℝ^{n×m}` | reference matrix; column `k` = cell type `k` signature |
| `w ∈ ℝ^m` | unknown cell-type fraction vector (`w ≥ 0`, `1ᵀw = 1`) |
| `ŵ` | estimated fractions |
| `κ(R)` | condition number of `R` (collinearity diagnostic) |
| WPS | Windowed Protection Score (Snyder 2016) |

*Attribution: several references in this report were retrieved via PubMed; DOIs are linked
inline. Methods named (Snyder/WPS, Ulz, Griffin/Doebley, DELFI/Cristiano, Houseman, EpiDISH,
MethAtlas/Moss, CelFiE, Loyfer atlas, MetDecode, cfDecon, DECA, DeconPeaker) refer to the real
published works cited above.*
