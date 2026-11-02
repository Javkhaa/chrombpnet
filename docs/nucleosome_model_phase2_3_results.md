# Phase 2 & 3 — end-motif/bias and cell-type conditioning

_2026-07-26. Follows `nucleosome_model_phase1_results.md` (sequence→endpoint model,
held-out profile corr 0.68, NRL recovery). Phase 2 = the cfDNA cleavage-bias / end-motif
signal and an ablation separating it from positioning. Phase 3 = the FiLM cell-type
conditioning machinery and an honest account of its data limitation._

## Phase 2 — end-motif (block D) and the bias/positioning separation

### 2a. End-motif spectrum

The 5′ end 4-mer of each fragment reads out the apoptotic nuclease's cut-site sequence
preference. Over 5M IH01 fragments (both strand ends, 10M motifs):

- **MDS (motif diversity score) = 0.9199** — high, consistent with healthy plasma.
- **Top motif CCCA (3.57%)**, followed by CCTG, CCAG, CCCT, CCCC — all C-initiated, the
  canonical DFFB/DNASE1L3 preference (Jiang et al. 2020).
- Bottom motifs are CpG-containing (TTCG, ATCG, TCGG, GTCG, ~0.01%) — CpG depletion at ends.

This is the fragmentomics *block D* feature, reproduced and matching the literature.
Assets: `phase2_endmotif.py`, `models/endmotif_IH01.npz`.

### 2b. Receptive-field ablation — is positioning just cut-site bias?

Same head/data, trunk depth (≈ receptive field) swept; held-out chr1+chr9:

| Model | Receptive field | profile_corr | NRL peak |
|---|---|---|---|
| **End-motif floor** (pure 4-mer bias) | 4 bp | **0.213** | — |
| L=1 | ~100 bp | 0.632 | 200 bp ✓ |
| L=2 | ~130 bp | 0.634 | 200 bp ✓ |
| L=4 | ~250 bp | 0.638 | 200 bp ✓ |
| L=8 (phase-1 model) | ~1 kb | 0.674 | 200 bp ✓ |

**Positioning is not cut-site bias.** The pure 4-mer end-motif predicts endpoint density at
only 0.213; the model reaches 0.63–0.67 — a **+0.46 gap** that is genuine nucleosome
positioning, not local cleavage preference.

**Most of positioning is encoded at ~100 bp** — one nucleosome's worth of sequence. L=1
(~100 bp) already captures 0.63 and recovers the ~190 bp periodicity; extending to ~1 kb
adds only +0.04 (0.63→0.67), the multi-nucleosome phasing contribution.

Caveat on NRL: every depth "passes" NRL because even L=1 has ~100 bp effective context (the
k=75 profile head), enough to resolve one ~190 bp period. NRL is therefore not the
discriminator between bias and positioning here — the **profile_corr vs the 0.213 end-motif
floor** is. Assets: `phase2_ablation.sh`, `models/nuc_L{1,2,4,8}.pt`.

## Phase 3 — cell-type conditioning (FiLM): machinery + data limitation

### What was built

`CondSeqNucleosomeModel` (`train_nuc_cond.py`): the phase-1 trunk with a per-condition
**FiLM** modulation (scale+bias per layer, on a condition embedding) and one head. FiLM is
**identity-initialized** (γ=β=0 → scale 1, bias 0), verified: at init, two conditions
produce *identical* output. The model therefore starts equal to the unconditioned phase-1
model and can only *learn* per-condition deviation — the design's guard against the Role-b
collapse (a conditioned model regressing to the invariant backbone).

Per-sample endpoint tracks built (`aligned/{IH01,BH01}_endpoints_hg38.bw`, 24/24 chroms).
Conditioned model trains and reports per-condition held-out profile_corr (ep0: IH01 0.583,
BH01 0.492, both climbing toward the ~0.68 phase-1 level).

### The honest limitation

**This cannot yet demonstrate a cell-type result, because the data has no cell-type
contrast.** BH01 and IH01 are both healthy plasma, and plasma cfDNA is compositionally
dominated by one lineage (hematopoietic) — the same fact that made cell-type
*deconvolution* hard in the first place. Conditioning on two healthy samples measures
sample-to-sample/technical difference (depth 101× vs 96×), not cell-type deviation.

So phase 3's deliverable is the **infrastructure**, not a biological claim:
- the FiLM-conditioned architecture, identity-safe by construction;
- per-condition held-out evaluation;
- the explicit protocol requirement (from phase-1's risk section): **evaluate on
  cell-type deviation, never on absolute prediction** — a model that nails the global
  backbone and learns zero residual would still score ~0.68.

### What would unblock a real result

A conditioning signal with genuine nucleosome-landscape contrast, e.g.:
- **cell-type-resolved** nucleosome maps (the missing atlas — does not exist for cfDNA);
- **high-ctDNA** samples where tumor-derived fragments meaningfully shift positioning
  (low-ctDNA plasma shows little, as our cancer-detection negatives already showed);
- non-plasma tissue nucleosome maps (MNase/other), accepting a domain gap.

The Snyder per-condition tracks (14 disease conditions, hg38) are the nearest available
weak supervision, but plasma's lineage dominance caps the achievable contrast. Pulling a
Snyder cancer sample was deliberately **not** done autonomously — ~5 h of alignment for a
likely-small, low-ctDNA contrast is a call for the user, not a default.

## Bottom line

- **Phase 2:** the cfDNA cleavage bias (end-motif, block D) is captured and *separated* —
  positioning is +0.46 above the end-motif floor, and is largely a ~100 bp local sequence
  grammar with a smaller long-range phasing component.
- **Phase 3:** the cell-type conditioning machinery is built and identity-safe, but a real
  cell-type result is data-blocked by plasma's lineage dominance. The infrastructure is
  ready for cell-type-contrastive data if/when it exists.

## Assets

- `phase2_endmotif.py`, `models/endmotif_IH01.npz`
- `phase2_ablation.sh`, `models/nuc_L{1,2,4,8}.pt`
- `train_nuc_cond.py`, `models/nuc_cond.pt`, `aligned/{IH01,BH01}_endpoints_hg38.bw`
