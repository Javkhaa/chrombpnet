"""In-silico cfDNA cell-type deconvolution (Phase 0 proof-of-concept).

Reference-based linear-mixture deconvolution of a cfDNA-like feature vector
`y ~ R w` into per-cell-type fractions `w` (w >= 0, sum w = 1), where the
columns of `R` are per-cell-type nucleosome-dyad / accessibility signatures
built from the multi-task model's reference tracks. See docs/cfdna_deconvolution.md
for the full methodology; this package implements Phase 0 (in-silico only, no
plasma data, no new model training).
"""
