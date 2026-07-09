# PNAS 2026 Brief Communication — codon↔backbone robustness analysis

Robustness and biological-significance analysis supporting the Brief Communication
(Rosenberg, Marx, Bronstein) on whether synonymous codon identity affects protein
backbone (φ,ψ) conformation in *E. coli*, in the **original single-residue setting**
(per amino acid, comparing synonymous codons within each secondary-structure class).

## Reports
- **`statistics_comparison_report.{md,pdf}`** — §0 aggregation cleanup; Part I: five
  distributional statistics side by side (KDE-fixed 10°, KDE-CV, RBF-MMD, unbiased
  MMD², torus-W2) — rejected pairs, adversarial breakdown-k, spurious reference;
  Part II: biological significance.
- **`robustness_methodology_summary.{md,pdf}`** — full methodology for both robustness
  arms (KDE-L1 and torus projected-Wasserstein): adversarial breakdown-k, AA+SS and
  within-pair-pooled controls, PDB provenance, cleaned-aggregation sensitivity.
- **`robustness_worst_angles_appendix.{md,pdf}`** — per-structure angles ± circular
  spread for the adversarially worst samples, original vs cleaned aggregation.

## Key findings (original setting)
- Published KDE-L1 result reproduces bit-exact (101,419 positions; ddist to ~1e-9).
- Only **A-GCG:A-GCT** (Ala, TURN) and **L-CTC:L-TTG** (Leu, HELIX) are significant
  *and* breakdown-robust across all five statistics; signal is not an outlier,
  aggregation, or sample-size artifact.
- Adopt **unbiased MMD²** (removes KDE-L1 finite-sample size bias:
  Spearman(min n, KDE-L1)=−0.64 → +0.04).
- Signal does **not** track translation speed (the Nat. Commun. 2022 speed↔distance
  r²≈0.6 is a sample-size artifact); sits on rigid residues; survives expression
  stratification but concentrates in high-expression genes.

## Caveat (open — not assessed here)
A possible **sequence-context / neighbouring-residue confound** — whether the apparent
per-codon effect is partly carried by the identity of the flanking residues — is **not
evaluated in this analysis** and is **not** accounted for in the conclusions above. It
requires a dedicated multi-residue dataset and will be handled explicitly and
separately. The corresponding exploratory (multi-AA) analyses are intentionally
excluded from this commit and their conclusions are not yet trusted.

## Scripts (in `scripts/`, run from repo root)
Core statistic: `mmd_breakdown.py` (unbiased/biased MMD², flat-torus Gaussian kernel,
fast permutation, breakdown-k).

Robustness / statistics:
- `robustness_outliers.py` — KDE-L1 breakdown-k arm + controls
- `robustness_torus.py` — torus projected-Wasserstein breakdown-k arm
- `robustness_cv.py` — KDE cross-validated bandwidth (double-slab)
- `spurious_control.py` — spurious-but-significant reference for breakdown-k
- `clean_aggregation.py` — outlier-robust re-aggregation sensitivity
- `worst_sample_angles.py`, `compare_spread.py` — adversarial-sample angles & spread

Biological significance:
- `directional_speed.py`, `directional_speed_mmd.py` — translation-speed (Chevance 2014) test
- `per_aa_directional.py` — per-AA directional speed→angle trend
- `bfactor_analysis.py` — B-factor / flexibility control
- `expression_confound.py`, `expression_confound_mmd.py` — PaxDb expression stratification
- `codon_optimality.py`, `make_codon_table_fig.py` — codon speed/tRNA table & figure

## Running
```
PYTHONPATH=src:scripts conda run -n pp5 python -u scripts/<script>.py
```
Input data (aggregated `dataset.csv`, collected precs) live under `out/` (gitignored);
regenerate with the published pipeline in `docs/pnas_2026.md`.
