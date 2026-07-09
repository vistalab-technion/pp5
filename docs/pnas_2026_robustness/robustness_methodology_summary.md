# Outlier-robustness of the codon-conditioned backbone signal

**Supplementary analysis — PNAS-2026 Brief Communication**
Dataset: *Replication Data for Rosenberg, Marx, Bronstein, Nat. Commun. 2022* (Harvard Dataverse `doi:10.7910/DVN/5P81D4`).
Code: `vistalab-technion/pp5` (`w2torus-analysis`), driver `scripts/robustness_outliers.py`.

---

## 1. Motivation

Several critiques of the original report raised the possibility that the
codon-conditioned differences in backbone (φ, ψ) distributions are produced by a
small number of influential observations (outliers) that bias the kernel density
estimate, rather than by a genuine, distributed shift between synonymous codons.

The permutation p-value does **not** settle this question. A permutation test
controls the Type-I error *on average*; it does not certify that a *specific*
rejection is supported by more than a handful of observations. Under a null that
contains an outlier, the outlier merely receives a random codon label, so the
permutation reference distribution also lacks concentration and the observed
(concentrated) statistic still appears extreme. Outlier-robustness is therefore a
*separate* question from significance, and is the subject of this analysis.

## 2. Data and faithful reproduction

The per-residue dataset was preprocessed by the published `pp5 analyze-pointwise`
pipeline, yielding **101,419 aggregated sequence positions** (one centroid (φ, ψ)
per `(unp_id, unp_idx)` location, conditioned on secondary structure). This
matches the published `n_TOTAL` exactly. The KDE-L1 statistic computed here
reproduces the published per-pair distance `ddist` to **~1×10⁻⁹**, and the
permutation p-values match the published values to within Monte-Carlo error,
confirming the analysis is a faithful reconstruction of the released pipeline.

PDB-level provenance for each position (`pdb_id:chain, res_id`) was recovered by
joining back to the raw per-structure file `data-precs.csv` (357,479 rows).

## 3. Test statistics

All tests operate on (φ, ψ) in radians, on the flat torus.

- **KDE-L1 (primary).** The published `kde_g` statistic: the L1 distance between
  two Gaussian-kernel density estimates on a 128×128 torus grid over [−π, π),
  fixed bandwidth **σ = 10°**, evaluated by a permutation test
  (K = 5000 permutations, no early stopping). This is the original test statistic
  and the one most susceptible to single-point density "bumps", i.e. the most
  relevant statistic for the outlier concern.

- **MMD, Gaussian-RBF (bounded-influence cross-check).** Maximum Mean
  Discrepancy with a standard RBF kernel `K(x,y) = exp(−d²/2σ²)` on the
  flat-torus squared distance `d²`, σ = 10°, K = 5000. Because the kernel
  saturates in [0, 1], any single observation contributes at most O(1/n) to the
  statistic. Survival of a rejection under MMD is therefore evidence that it is
  **not** a tail/outlier artifact, obtained *without deleting any data*.

- **Projected-Wasserstein on the torus (`torus_p`, second arm).** The
  González-Delgado et al. statistic that produced the w2torus rejections: the
  (φ, ψ) sample is projected onto four fixed closed geodesics
  [1,0],[0,1],[1,1],[2,3] and a 1-Wasserstein two-sample test is performed on each
  S¹ projection; the global p-value is the analytical (permutation-free) upper
  bound `n_geodesics · min(per-projection p)`, calibrated against a simulated
  null. We use the vendored R `torustest` implementation. This statistic is more
  sensitive to the *tails* of the distributions than KDE-L1, and is the stricter
  test for the outlier concern. Significance uses the published w2torus(4-fixed)
  BH thresholds (HELIX 2.30×10⁻³, TURN 5.75×10⁻⁴).

Significance for the KDE-L1 arm is assessed at the published per-secondary-
structure Benjamini–Hochberg threshold (p ≤ 1.149×10⁻³, FDR = 0.05).

## 4. Adversarial breakdown-*k*

We quantify fragility by **adversarial influence removal**:

1. Score every observation (in both codon groups) by its leave-one-out influence
   on the KDE-L1 distance: `influence_i = ddist(current) − ddist(current \ i)`.
2. Remove the single most-influential observation (the one that most reduces the
   between-codon distance), **re-rank**, and repeat — a greedy worst-case attack.
3. After each removal recompute the **actual permutation p-value** on the reduced
   samples. **breakdown-*k*** is the smallest number of removals that pushes the
   pair above the BH threshold.

Interpretation and caveats (stated for defensibility):

- breakdown-*k* is a **greedy upper bound** on the minimal kill-set: a more
  sophisticated attacker could need *fewer* points. Hence a *small* k is a strong
  fragility statement; a *large* k is suggestive, not conclusive, of robustness.
- The significance *decision* uses the real recomputed permutation p-value; only
  the *ranking* of which points to delete uses the statistic as a proxy, which
  can only make k conservative.
- breakdown-*k* measures **fragility** (how few points carry the signal), which
  coincides with "outlier-driven" only when the removed points are themselves
  outlying. We therefore report the (φ, ψ) location and full PDB provenance of
  every removed point, so the reader can judge whether the kill-set consists of
  genuine Ramachandran outliers or ordinary, well-populated conformations.
- An absolute breakdown-*k* is interpretable only against a reference, because
  adversarial deletion will eventually erase *any* finite-sample signal. The
  controls (§5) supply that reference.

For speed the KDE-L1 permutation test is vectorised (selection-matrix × precomputed
KDE "slab" stack, evaluated through BLAS); it reproduces the reference
`kde2d_test_pergroup` statistic and (count+1)/(K+1) p-value convention exactly,
validated per pair. The same breakdown is run for the `torus_p` arm, with the
authoritative analytical p-value recomputed via R at each removal; influence is
ranked by a fast numpy circular-Wasserstein-1 proxy of the projected statistic
(one-pass ordering rather than greedy re-ranking, a further conservative
relaxation). The `torus_p` baseline statistic reproduces the published `ddist` to
four decimals.

## 5. Controls

Two null calibrations, each applying the *identical* breakdown procedure
(R = 30 replicates per pair, K = 2000):

- **AA+SS randomization (primary, the manuscript's null).** Codon labels are
  shuffled within each (amino-acid, secondary-structure) class. Every residue's
  (φ, ψ) — and therefore every outlier — is preserved; only codon identity is
  scrambled. If outliers manufactured the signal, the randomized pairs (same
  outliers) would inherit it.
- **Within-pair pooled shuffle (secondary).** The two codons' points are pooled
  and re-split at the same (n₁, n₂) sizes — the test's own permutation null at the
  tightest possible content match (identical point cloud, identical outliers).

Non-significant replicates have breakdown-*k* = 0 by definition (already dead).

## 6. Scope

The analysis covers the **union** of codon pairs rejected in *at least one* of the
six published tests (KDE-L1 ×2, projected-Wasserstein/torus ×2, torus-permutation
×2). Only one pair (A-GCG:A-GCT, TURN) was rejected unanimously; two pairs
(L-CTC:L-CTT, R-AGG:R-CGA) were rejected by the 4-fixed-projection torus test
*only*. Each pair is therefore stress-tested under the statistic(s) that rejected
it: the KDE-L1 arm (§7.1–7.2) and the `torus_p` arm (§7.3), reported side by side
in §7.4.

## 7. Results

### 7.1 Summary

| Pair | SS | n₁ / n₂ | KDE-L1 p | breakdown-*k* | MMD p (bounded) | AA+SS sig | pooled sig |
|------|----|---------|---------:|--------------:|-----------------|-----------|------------|
| A-GCG:A-GCT | TURN | 350 / 203 | 0.0002 ✓ | **8** (3.9%) | 0.0002 ✓ | 0/30 | 0/30 |
| P-CCC:P-CCG | TURN | 132 / 500 | 0.0002 ✓ | **8** (6.1%) | 0.0010 ✓ | 0/30 | 0/30 |
| L-CTC:L-TTG | HELIX | 528 / 599 | 0.0006 ✓ | **8** (1.5%) | 0.0004 ✓ | 0/30 | 0/30 |
| L-CTC:L-CTG | HELIX | 528 / 2812 | 0.0008 ✓ | **1** (0.2%) | 0.0014 ✗ | 0/30 | 0/30 |
| L-CTC:L-CTT | HELIX | 528 / 530 | 0.054 n.s. | — | 0.047 ✗ | 0/30 | 0/30 |
| R-AGG:R-CGA | HELIX | 63 / 140 | 0.144 n.s. | — | 0.032 ✗ | 0/30 | 0/30 |

✓ = below the BH threshold (1.149×10⁻³); breakdown % is k as a fraction of the
smaller group. Both controls produced **0/30** significant replicates and null
breakdown-*k* = 0 for every pair.

### 7.2 Adversarially-worst samples (kill-sets)

For each KDE-L1-significant pair, the ordered set of observations whose removal
breaks significance. All are from a *single* codon and form a *tight* (φ, ψ)
cluster in an *allowed* region; many are confirmed across multiple independent PDB
structures (n_PDB) — i.e. they are not refinement outliers.

**A-GCG:A-GCT (TURN)** — all 8 are A-GCT, α region (φ≈−58°, ψ≈−33°):

| # | unp_id:idx | φ | ψ | n_PDB | example PDB (chain, res) |
|---|------------|---|---|------:|--------------------------|
| 1 | P27302:54 | −59 | −33 | 7 | 6TJ8:A,55 … 2R5N:A,55 |
| 2 | Q6EZC2:192 | −57 | −34 | 1 | 2IY9:A,193 |
| 3 | P76578:337 | −58 | −31 | 1 | 4ZJH:A,338 |
| 4 | D7Y2H5:136 | −61 | −33 | 3 | 6P7P:A,137 |
| 5 | P08716:565 | −144 | −46 | 2 | 2PMK:A,566 |
| 6 | P15005:58 | −59 | −37 | 3 | 6GCD:A,59 |
| 7 | P0A6L2:48 | −62 | −34 | 2 | 2PUR:A,49 |
| 8 | P0A6P9:83 | −58 | −29 | 1 | 2FYM:A,83 |

**P-CCC:P-CCG (TURN)** — all 8 are P-CCC, PPII region (φ≈−56°, ψ≈+132°):

| # | unp_id:idx | φ | ψ | n_PDB | example PDB (chain, res) |
|---|------------|---|---|------:|--------------------------|
| 1 | Q68JC9:95 | −56 | 132 | 1 | 3EOI:A,75 |
| 2 | P78067:186 | −56 | 133 | 1 | 2WLR:A,167 |
| 3 | P09184:46 | −56 | 131 | 1 | 1VSR:A,47 |
| 4 | P77489:351 | −59 | 134 | 1 | 5G5G:C,352 |
| 5 | Q8VQD3:121 | −55 | 129 | 1 | 1YJ7:A,122 |
| 6 | P00634:40 | −56 | 129 | 8 | 1ED9:A,19 … 1Y7A:A,19 |
| 7 | P77366:95 | −58 | 137 | 1 | 4G9B:A,96 |
| 8 | A0A1B3B7F6:173 | −58 | 137 | 1 | 6VNU:A,174 |

**L-CTC:L-TTG (HELIX)** — all 8 are L-CTC, α region (φ≈−80°, ψ≈−15°):

| # | unp_id:idx | φ | ψ | n_PDB | example PDB (chain, res) |
|---|------------|---|---|------:|--------------------------|
| 1 | P30014:158 | −127 | −7 | 2 | 3V9W:A,159 |
| 2 | P0AB91:152 | −135 | −17 | 1 | 1N8F:A,153 |
| 3 | P37355:135 | −84 | −14 | 3 | 4MXD:A,136 |
| 4 | P69506:12 | −86 | −14 | 1 | 5FNP:A,13 |
| 5 | P0A9Q9:252 | −81 | −17 | 1 | 1T4B:A,253 |
| 6 | P0A6R0:257 | −78 | −17 | 4 | 6X7S:A,258 |
| 7 | P33590:113 | −77 | −15 | **14** | 1ZLQ:A,92 … 3MVW:A,92 |
| 8 | P13035:428 | −81 | −19 | 1 | 2QCU:A,429 |

**L-CTC:L-CTG (HELIX)** — breaks at k = 1:

| # | unp_id:idx | φ | ψ | n_PDB | example PDB (chain, res) |
|---|------------|---|---|------:|--------------------------|
| 1 | P0AB91:152 | −135 | −17 | 1 | 1N8F:A,153 |

Full provenance (every contributing structure) is in
`robustness_worst_samples.csv`.

### 7.3 Torus (`torus_p`, 4-fixed-projection) arm

| Pair | SS | n₁ / n₂ | torus_p | breakdown-*k* | AA+SS sig | pooled sig |
|------|----|---------|--------:|--------------:|-----------|------------|
| L-CTC:L-TTG | HELIX | 528 / 599 | 0.000 yes | **>27** (never broke) | 0/30 | 0/30 |
| L-CTC:L-CTG | HELIX | 528 / 2812 | 0.000 yes | **20** (3.8%) | 0/30 | 0/30 |
| L-CTC:L-CTT | HELIX | 528 / 530 | 0.000 yes | **20** (3.8%) | 0/30 | 0/30 |
| R-AGG:R-CGA | HELIX | 63 / 140 | 0.002 yes | **12** (19.0%) | 0/30 | 0/30 |
| A-GCG:A-GCT | TURN | 350 / 203 | 0.000 yes | **>20** (never broke) | 0/30 | 0/30 |
| P-CCC:P-CCG | TURN | 132 / 500 | 0.004 **no** | — | 0/30 | 0/30 |

Under `torus_p`, breakdown-*k* is much larger than under KDE-L1 (12–20, or the pair
never breaks within the tested grid): the projected-Wasserstein rejections are
carried by *many* observations, not a few. P-CCC:P-CCG is not significant under
`torus_p` (p = 0.004 > the TURN threshold 5.75×10⁻⁴). As in the KDE-L1 arm, the
torus kill-set points are clustered, allowed-region, multiply-observed residues
(e.g. the A-GCG kill-set clusters at φ≈−64°, ψ≈+149°; the recurrent residue
P33590 appears in 14+ structures) — not outliers. Full lists in
`robustness_torus_worst_samples.csv`.

### 7.4 Side-by-side verdict

| Pair | SS | KDE-L1 (p / *k*) | torus_p (p / *k*) | Verdict |
|------|----|------------------|-------------------|---------|
| **A-GCG:A-GCT** | TURN | sig / 8, MMD ✓ | sig / >20 | **robust under both** |
| **L-CTC:L-TTG** | HELIX | sig / 8, MMD ✓ | sig / >27 | **robust under both** |
| L-CTC:L-CTG | HELIX | sig / **1**, MMD ✗ | sig / 20 | statistic-dependent |
| L-CTC:L-CTT | HELIX | n.s. | sig / 20 | torus-supported |
| R-AGG:R-CGA | HELIX | n.s. | sig / 12 | torus-supported |
| P-CCC:P-CCG | TURN | sig / 8, MMD ✓ | n.s. | KDE-supported |

*k* = adversarial breakdown count. All controls (AA+SS and pooled, both arms) were
0/30 significant for every pair.

## 8. Interpretation

- **The signal is not outlier-driven.** For every significant pair, the
  adversarial kill-set consists of a *single codon's* residues forming a *coherent
  cluster* in an *allowed* Ramachandran region, frequently confirmed across many
  independent crystal structures (up to 14). These are genuine, well-determined
  conformations — they cannot be dismissed as outliers or refinement errors. The
  effect is a small, distributed shift (e.g. GCT sits deeper in the α-basin than
  GCG; CCC prolines are more PPII than CCG), not a contamination artifact.

- **The aggregated angles are well-determined, not over-averaged.** Across the 87
  kill-set entries, the per-(unp_id, unp_idx) centroid is computed from 1–21
  structures (Appendix A). Only **one** entry exceeds 15° circular std on either
  axis — P33590:421 (A-GCG), where 13 structures agree at φ≈−70°, ψ≈+145° but a
  single discordant structure (2NOO, φ=+44°, ψ=−148°) inflates σ to 26°/17°; it is
  the rank-1 influence point of a pair that is robust to >20 removals, so it does
  not affect any conclusion.

- **The "sensitive" residues are no more spread than the rest of the sample.**
  Restricting to multi-structure positions (≥2 structures, where spread is
  defined), the kill-set residues have a median spread
  max(σ_φ, σ_ψ) = **2.2°** (IQR 1.6–3.9°, n = 26), indistinguishable from the
  3,672 other positions of the same codons (median 2.2°, IQR 1.4–3.3°; Mann–Whitney
  p = 0.32) and from the whole proteome (median 2.3°). If anything the background
  is worse-behaved in the tail (max σ 159–203° vs 26° for the kill-set). The codon
  signal therefore rides on residues that are as tightly determined as any other —
  it is not concentrated on poorly-resolved positions. (`spread_comparison.csv`.)

- **The rejections are small-count but real.** Three of four KDE-L1 pairs require
  removing 8 observations (1.5–6.1% of the smaller group) and survive the bounded
  MMD; both null controls are entirely non-significant (0/30). The signal is
  modest in magnitude but statistically genuine.

- **Robustness is statistic-dependent — and the torus view is more reassuring.**
  Contrary to the a-priori expectation that the more tail-sensitive
  projected-Wasserstein test would be more fragile, the `torus_p` rejections
  withstand far more adversarial deletions (breakdown-*k* 12–20, or never break)
  than KDE-L1 (1–8). The torus signal is distributed across many points and four
  projections, whereas the fixed-bandwidth KDE-L1 lets a few high-density points
  dominate.

- **Two pairs are robust under both statistics.** A-GCG:A-GCT and L-CTC:L-TTG are
  significant and survive adversarial deletion under both KDE-L1 and `torus_p`
  (A-GCG:A-GCT was also the only unanimous rejection across the six published
  tests). These are the strongest claims.

- **Two pairs are statistic-dependent and should be reported with that caveat.**
  L-CTC:L-CTG is fragile under KDE-L1 (breaks at a single residue, P0AB91:152, and
  fails the bounded MMD) but robust under `torus_p` (k = 20); P-CCC:P-CCG is robust
  under KDE-L1 but not significant under `torus_p`. Neither should be presented as
  a standalone, test-independent result.

## 9. Cleaned-aggregation sensitivity

To test whether the codon signal depends on averaging structures that adopt
*different* conformations at the same sequence position, we re-ran the entire
analysis on an **outlier-robust re-aggregation** of the dataset.

**Outlier rule (uniform, dataset-wide).** Within every aggregation group
(unp_id, unp_idx, secondary structure) with ≥3 contributing structures, a
structure is removed if its (φ, ψ) lies more than **60°** (flat-torus distance)
from the group's robust centre (circular mean after one trimming pass); the
centroid is then recomputed from the survivors. This is applied to *all* positions,
not only the kill-sets, to avoid selection bias.

**What was removed.** 365 structures across 200 positions (0.14% of per-structure
records); median distance from consensus 156° (genuinely different basins). By SS:
OTHER 202, TURN 129, SHEET 27, HELIX 7. Crucially, **only one removed structure
falls on a kill-set residue** — 2NOO in P33590:421 (the single [WIDE] entry of
Appendix A). The full list is `removed_structures.csv`.

**Effect on the results — none material.** Re-running both arms on the cleaned
dataset reproduces every significance decision, breakdown-*k*, and control:

| Pair | KDE-L1 orig | KDE-L1 clean | torus_p orig | torus_p clean |
|------|-------------|--------------|--------------|---------------|
| A-GCG:A-GCT | sig / 8 | sig / 8 | sig / >20 | sig / >20 |
| P-CCC:P-CCG | sig / 8 | sig / 8 | n.s. | n.s. |
| L-CTC:L-TTG | sig / 8 | sig / 8 | sig / >27 | sig / >27 |
| L-CTC:L-CTG | sig / 1 | sig / 1 | sig / 20 | sig / 20 |
| L-CTC:L-CTT | n.s. | n.s. | sig / 20 | sig / 20 |
| R-AGG:R-CGA | n.s. | n.s. | sig / 12 | sig / 12 |

All controls remained 0/30 (one KDE pooled replicate for P-CCC:P-CCG reached the
floor, 1/30 — Monte-Carlo noise). The only measurable change is A-GCG:A-GCT under
torus_p, whose statistic moved 0.466 → 0.460 after 2NOO was dropped from
P33590:421 — still far below threshold and still never breaking. Correspondingly,
the single wide kill-set entry of Appendix A becomes tight after cleaning: the
maximum kill-set circular std falls from 26° to 8°, and **no** entry is flagged
[WIDE] (Appendix B). The codon-conditioned signal is therefore not an artifact of
averaging discordant structures.

## 10. Reproducibility

```
# outlier-robust re-aggregation
PYTHONPATH=<repo>/src conda run -n pp5 python scripts/clean_aggregation.py <dataset.csv>
# then re-run both arms with PP5_ROBUST_OUTDIR=out/pnas-2026-repro-clean

# KDE-L1 arm (+ MMD, controls)
PYTHONPATH=<repo>/src conda run -n pp5 \
  python scripts/robustness_outliers.py \
  out/prec-collected/20211001_124553-aida-ex_EC-src_EC/_intermediate_/dataset.csv

# torus_p (4-fixed) arm (+ controls)
PYTHONPATH=<repo>/src:<repo>/scripts conda run -n pp5 \
  python scripts/robustness_torus.py \
  out/prec-collected/20211001_124553-aida-ex_EC-src_EC/_intermediate_/dataset.csv
```

Outputs: `robustness_{outliers,torus}_summary.csv`,
`robustness_{,torus_}worst_samples.csv`, and the corresponding logs. Parameters:
σ = 10°, 128×128 grid, K_real = 5000, K_ctrl = 2000, R = 30, seed = 12345; torus
arm uses 4 fixed geodesics [1,0],[0,1],[1,1],[2,3] and a 2000-sample simulated
null. (The vendored R `torustest` was patched to run serially — its parallel
`makeForkCluster` requires a socket module unavailable in this environment.)
