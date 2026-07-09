---
title: "Codon–backbone sensitivity analysis: comparison of statistics"
date: "June 2026"
---

# Summary

After correcting the per-position aggregation (key = (unp_id, unp_idx, secondary
structure); removal of gross conformational outliers — §0, shown to be
inconsequential), we re-examined the synonymous-codon backbone signal under **five
two-sample statistics**, reporting for each (i) the FDR-rejected codon pairs and (ii) an
adversarial **breakdown-k** sensitivity analysis, calibrated against a
spurious-but-significant reference. Two pairs — **A-GCG:A-GCT (TURN)** and
**L-CTC:L-TTG (HELIX)** — are significant and breakdown-robust across statistics;
the remaining rejections are statistic-specific. We recommend the **unbiased
MMD²** statistic as the primary one, because it removes the KDE-L1 sample-size
bias by construction and (with a bounded kernel) is outlier-robust.

# 0. Data cleanup: aggregation key and conformational outliers

Before any statistic, two data-quality issues in the per-position aggregation were
fixed. Both concern how the dihedral angles of a residue are pooled across the PDB
structures in which it appears.

**(a) Aggregation key.** The original pipeline aggregated angles per
**(unp_id, unp_idx)** — one centroid per UniProt sequence position, pooling *all*
structures of that residue regardless of conformation. But a residue can adopt
different backbone states (and DSSP secondary-structure assignments) in different
structures, so pooling across structure can average genuinely distinct
conformations into a meaningless centroid. We now aggregate per
**(unp_id, unp_idx, secondary structure)**, keeping distinct conformational states
separate.

**(b) Conformational outliers.** Within each aggregation group of ≥3 structures, a
structure whose (φ,ψ) lies more than **60°** (flat-torus distance) from the group's
robust centre — i.e. in a clearly different Ramachandran basin — is removed before
averaging, and the centroid recomputed. Applied uniformly across the whole dataset.

**Size of the phenomenon.**

| effect | count | fraction |
|---|---|---|
| positions split across >1 SS class (key fix) | 1,569 | 1.5% of ~101k |
| outlier structures removed (≥3-structure groups) | 365 | 0.14% of per-structure records |
| positions affected by outlier removal | 200 | 0.2% |

Removed outliers sit a median of **156°** from their group consensus (genuinely
different basins), and are concentrated in flexible regions — by SS: HELIX 7,
SHEET 27, TURN 129, OTHER 202 (rare in well-ordered helices, common in turns/loops).

**Why this is inconsequential for our conclusions.** We re-ran the *entire* analysis
— both KDE-L1 and torus arms, all controls, all breakdown-k — on the cleaned
dataset. **Every significance decision, breakdown-k, and control outcome is
reproduced unchanged.** The reasons:

- The base reconstruction matches the original aggregation to a **median of 0.000°**
  at unaffected positions; only 200 of ~101k positions move at all.
- Of the 365 removed structures, **exactly one** falls on a kill-set
  (signal-carrying) residue — 2NOO in P33590:421, whose aggregated spread drops from
  σ = 26° to < 8° once removed.
- The signal-carrying residues were already well-determined (median circular
  std ≈ 1.8°, all in allowed Ramachandran regions), so removing outliers — which lie
  overwhelmingly in flexible regions away from the rejected pairs — leaves them
  untouched.

The only measurable change anywhere is A-GCG:A-GCT's torus statistic shifting
0.466 → 0.460 (still far below threshold, still never breaking). The phenomenon is
real and worth fixing for correctness, but it does not alter any conclusion in this
report. All results below use the cleaned, (unp_id, unp_idx, SS)-aggregated dataset.

# 1. Statistics compared

| short name | statistic | bandwidth | notes |
|---|---|---|---|
| KDE-fix | KDE-L1 on torus | fixed σ = 10° | original 2022 statistic |
| KDE-CV | KDE-L1, per-(codon,SS) σ | LOO-CV (double-slab) | revision's principled bandwidth |
| RBF-MMD | MMD² (biased, V-stat) | Gaussian, σ = 10° | bounded-influence |
| **MMD²-unbiased** | MMD² (U-stat) | Gaussian, σ = 10° | **size-unbiased + bounded** |
| torus-W2 | projected Wasserstein | 4 fixed geodesics | González-Delgado test |

All on the bit-exact reproduced dataset (101,419 positions); permutation tests
K=5000; FDR = 0.05, BH per secondary-structure class; "noself" filter; AA+SS
randomized control.

# 2. Rejected pairs (FDR = 0.05)

Candidate set = the union of pairs rejected by any statistic (6 pairs).

| statistic | # rejected | rejected pairs |
|---|---|---|
| KDE-fix | 4 | L-CTC:L-TTG, L-CTC:L-CTG (HELIX); A-GCG:A-GCT, P-CCC:P-CCG (TURN) |
| KDE-CV | 2 | L-CTC:L-TTG (HELIX); A-GCG:A-GCT (TURN) |
| RBF-MMD | 2 | L-CTC:L-TTG (HELIX); A-GCG:A-GCT (TURN) |
| MMD²-unbiased | 2 | L-CTC:L-TTG (HELIX); A-GCG:A-GCT (TURN) |
| torus-W2 | 5 | L-CTC:L-TTG, L-CTC:L-CTG, L-CTC:L-CTT, R-AGG:R-CGA (HELIX); A-GCG:A-GCT (TURN) |

(MMD significance assessed at the published per-SS BH levels — HELIX 0.00115,
TURN 0.00057 — for a common footing; a statistic-specific MMD-BH over all 87
pairs is a later refinement. P-CCC:P-CCG is a borderline miss under MMD, p≈0.0006–0.001.)

# 3. Sensitivity — adversarial breakdown-k

Greedy removal of the most influential observations (re-ranked each step) until
the pair leaves FDR significance. "robust" = never broke within cap; "boundary" =
exactly at the FDR boundary.

| SS | pair | KDE-fix | KDE-CV | RBF-MMD | **MMD²-unb.** | torus-W2 |
|---|---|---|---|---|---|---|
| HELIX | **L-CTC:L-TTG** | k=8 | boundary | k=5 | **k=5** | robust |
| HELIX | L-CTC:L-CTG | k=1 | n.s. | n.s. | n.s. | k=20 |
| HELIX | L-CTC:L-CTT | n.s. | n.s. | n.s. | n.s. | k=20 |
| HELIX | R-AGG:R-CGA | n.s. | n.s. | n.s. | n.s. | k=12 |
| TURN | **A-GCG:A-GCT** | k=8 | k=2 | k=8 | **k=8** | robust |
| TURN | P-CCC:P-CCG | k=8 | n.s. | n.s. | n.s. | n.s. |

**Caveat:** breakdown-k depends on the BH threshold, which depends on how many
rejections occur in that SS class (e.g. KDE-CV's k=2 vs KDE-fix's k=8 for
A-GCG:A-GCT is partly the tighter CV threshold, 0.00057 vs 0.00115). breakdown-k
should be read together with the p(k) trajectory, not as a single absolute number.

# 4. Sample-size sensitivity (and the fix)

KDE-L1 is a **biased** estimator of the density distance — smaller samples give
larger estimated distances:

| statistic | Spearman(min n, distance) over 316 pairs |
|---|---|
| KDE-L1 | **−0.64** (p = 2×10⁻³⁷) |
| unbiased MMD² | **+0.04** (p = 0.46, null) |

The unbiased MMD² (U-statistic, mean-zero under H₀ at any n) **removes the bias by
construction** — no subsampling needed, effect sizes comparable across pairs. With
a bounded Gaussian kernel it is simultaneously **outlier-robust**, and it uses a
single kernel width (no per-pair CV fragility). Hence the recommendation to adopt
it as the primary statistic.

# 5. A reference for breakdown-k

control breakdown-k = 0 is tautological (no AA+SS control pair is significant).
We instead manufactured **spurious-but-significant** pairs: split one amino acid's
residues in one SS into two pseudo-codon groups by a random circular-aware
direction in (φ,ψ) + tunable noise, matched to each real pair's n and p, under
unbiased MMD².

| matched to | real breakdown-k | spurious breakdown-k |
|---|---|---|
| A-GCG:A-GCT (p≈floor) | 8 | ≥21 (all reps; never broke) |
| L-CTC:L-TTG (p≈0.0008) | 5 | median 20 (bimodal: 0 or ≥20) |

Interpretation: a *genuinely distributed* difference at the same significance is
**more** breakdown-robust (k≥20) than the real pairs (5–8). So at matched p,
breakdown-k does carry information beyond the p-value — and the real codon
differences are **real but comparatively concentrated** (carried by fewer points
than a fully distributed effect). The spurious construction never yields a
*fragile* significant rejection (it is bimodal: non-significant, or robustly
significant), and the AA+SS controls give 0/30 — so the real pairs are not
competing with noise.

# 6. Conclusions

1. **A-GCG:A-GCT** and **L-CTC:L-TTG** are significant under (essentially) all five
   statistics and survive 5–8 adversarial deletions — the solid claims.
2. **L-CTC:L-CTG, L-CTC:L-CTT, R-AGG:R-CGA** are torus-only; **P-CCC:P-CCG** is
   KDE-fix-only. These four are statistic-dependent and should be reported with
   that caveat.
3. **Adopt unbiased MMD²** as the primary statistic: it removes the sample-size
   bias, is outlier-robust, and avoids bandwidth fragility — addressing the main
   methodological critiques in one move.
4. Report breakdown-k **with p(k) trajectories and the spurious reference**, not as
   a single absolute number, because it is threshold- and significance-dependent.

\newpage

# Part II — Biological significance

The statistics above establish that the codon–backbone signal is **real**. Part II
asks whether it has a **biological mechanism**, probing the four axes a reviewer
would raise. Throughout, conclusions are anchored on the two pairs that are robust
across all statistics — **A-GCG:A-GCT (TURN)** and **L-CTC:L-TTG (HELIX)** — and the
size-unbiased **unbiased MMD²** statistic is used wherever a distributional distance
is needed, so that none of the biological conclusions inherit the KDE-L1 sample-size
bias.

## 7. Structural quality — are the residues well-determined?

*Methodology.* B-factors are not comparable across structures, so we z-normalise B
within each PDB structure and average per aggregation position (B_z). We compare
B_z between the two codons with a Mann–Whitney test (a rank test on a scalar — not a
density distance, hence immune to the sample-size bias), and check each kill-set
residue against the core-allowed Ramachandran regions.

*Result.* Kill-set residues are **rigid** (median B_z ≈ −0.34, well below the
proteome mean), the two codons are **equally flexible** (MWU n.s.), and **100%** of
kill-set residues lie in allowed Ramachandran regions. The signal therefore sits on
well-ordered atoms — **not a disorder or mis-modelling artifact**. (37/63 kill-set
residues are single-PDB; these are equally rigid and allowed, but cannot be
cross-validated across structures — a residual caveat.)

## 8. Translation speed — does the shift track decoding rate?

*Data.* Experimental E. coli codon speeds from **Chevance et al. 2014** (his-leader
operon expression; high = slow), digitised from their Fig. 3 and verified against
the published figure. Speeds are compared as **ratios / log-ratios**, not
differences, since the units are arbitrary.

*Methodology.* Across all synonymous codon pairs (within AA × SS) we correlate
\|log speed-ratio\| with the structural divergence measured by **unbiased MMD²**.
This statistic is essential here: the KDE-L1 density distance is sample-size-biased
(Spearman(min n, KDE-L1) = −0.64), and that bias is exactly what produced the
spurious r² = 0.6 speed↔distance trend reported earlier (rare = slow = small-n =
inflated distance). Unbiased MMD² has Spearman(min n, MMD²) = +0.04 — no bias.

*Result.* **No relationship** — Spearman(\|log speed-ratio\|, MMD²) = +0.04
(p = 0.44). The simple co-translational-**speed** mechanism is **not supported**;
the earlier speed correlation was a sample-size artifact.

## 9. Per-amino-acid directionality — is any single residue special?

*Methodology.* A pooled test can mask an amino-acid-specific effect. For each amino
acid we correlate codon speed with the **circular mean** (φ,ψ) of its residues,
SS-centred (so secondary structure is controlled), with a permutation test over the
codon→speed labels. Circular means are used (not density distances) because they
converge cleanly with n.

*Result.* Only **proline in turns** shows a clean monotonic trend — slower proline
codon → higher ψ (toward PPII), CCG < CCA < CCU < CCC in both speed and ψ, spanning
~24° (codon-mean Spearman = 1.0). But it is **underpowered** (proline has only 4
codons → permutation floor p ≈ 0.08–0.17), **confounded with 3rd-nucleotide
identity**, and sits on a pair (P-CCC:P-CCG) that is **not robust across statistics**
(KDE-fix-only). It is a **speculative, hypothesis-generating** lead, not an
established effect.

## 10. Expression confounder (Cope–Gilchrist)

*Data.* A non-circular expression axis — **PaxDb** integrated protein abundance for
E. coli K-12 — joined to each position's gene via UniProt → locus tag (72% of
proteins mapped).

*Methodology.* Positions are split into LOW/MID/HIGH gene-abundance tertiles. Within
each stratum (where expression is ~constant) we recompute each pair's **unbiased
MMD²** and its permutation p. Because MMD² is size-unbiased, the cross-stratum effect
**magnitudes are directly comparable**.

*Result.* For the robust pairs the effect **magnitude is concentrated in
high-expression genes** (MMD² ≈ 6–20× larger in HIGH than LOW/MID: L-CTC:L-TTG
0.006/0.002/**0.034**; A-GCG:A-GCT 0.004/−0.002/**0.058**). Two implications:
(i) the effect is **present within** the high-expression stratum, so it is **not an
expression artifact** (the simplest Cope–Gilchrist account is rejected); (ii) but it
is **expression-modulated** — strongest where translational flux and codon selection
are both highest, which this observational design cannot separate. Per-stratum
*significance* is weaker than the magnitude implies, because the abundance-mapped
subset (76%), split into thirds, leaves ~¼ of the data per stratum.

## 11. Biological synthesis

| axis | finding |
|---|---|
| structural order | on rigid, well-ordered, allowed-region residues — not a disorder artifact |
| translation speed | no relationship (size-corrected) — not a clean speed mechanism |
| per-AA directionality | only proline-in-turns suggestive — speculative (underpowered, confounded) |
| expression | magnitude concentrated in high-expression genes; not an expression artifact, but expression-modulated |
| concentration | breakdown-k 5–8 vs distributed-fake ≥20 — real but carried by a sub-population |

**Defensible position:** the codon-conditioned backbone signal is **real, specific,
on well-determined residues, and not explained by expression** — yet it does **not**
behave like a clean translation-speed effect once sample size is controlled, and it
is strongest in highly-translated genes. The mechanism remains **open**, with a
concentrated, expression-linked, co-translational *flavour* but no single mechanism
established. This is the cautious framing the Brief Communication adopts, now
supported on every axis a referee is likely to probe.
