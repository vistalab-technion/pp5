"""Shared constants for the PNAS-2026 codon<->backbone robustness scripts."""

# Union of codon pairs rejected by at least one of the 6 published tests
# (KDE-L1 x2, torus_p x2, torus_perm x2) -- the stress-test set every
# robustness arm (MMD, KDE-fix, KDE-CV, torus) is evaluated on.
PAIRS = [
    ("HELIX", "L-CTC", "L-TTG"),
    ("HELIX", "L-CTC", "L-CTG"),
    ("HELIX", "L-CTC", "L-CTT"),
    ("HELIX", "R-AGG", "R-CGA"),
    ("TURN", "A-GCG", "A-GCT"),
    ("TURN", "P-CCC", "P-CCG"),
]

# Shared seed for permutation tests and AA+SS/pooled control replicate
# generation across the robustness scripts. The torus arm's null-simulation
# seed (42) is a separate, unrelated constant (see robustness_torus.py's
# NULL_SIM_SEED) -- it seeds the R torustest null distribution, not anything
# here.
SEED = 12345
