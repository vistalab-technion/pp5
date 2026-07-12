#!/usr/bin/env python
"""Render the digitized Chevance Fig.3 per-codon values in the original genetic-code
table layout (2nd position = columns, 1st = rows, 3rd = within-cell top->bottom),
for visual comparison against the published figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

B = ["U", "C", "A", "G"]
V = {
    "UUU": 2.2,
    "UUC": 2.0,
    "UUA": 1.6,
    "UUG": 1.5,
    "UCU": 1.9,
    "UCC": 2.1,
    "UCA": 1.4,
    "UCG": 1.6,
    "UAU": 2.8,
    "UAC": 2.3,
    "UAA": 25,
    "UAG": 27,
    "UGU": 4.4,
    "UGC": 2.0,
    "UGA": 24,
    "UGG": 2.4,
    "CUU": 2.3,
    "CUC": 2.1,
    "CUA": 1.6,
    "CUG": 1.0,
    "CCU": 2.5,
    "CCC": 3.3,
    "CCA": 1.7,
    "CCG": 1.5,
    "CAU": 1.7,
    "CAC": 1.0,
    "CAA": 1.5,
    "CAG": 1.0,
    "CGU": 7.9,
    "CGC": 1.7,
    "CGA": 7.3,
    "CGG": 4.1,
    "AUU": 1.8,
    "AUC": 1.6,
    "AUA": 2.9,
    "AUG": 1.0,
    "ACU": 1.1,
    "ACC": 1.2,
    "ACA": 0.9,
    "ACG": 0.8,
    "AAU": 1.9,
    "AAC": 1.4,
    "AAA": 1.3,
    "AAG": 1.2,
    "AGU": 6.7,
    "AGC": 1.4,
    "AGA": 5.0,
    "AGG": 9.2,
    "GUU": 1.8,
    "GUC": 1.8,
    "GUA": 1.1,
    "GUG": 1.3,
    "GCU": 1.1,
    "GCC": 1.0,
    "GCA": 0.7,
    "GCG": 0.7,
    "GAU": 2.3,
    "GAC": 1.5,
    "GAA": 1.7,
    "GAG": 2.0,
    "GGU": 5.2,
    "GGC": 1.7,
    "GGA": 2.1,
    "GGG": 2.0,
}
AA = {
    "UUU": "Phe",
    "UUC": "Phe",
    "UUA": "Leu",
    "UUG": "Leu",
    "CUU": "Leu",
    "CUC": "Leu",
    "CUA": "Leu",
    "CUG": "Leu",
    "AUU": "Ile",
    "AUC": "Ile",
    "AUA": "Ile",
    "AUG": "Met",
    "GUU": "Val",
    "GUC": "Val",
    "GUA": "Val",
    "GUG": "Val",
    "UCU": "Ser",
    "UCC": "Ser",
    "UCA": "Ser",
    "UCG": "Ser",
    "CCU": "Pro",
    "CCC": "Pro",
    "CCA": "Pro",
    "CCG": "Pro",
    "ACU": "Thr",
    "ACC": "Thr",
    "ACA": "Thr",
    "ACG": "Thr",
    "GCU": "Ala",
    "GCC": "Ala",
    "GCA": "Ala",
    "GCG": "Ala",
    "UAU": "Tyr",
    "UAC": "Tyr",
    "UAA": "Stop",
    "UAG": "Stop",
    "CAU": "His",
    "CAC": "His",
    "CAA": "Gln",
    "CAG": "Gln",
    "AAU": "Asn",
    "AAC": "Asn",
    "AAA": "Lys",
    "AAG": "Lys",
    "GAU": "Asp",
    "GAC": "Asp",
    "GAA": "Glu",
    "GAG": "Glu",
    "UGU": "Cys",
    "UGC": "Cys",
    "UGA": "Stop",
    "UGG": "Trp",
    "CGU": "Arg",
    "CGC": "Arg",
    "CGA": "Arg",
    "CGG": "Arg",
    "AGU": "Ser",
    "AGC": "Ser",
    "AGA": "Arg",
    "AGG": "Arg",
    "GGU": "Gly",
    "GGC": "Gly",
    "GGA": "Gly",
    "GGG": "Gly",
}
cmap = mcolors.LinearSegmentedColormap.from_list(
    "pk", ["white", "#f7c5da", "#ec5c9f", "#c2185b"]
)
norm = mcolors.Normalize(0.7, 9.2)

fig, ax = plt.subplots(figsize=(12, 8.5))
ax.set_xlim(-0.55, 4.35)
ax.set_ylim(-0.4, 4.6)
ax.axis("off")
ax.set_aspect("equal")
ax.text(1.9, 4.5, "Second position of codon", ha="center", fontsize=13)
ax.text(
    -0.5, 2, "First position of codon (5′ end)", rotation=90, va="center", fontsize=13
)
ax.text(
    4.3, 2, "Third position of codon (3′ end)", rotation=90, va="center", fontsize=13
)
for j, b2 in enumerate(B):
    ax.text(j + 0.5, 4.18, b2, ha="center", fontsize=18, weight="bold")
for i, b1 in enumerate(B):
    ax.text(-0.30, 3.5 - i, b1, ha="center", va="center", fontsize=18, weight="bold")
    for j, b2 in enumerate(B):
        x0, y0 = j, 3 - i
        ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False, ec="k", lw=1.3))
        cys = [y0 + 1 - (k + 0.5) * 0.25 for k in range(4)]  # oval y-centers
        # amino-acid labels grouped by consecutive same AA
        k = 0
        while k < 4:
            aa = AA[b1 + b2 + B[k]]
            k2 = k
            while k2 + 1 < 4 and AA[b1 + b2 + B[k2 + 1]] == aa:
                k2 += 1
            ymid = (cys[k] + cys[k2]) / 2
            ax.text(
                x0 + 0.30,
                ymid,
                aa,
                ha="center",
                va="center",
                fontsize=12 if len(aa) <= 3 else 9,
                color="#39a0db",
                weight="bold",
            )
            k = k2 + 1
        for k, b3 in enumerate(B):
            cod = b1 + b2 + b3
            v = V[cod]
            cy = cys[k]
            stop = AA[cod] == "Stop"
            fc = "#9c0026" if stop else cmap(norm(v))
            ax.add_patch(Ellipse((x0 + 0.72, cy), 0.30, 0.18, fc=fc, ec="k", lw=0.8))
            ax.text(
                x0 + 0.72,
                cy,
                f"{v:g}",
                ha="center",
                va="center",
                fontsize=9.5,
                weight="bold",
                color="white" if (stop or v >= 5) else "black",
            )
            ax.text(
                x0 + 0.95, cy, b3, ha="center", va="center", fontsize=8.5, color="0.45"
            )
plt.tight_layout()
out = "out/pnas-2026-repro/codon_speed_table.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
