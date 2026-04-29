"""Per-leaf AGOP diagonal — mirrors paper Figure 7B (multi-leaf comparison style).

Two horizontal bar charts (one per leaf) showing the top-12 features by AGOP
diagonal magnitude. Random features (rv1, rv2) are visually marked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "results" / "downloads" / "interpretability_appliances_energy_seed42.json"
OUT = ROOT / "figures" / "per_leaf_agop_appliances.png"

with JSON.open() as f:
    data = json.load(f)

names = data["feature_names"]
random_features = set(data.get("random_features", []))
diagonals = [np.array(d) for d in data["agop_diagonals_per_leaf"]]
n_leaves = len(diagonals)

fig, axes = plt.subplots(1, n_leaves, figsize=(11, 6.5), sharex=False)
if n_leaves == 1:
    axes = [axes]

for li, (ax, diag) in enumerate(zip(axes, diagonals)):
    # Show all features sorted descending (so the top is at the top of the bar chart)
    order = np.argsort(-diag)
    vals = diag[order][::-1]
    labels = [names[i] for i in order][::-1]
    colors = ["#d62728" if l in random_features else "#1f77b4" for l in labels]

    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("AGOP diagonal entry (normalised)", fontsize=14)
    ax.set_title(f"Leaf {li}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    # Annotate random features with their rank
    n = len(diag)
    for i, lab in enumerate(labels):
        if lab in random_features:
            display_rank = n - i  # because we reversed for display
            ax.text(vals[i] + 0.01, i, f"  ← rank {display_rank}/{n} (random noise)",
                    va="center", fontsize=10, color="#d62728")

fig.suptitle("Per-leaf AGOP diagonal — all 27 features per leaf (Appliances Energy, tuned-HP xRFM). Red bars = injected random noise.",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")

# Also print the leaf-local ranks for the random features so the report can quote them
print("\nLeaf-local random-feature ranks:")
for li, diag in enumerate(diagonals):
    order = np.argsort(-diag)
    for rf in random_features:
        idx = names.index(rf)
        rank = int(np.where(order == idx)[0][0]) + 1
        print(f"  Leaf {li}: {rf} rank={rank}/{len(diag)}, val={diag[idx]:.4f}")
