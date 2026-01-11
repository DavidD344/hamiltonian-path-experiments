import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, os

CSV_PATH = "./results_hamilton.csv"
OUT_CSV  = "./imgs/timeout_counts_by_n_p.csv"
OUT_PNG  = "./imgs/timeout_rate_by_n_p.png"

os.makedirs("./imgs", exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(CSV_PATH, dtype=str)

def to_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return float(str(x).replace(",", "."))

df["n"] = df["n"].map(to_float).astype(int)
df["p"] = df["p"].map(to_float).astype(float)

# ---------- Aggregate TIMEOUT rate ----------
rows = []
for (n, p), g in df.groupby(["n", "p"]):
    total = len(g)
    for alg in ["bt", "bb"]:
        status = g[f"{alg}_status"].fillna("missing")
        timeout = int((status == "timeout").sum())
        rows.append({
            "n": n,
            "p": p,
            "alg": alg,
            "total": total,
            "timeout": timeout,
            "timeout_rate": timeout / total if total else 0.0
        })

summary = pd.DataFrame(rows).sort_values(["p", "n", "alg"])
summary.to_csv(OUT_CSV, index=False)

# ---------- Plot: TIMEOUT rate vs n (one panel per p) ----------
p_values = sorted(summary["p"].unique())
fig, axes = plt.subplots(
    1, len(p_values),
    figsize=(4.8 * len(p_values), 3.8),
    sharey=True
)

if len(p_values) == 1:
    axes = [axes]

for ax, p_val in zip(axes, p_values):
    sub = summary[summary["p"] == p_val].sort_values("n")

    for alg, label in [("bt", "Backtracking (BT)"),
                       ("bb", "Branch-and-Bound (BB)")]:
        s = sub[sub["alg"] == alg]
        ax.plot(s["n"], s["timeout_rate"], marker="o", label=label)

    ax.set_title(f"p = {p_val:g}")
    ax.set_xlabel("n (vertices)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.8)

axes[0].set_ylabel("Timeout rate (status = timeout)")

# Title (single, clean)
fig.text(
    0.5, 0.97,
    "Timeout rate by instance size (n) and edge probability (p)",
    ha="center", va="top", fontsize=12
)

# Legend (below title)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.92)
)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close(fig)
