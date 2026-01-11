import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, os

CSV_PATH = "./results_hamilton.csv"
OUT_PNG = "./imgs/figC_expansions_vs_n.png"
EPS = 1e-6

os.makedirs("./imgs", exist_ok=True)

df = pd.read_csv(CSV_PATH, dtype=str)

def to_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return float(str(x).replace(",", "."))

for c in ["n", "p", "bt_expansions", "bb_expansions"]:
    df[c] = df[c].map(to_float)

df["n"] = df["n"].astype(int)
df["p"] = df["p"].astype(float)

rows = []
for alg in ["bt", "bb"]:
    # você pode escolher "ok" apenas, ou incluir timeout também.
    # Eu recomendo "ok" para comparar esforço até achar/provar.
    sub = df[df[f"{alg}_status"] == "ok"]

    for (density, n), g in sub.groupby(["density", "n"]):
        vals = g[f"{alg}_expansions"].astype(float).to_numpy()
        rows.append({
            "density": density,
            "n": n,
            "alg": alg,
            "median": float(np.median(vals)),
            "p95": float(np.quantile(vals, 0.95)),
            "p": float(g["p"].iloc[0]),
        })

agg = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
densities = ["sparse", "mid", "dense"]
colors = {"bt": "#1f77b4", "bb": "#2ca02c"}

for ax, d in zip(axes, densities):
    sub = agg[agg["density"] == d].sort_values("n")
    p_val = sub["p"].iloc[0] if len(sub) else None

    for alg, label in [("bt", "Backtracking (BT)"),
                       ("bb", "Branch-and-Bound (BB)")]:
        s = sub[sub["alg"] == alg]
        ax.plot(s["n"], s["median"] + EPS, marker="o", linewidth=1.8,
                color=colors[alg], label=label)
        ax.plot(s["n"], s["p95"] + EPS, linestyle="--", linewidth=1.0,
                color=colors[alg], alpha=0.5)

    ax.set_title(f"{d}" + (f" (p={p_val:g})" if p_val is not None else ""))
    ax.set_xlabel("n (vertices)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.8)

axes[0].set_ylabel("Expansions\nmedian (solid), p95 (dashed)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Figure C — Search effort vs n (successful runs only)", y=1.10, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close(fig)
