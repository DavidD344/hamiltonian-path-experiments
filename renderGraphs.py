# plot_results.py
# Generates publication-quality plots for the Hamiltonian Path experiments CSV.
# Assumes the CSV columns:
# n,m,density,p,graph_id,bt_answer,bt_status,bt_iterations,bt_expansions,bt_time_s,
# bb_answer,bb_status,bb_iterations,bb_expansions,bb_time_s

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results_hamilton.csv"
OUT_DIR = "figs_hamilton"
DENSITY_ORDER = ["sparse", "mid", "dense"]

os.makedirs(OUT_DIR, exist_ok=True)

def _read_csv_robust(path: str) -> pd.DataFrame:
    # Your CSV uses comma as decimal separator in some places (e.g., "0,1").
    # We read as strings then normalize.
    df = pd.read_csv(path, dtype=str)

    # Normalize decimals for numeric columns
    numeric_cols = ["n", "m", "p", "graph_id",
                    "bt_iterations", "bt_expansions", "bt_time_s",
                    "bb_iterations", "bb_expansions", "bb_time_s"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize density as categorical
    if "density" in df.columns:
        df["density"] = df["density"].str.strip().str.lower()
        df["density"] = pd.Categorical(df["density"], categories=DENSITY_ORDER, ordered=True)

    # Normalize status strings
    for c in ["bt_status", "bb_status", "bt_answer", "bb_answer"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    return df


df = _read_csv_robust(CSV_PATH)

# ----------------------------
# 1) Timeout rate by density (BT vs BB)
# ----------------------------
def plot_timeout_rate(df: pd.DataFrame):
    g = df.copy()

    g["bt_timeout"] = g["bt_status"].eq("timeout")
    g["bb_timeout"] = g["bb_status"].eq("timeout")

    agg = (
        g.groupby("density", observed=True)[["bt_timeout", "bb_timeout"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(agg))

    # Grouped bars (matplotlib default color cycle)
    ax.bar([i - 0.2 for i in x], agg["bt_timeout"], width=0.4, label="Backtracking (BT)")
    ax.bar([i + 0.2 for i in x], agg["bb_timeout"], width=0.4, label="Branch-and-Bound (BB)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in agg["density"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Timeout rate")
    ax.set_title("Timeout Rate by Graph Density")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "timeout_rate_by_density.png"), dpi=300)
    plt.close(fig)


# ----------------------------
# 2) Runtime vs number of vertices (6 lines = 2 algos × 3 densities)
#    Use median runtime of successful runs (status == ok).
# ----------------------------
def plot_runtime_vs_n(df: pd.DataFrame):
    g = df.copy()

    bt_ok = g[g["bt_status"].eq("ok")].copy()
    bb_ok = g[g["bb_status"].eq("ok")].copy()

    bt_series = (
        bt_ok.groupby(["density", "n"], observed=True)["bt_time_s"]
        .median()
        .reset_index()
        .rename(columns={"bt_time_s": "time_s"})
    )
    bt_series["algorithm"] = "BT"

    bb_series = (
        bb_ok.groupby(["density", "n"], observed=True)["bb_time_s"]
        .median()
        .reset_index()
        .rename(columns={"bb_time_s": "time_s"})
    )
    bb_series["algorithm"] = "BB"

    s = pd.concat([bt_series, bb_series], ignore_index=True)
    s = s.sort_values(["density", "algorithm", "n"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for density in DENSITY_ORDER:
        for algo in ["BT", "BB"]:
            sub = s[(s["density"] == density) & (s["algorithm"] == algo)]
            if sub.empty:
                continue
            ax.plot(sub["n"], sub["time_s"], marker="o", linewidth=2, label=f"{algo} – {density}")

    ax.set_xlabel("Number of vertices (n)")
    ax.set_ylabel("Runtime (s) – median over runs (status=ok)")
    ax.set_title("Runtime vs Graph Size (G(n,p))")
    ax.grid(True, alpha=0.3)

    # Optional but usually best for exponential-ish growth:
    ax.set_yscale("log")

    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "runtime_vs_n_6lines.png"), dpi=300)
    plt.close(fig)


# ----------------------------
# 3) Status distribution (single chart, grouped bars)
#    One bar per (status), with BT and BB counts side-by-side.
# ----------------------------
def plot_status_distribution(df: pd.DataFrame):
    g = df.copy()

    # Collect statuses present
    statuses = sorted(set(g["bt_status"].dropna().unique()) | set(g["bb_status"].dropna().unique()))

    bt_counts = g["bt_status"].value_counts().reindex(statuses, fill_value=0)
    bb_counts = g["bb_status"].value_counts().reindex(statuses, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(statuses))

    ax.bar([i - 0.2 for i in x], bt_counts.values, width=0.4, label="BT")
    ax.bar([i + 0.2 for i in x], bb_counts.values, width=0.4, label="BB")

    ax.set_xticks(list(x))
    ax.set_xticklabels(statuses, rotation=0)
    ax.set_ylabel("Count of instances")
    ax.set_title("Status Distribution (All Instances)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "status_distribution_grouped.png"), dpi=300)
    plt.close(fig)


# ----------------------------
# 4) Scatter: BT time vs BB time (points colored by density)
#    Only instances where both are status=ok.
# ----------------------------
def plot_bt_vs_bb_scatter(df: pd.DataFrame):
    g = df.copy()
    g = g[g["bt_status"].eq("ok") & g["bb_status"].eq("ok")].copy()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Plot per density so matplotlib assigns distinct colors automatically
    for density in DENSITY_ORDER:
        sub = g[g["density"] == density]
        if sub.empty:
            continue
        ax.scatter(sub["bt_time_s"], sub["bb_time_s"], alpha=0.8, label=str(density))

    # Reference line y=x
    # choose limits from data
    all_x = g["bt_time_s"].dropna()
    all_y = g["bb_time_s"].dropna()
    if len(all_x) and len(all_y):
        mn = min(all_x.min(), all_y.min())
        mx = max(all_x.max(), all_y.max())
        ax.plot([mn, mx], [mn, mx], linewidth=1.5, linestyle="--", label="y = x")

    ax.set_xlabel("Backtracking runtime (s)")
    ax.set_ylabel("Branch-and-Bound runtime (s)")
    ax.set_title("Runtime Comparison per Instance (BT vs BB)")
    ax.grid(True, alpha=0.3)

    # log scales usually make this plot readable (times span orders of magnitude)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scatter_bt_vs_bb_by_density.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_timeout_rate(df)
    plot_runtime_vs_n(df)
    plot_status_distribution(df)
    plot_bt_vs_bb_scatter(df)

    print(f"OK: plots saved in: {OUT_DIR}/")
