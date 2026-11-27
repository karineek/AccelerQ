"""
Objective 2 analysis: Hamiltonian structural correlation.

Produces structural correlation plots from records and per-implementation ham
CSVs produced by stages 1–3. Runnable independently after the pipeline completes.

SHAP attributions live in kcl_obj4_analyse.py (outputs under ../data/O4/).

Writes under ../data/O2/:
  {HEA,UCCSD}_per_ham_summary.csv
  {HEA,UCCSD}_{feature}_vs_error.png
  cross_impl_error_comparison.png

Usage:
    python3 kcl_obj2_analyse.py

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

STRUCTURAL_IMPLS = ["HEA", "UCCSD"]

IMPL_DATA_DIRS = {
    "HEA": "../data/HEA",
    "UCCSD": "../data/UCCSD",
}

RECORDS_CSV = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"records_{impl}.csv")
    for impl in STRUCTURAL_IMPLS
}

# Per-implementation ham CSVs written by stage 1 mining (kcl_prepare_data.miner)
HAM_CSV = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"hamiltonian_{impl}.csv")
    for impl in STRUCTURAL_IMPLS
}

OUTPUT_DIR = "../data/O2/"

STRUCTURAL_FEATURES = ["sparsity", "term_count", "coefficient_variance"]

# Minimum test samples per Hamiltonian to include in correlation analysis.
# Hamiltonians with fewer test records than this threshold are excluded to
# avoid unreliable mean error estimates from very small samples.
MIN_TEST_SAMPLES_PER_HAM = 4

_X_FILE_SEED_RE = re.compile(r"^(\d+)qubits_(\d+)")


def _parse_seed_from_x_file(x_file):
    """Parse (n_qubits, seed) from stems like 04qubits_05A.X.data; strips trailing 'A'."""
    if not x_file or not isinstance(x_file, str):
        return None, None
    stem = x_file
    if stem.endswith(".npy"):
        stem = stem[:-4]
    # e.g. 04qubits_05A.X.data -> 04qubits_05
    prefix = stem.split(".")[0]
    if prefix.endswith("A") and len(prefix) > 1:
        prefix = prefix[:-1]
    match = _X_FILE_SEED_RE.match(prefix)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def load_analysis_data(impl_name):
    """Join test success records to ham CSV on ham_id; None if either file is missing."""
    records_path = RECORDS_CSV[impl_name]
    ham_path = HAM_CSV[impl_name]

    if not os.path.exists(records_path):
        print(f"[WARN] {impl_name}: records CSV not found at {records_path}")
        return None

    if not os.path.exists(ham_path):
        print(f"[WARN] {impl_name}: ham CSV not found at {ham_path}")
        return None

    print(f"[{impl_name}] Loading records from {records_path}")
    records = pd.read_csv(records_path)
    records = records[
        records["split"].isin(["test", "test_extra"]) & (records["status"] == "success")
    ].copy()

    print(f"[{impl_name}] Loading ham metadata from {ham_path}")
    ham_df = pd.read_csv(ham_path)

    records["ham_id"] = pd.to_numeric(records["ham_id"], errors="coerce")
    ham_df["ham_id"] = pd.to_numeric(ham_df["ham_id"], errors="coerce")

    merged = records.merge(ham_df, on="ham_id", how="inner")
    if merged.empty:
        print(f"[WARN] {impl_name}: join on ham_id produced no rows")
        return None

    seeds = merged["x_file"].apply(lambda x: _parse_seed_from_x_file(x)[1])
    merged["seed"] = seeds

    print(f"[{impl_name}] Merged {len(merged)} test records across "
          f"{merged['ham_id'].nunique()} Hamiltonians")
    return merged


def compute_per_ham_stats(df):
    """Per-ham mean errors + structural cols; drops hams below MIN_TEST_SAMPLES_PER_HAM."""
    df = df.copy()
    df["predicted_energy"] = pd.to_numeric(df["predicted_energy"], errors="coerce")
    df["true_energy"] = pd.to_numeric(df["true_energy"], errors="coerce")
    df["squared_error"] = pd.to_numeric(df["squared_error"], errors="coerce")
    df["abs_error"] = (df["predicted_energy"] - df["true_energy"]).abs()

    meta_cols = ["n_qubits", "seed", "dataset"] + [
        c for c in STRUCTURAL_FEATURES if c in df.columns
    ]

    grouped = df.groupby("ham_id", as_index=False).agg(
        mean_squared_error=("squared_error", "mean"),
        mean_abs_error=("abs_error", "mean"),
        n_test_samples=("record_id", "count"),
        **{col: (col, "first") for col in meta_cols},
    )

    n_before = len(grouped)
    excluded = grouped[grouped["n_test_samples"] < MIN_TEST_SAMPLES_PER_HAM]
    grouped = grouped[grouped["n_test_samples"] >= MIN_TEST_SAMPLES_PER_HAM].copy()

    if len(excluded) > 0:
        print(
            f"Excluded {len(excluded)} Hamiltonian(s) with fewer than "
            f"{MIN_TEST_SAMPLES_PER_HAM} test samples "
            f"(had {n_before}, kept {len(grouped)})"
        )

    return grouped


def plot_structure_vs_error(per_ham_df, impl_name, feature, output_dir):
    """Scatter feature vs MSE; colour=n_qubits, area=n_test_samples (no provenance encoding)."""
    plot_df = per_ham_df.dropna(subset=[feature, "mean_squared_error"]).copy()
    if plot_df.empty:
        print(f"[SKIP] {impl_name} {feature}: no data to plot")
        return

    x = plot_df[feature].values.astype(float)
    y = plot_df["mean_squared_error"].values.astype(float)
    n_qubits = pd.to_numeric(plot_df["n_qubits"], errors="coerce").values
    sizes = plot_df["n_test_samples"].values.astype(float)
    size_scale = 45.0 * (sizes / max(sizes.min(), 1.0))

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    scatter = ax.scatter(
        x, y, s=size_scale, c=n_qubits, cmap="viridis", marker="o",
        edgecolors="k", linewidths=0.4, alpha=0.6, zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("n_qubits", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    # Dashed trend line for visual reference only (not a fitted model)
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k--", linewidth=1,
                alpha=0.6, zorder=2)

    rho, p_value = scipy.stats.spearmanr(x, y)
    sig_tag = ""
    if p_value < 0.05:
        sig_tag = " (significant)"
    elif p_value < 0.10:
        sig_tag = " (marginal)"

    # headroom so the annotation box cannot sit on top of a data point
    ax.margins(x=0.08)
    y_lo, y_hi = float(y.min()), float(y.max())
    span = max(y_hi - y_lo, 1e-9)
    ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.30 * span)

    ax.annotate(
        f"Spearman $\\rho$ = {rho:.3f}, p = {p_value:.3g}{sig_tag}",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=12, zorder=6,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )

    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel("mean squared error", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(f"{impl_name} — {feature} vs prediction error", fontsize=15)
    ax.grid(alpha=0.25)

    out_path = os.path.join(output_dir, f"{impl_name}_{feature}_vs_error.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{impl_name}] Saved {out_path}")

def plot_cross_impl_error(per_ham_dfs, output_dir):
    """Grouped MSE bars matched across impls by (n_qubits, seed), not ham_id."""
    impl_names = list(per_ham_dfs.keys())
    if len(impl_names) < 2:
        return

    mse_by_key = {}
    for impl, df in per_ham_dfs.items():
        for _, row in df.iterrows():
            n_qubits = row.get("n_qubits")
            seed = row.get("seed")
            if pd.isna(n_qubits) or pd.isna(seed):
                continue
            key = (int(n_qubits), int(seed))
            mse_by_key.setdefault(key, {})[impl] = row["mean_squared_error"]

    common_keys = [k for k, v in mse_by_key.items() if len(v) == len(impl_names)]
    if not common_keys:
        print("[SKIP] cross_impl_error: no Hamiltonians shared across all implementations")
        return

    common_keys.sort(key=lambda k: (k[0], k[1]))
    labels = [f"{nq:02d}q_{seed:02d}" for nq, seed in common_keys]

    x = np.arange(len(common_keys))
    n_impl = len(impl_names)
    width = 0.8 / n_impl

    fig, ax = plt.subplots(figsize=(max(10, len(common_keys) * 0.6), 6))
    colours = plt.cm.tab10(np.linspace(0, 0.8, n_impl))

    for i, impl in enumerate(impl_names):
        offsets = x + (i - (n_impl - 1) / 2) * width
        heights = [mse_by_key[k][impl] for k in common_keys]
        ax.bar(offsets, heights, width, label=impl, color=colours[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Hamiltonian (n_qubits, seed)")
    ax.set_ylabel("mean squared error")
    ax.set_title("Cross-implementation error comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(output_dir, "cross_impl_error_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    """Run structural correlation analysis for all available implementations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_ham_dfs = {}

    for impl in STRUCTURAL_IMPLS:
        if not os.path.exists(RECORDS_CSV[impl]):
            print(f"[SKIP] {impl}: records CSV not found at {RECORDS_CSV[impl]}")
            continue

        print(f"\n── {impl} structural analysis ──")

        df = load_analysis_data(impl)
        if df is None:
            continue

        per_ham = compute_per_ham_stats(df)
        per_ham_dfs[impl] = per_ham

        summary_path = os.path.join(OUTPUT_DIR, f"{impl}_per_ham_summary.csv")
        per_ham.to_csv(summary_path, index=False)
        print(f"[{impl}] Saved {summary_path}")

        for feature in STRUCTURAL_FEATURES:
            plot_structure_vs_error(per_ham, impl, feature, OUTPUT_DIR)

    if len(per_ham_dfs) >= 2:
        plot_cross_impl_error(per_ham_dfs, OUTPUT_DIR)

    print(f"\n── Objective 2 complete. Outputs written to {OUTPUT_DIR} ──")


if __name__ == "__main__":
    main()
