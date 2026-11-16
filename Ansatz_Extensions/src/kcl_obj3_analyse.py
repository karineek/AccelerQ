
"""
Objective 3 analysis: computational resource profiling.

Produces memory, time, and iteration plots plus a text summary from Stage 1–3
profiling CSVs mined by the pipeline. Runnable independently after stages 1–3
complete.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

IMPLEMENTATIONS = ["HEA", "UCCSD", "NN_HEA"]

IMPL_DATA_DIRS = {
    "HEA": "../data/HEA",
    "UCCSD": "../data/UCCSD",
    "NN_HEA": "../data/NN_HEA",
}

IMPL_COLOURS = {
    "HEA": "#4C72B0",
    "UCCSD": "#DD8452",
    "NN_HEA": "#55A868",
}

OUTPUT_DIR = "../data/O3/"
MB = 1 / (1024 ** 2)
GA_MAX_ITER = 49

STAGE2_PROFILING = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"stage2_profiling_{impl}.csv")
    for impl in IMPLEMENTATIONS
}
STAGE2B_PROFILING = os.path.join(IMPL_DATA_DIRS["NN_HEA"], "stage2b_profiling_NN_HEA.csv")

STAGE3_ENERGIES = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"energies_stage3_{impl}.csv")
    for impl in IMPLEMENTATIONS
}

STAGE3_MEMORY_SERIES = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"stage3_memory_series_{impl}.csv")
    for impl in IMPLEMENTATIONS
}

STAGE1_RECORDS = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"records_{impl}.csv")
    for impl in IMPLEMENTATIONS
}

STAGE1_HAMILTONIAN = {
    impl: os.path.join(IMPL_DATA_DIRS[impl], f"hamiltonian_{impl}.csv")
    for impl in IMPLEMENTATIONS
}

# Written by kcl_obj1_analyse.py; baseline arm is shared across HEA / NN_HEA.
BASELINE_VS_OPT_CSV = "../data/O1/obj1_baseline_vs_opt.csv"

STAGE1_MEMORY_COLS = [
    "peak_traced_bytes", "rss_before_bytes", "rss_after_bytes", "rss_peak_bytes",
]

PROFILING_COLS = [
    "total_time_s",
    "checkpoint_times",
    "convergence_iter",
    "converged_early",
    "peak_memory_bytes",
    "final_corpus_size",
]


def _file_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _coerce_bool(series):
    def _one(val):
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("true", "1", "yes")

    return series.apply(_one)


def _parse_checkpoint_times(val):
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []
    points = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((int(item[0]), float(item[1])))
    return points


def _normalize_n_qubits_label(val):
    if pd.isna(val) or str(val).strip() == "":
        return "ALL"
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return str(val)


def _has_profiling_cols(df):
    return all(col in df.columns for col in PROFILING_COLS)


def _save_fig(fig, output_dir, filename):
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _qubit_colormap_values(n_qubits_series):
    numeric = pd.to_numeric(n_qubits_series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return np.zeros(len(n_qubits_series)), 0, 1
    q_min, q_max = valid.min(), valid.max()
    if q_min == q_max:
        return np.zeros(len(n_qubits_series)), q_min, q_max
    colours = (numeric - q_min) / (q_max - q_min)
    colours = colours.fillna(0).values
    return colours, q_min, q_max


def load_stage2(impl):
    path = STAGE2_PROFILING[impl]
    if not _file_nonempty(path):
        print(f"[WARN] {impl}: Stage 2 profiling CSV not found at {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] {impl}: Stage 2 profiling CSV is empty")
        return None

    df["train_time_s"] = pd.to_numeric(df.get("train_time_s"), errors="coerce")
    if "max_size" in df.columns:
        df["max_size"] = pd.to_numeric(df["max_size"], errors="coerce")
    df["n_qubits_label"] = df.get("n_qubits", pd.Series(dtype=object)).apply(
        _normalize_n_qubits_label
    )
    print(f"[{impl}] Loaded {len(df)} Stage 2 profiling row(s) from {path}")
    return df


def load_stage2b():
    if not _file_nonempty(STAGE2B_PROFILING):
        print(f"[WARN] NN_HEA: Stage 2b profiling CSV not found at {STAGE2B_PROFILING}")
        return None

    df = pd.read_csv(STAGE2B_PROFILING)
    if df.empty:
        print("[WARN] NN_HEA: Stage 2b profiling CSV is empty")
        return None

    df["train_time_s"] = pd.to_numeric(df.get("train_time_s"), errors="coerce")
    df["ham_prefix"] = df.get("ham_prefix", "ALL").fillna("ALL")
    df["n_qubits_label"] = "ALL"
    print(f"[NN_HEA] Loaded {len(df)} Stage 2b profiling row(s)")
    return df


def load_stage3(impl):
    path = STAGE3_ENERGIES[impl]
    if not _file_nonempty(path):
        print(f"[WARN] {impl}: Stage 3 energies CSV not found at {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] {impl}: Stage 3 energies CSV is empty")
        return None

    if not _has_profiling_cols(df):
        print(
            f"[WARN] {impl}: energies CSV missing profiling columns "
            f"(expected {PROFILING_COLS})"
        )
        return None

    df = df[df.get("status", "") == "success"].copy()
    if df.empty:
        print(f"[WARN] {impl}: no successful Stage 3 runs with profiling data")
        return None

    df["total_time_s"] = pd.to_numeric(df["total_time_s"], errors="coerce")
    df["peak_memory_bytes"] = pd.to_numeric(df["peak_memory_bytes"], errors="coerce")
    df["final_corpus_size"] = pd.to_numeric(df["final_corpus_size"], errors="coerce")
    df["convergence_iter"] = pd.to_numeric(df["convergence_iter"], errors="coerce")
    df["n_qubits"] = pd.to_numeric(df["n_qubits"], errors="coerce")
    df["converged_early"] = _coerce_bool(df["converged_early"])

    df["checkpoint_times_parsed"] = df["checkpoint_times"].apply(_parse_checkpoint_times)
    df["exit_iter"] = df.apply(
        lambda row: int(row["convergence_iter"])
        if row["converged_early"] and pd.notna(row["convergence_iter"])
        else GA_MAX_ITER,
        axis=1,
    )
    df["peak_memory_mb"] = df["peak_memory_bytes"] * MB

    df = df.dropna(subset=["total_time_s", "peak_memory_bytes"])
    if df.empty:
        print(f"[WARN] {impl}: no rows with valid profiling metrics")
        return None

    print(f"[{impl}] Loaded {len(df)} profiled Stage 3 run(s)")
    return df


def load_memory_series(impl):
    path = STAGE3_MEMORY_SERIES[impl]
    if not _file_nonempty(path):
        print(f"[WARN] {impl}: memory series CSV not found at {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] {impl}: memory series CSV is empty")
        return None

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["current_bytes"] = pd.to_numeric(df["current_bytes"], errors="coerce")
    df["peak_bytes"] = pd.to_numeric(df["peak_bytes"], errors="coerce")
    df["n_qubits"] = pd.to_numeric(df["n_qubits"], errors="coerce")
    df["current_mb"] = df["current_bytes"] * MB
    df["peak_mb"] = df["peak_bytes"] * MB

    print(f"[{impl}] Loaded {len(df)} memory series row(s)")
    return df


def _load_hamiltonian_meta(ham_path):
    if not _file_nonempty(ham_path):
        return {}
    ham_df = pd.read_csv(ham_path)
    meta = {}
    for _, row in ham_df.iterrows():
        ham_id = row.get("ham_id")
        if pd.isna(ham_id):
            continue
        meta[int(ham_id)] = row.to_dict()
    return meta


def load_stage1(impl):
    records_path = STAGE1_RECORDS[impl]
    ham_path = STAGE1_HAMILTONIAN[impl]
    if not _file_nonempty(records_path):
        print(f"[WARN] {impl}: Stage 1 records CSV not found at {records_path}")
        return None

    df = pd.read_csv(records_path)
    if df.empty:
        print(f"[WARN] {impl}: Stage 1 records CSV is empty")
        return None

    missing_mem = [c for c in STAGE1_MEMORY_COLS if c not in df.columns]
    if missing_mem:
        print(
            f"[WARN] {impl}: records CSV missing Stage 1 memory columns {missing_mem} "
            "(re-run Phase 1 mining to populate)"
        )

    ham_meta = _load_hamiltonian_meta(ham_path)
    n_qubits_vals = []
    for ham_id, meta in ham_meta.items():
        n_qubits = meta.get("n_qubits")
        if pd.notna(n_qubits):
            n_qubits_vals.append(int(float(n_qubits)))

    df["ham_id"] = pd.to_numeric(df["ham_id"], errors="coerce")
    df["n_qubits"] = df["ham_id"].map(
        lambda hid: ham_meta.get(int(hid), {}).get("n_qubits") if pd.notna(hid) else np.nan
    )
    df["n_qubits"] = pd.to_numeric(df["n_qubits"], errors="coerce")
    df["dataset"] = df["ham_id"].map(
        lambda hid: ham_meta.get(int(hid), {}).get("dataset") if pd.notna(hid) else ""
    )

    df["mining_time"] = pd.to_numeric(df.get("mining_time"), errors="coerce")
    for col in STAGE1_MEMORY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["peak_traced_mb"] = df["peak_traced_bytes"] * MB
    df["rss_peak_mb"] = df["rss_peak_bytes"] * MB
    df["rss_delta_mb"] = (df["rss_after_bytes"] - df["rss_before_bytes"]) * MB

    n_failed = int((df["status"] != "success").sum())
    success_df = df[df["status"] == "success"].copy()
    if success_df.empty:
        print(f"[WARN] {impl}: no successful Stage 1 mining records")
        return None

    print(
        f"[{impl}] Loaded {len(success_df)} successful Stage 1 record(s) "
        f"({n_failed} non-success); n_qubits in hamiltonian meta: "
        f"{sorted(set(n_qubits_vals)) if n_qubits_vals else 'unknown'}"
    )
    success_df.attrs["n_failed_records"] = n_failed
    success_df.attrs["n_total_records"] = len(df)
    return success_df


def plot_stage1_peak_traced_comparison(stage1_dfs, output_dir):
    available = {impl: df for impl, df in stage1_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage1_peak_traced_comparison: no Stage 1 data")
        return

    qubit_counts = sorted({
        int(n_qubits) for df in available.values()
        for n_qubits in df["n_qubits"].dropna().unique()
    })
    if not qubit_counts:
        print("[SKIP] stage1_peak_traced_comparison: no n_qubits values")
        return

    impl_names = [impl for impl in IMPLEMENTATIONS if impl in available]
    x = np.arange(len(qubit_counts))
    width = 0.8 / max(len(impl_names), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(qubit_counts) * 1.6), 5.2))
    for i, impl in enumerate(impl_names):
        df = available[impl]
        means, stds = [], []
        for n_qubits in qubit_counts:
            sub = df[df["n_qubits"] == n_qubits]["peak_traced_mb"].dropna()
            means.append(sub.mean() if not sub.empty else np.nan)
            stds.append(sub.std() if len(sub) > 1 else 0)
        offsets = x + (i - (len(impl_names) - 1) / 2) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=4,
               label=impl, color=IMPL_COLOURS[impl], alpha=0.9,
               edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=1.1, capthick=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels([str(n_qubits) for n_qubits in qubit_counts], fontsize=12)
    ax.set_xlabel("n_qubits", fontsize=14)
    ax.set_ylabel("peak traced memory (MB)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Stage 1 peak tracemalloc memory by qubit count", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save_fig(fig, output_dir, "stage1_peak_traced_comparison.png")


def plot_stage1_peak_rss_comparison(stage1_dfs, output_dir):
    available = {impl: df for impl, df in stage1_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage1_peak_rss_comparison: no Stage 1 data")
        return

    qubit_counts = sorted({
        int(n_qubits) for df in available.values()
        for n_qubits in df["n_qubits"].dropna().unique()
    })
    if not qubit_counts:
        print("[SKIP] stage1_peak_rss_comparison: no n_qubits values")
        return

    impl_names = [impl for impl in IMPLEMENTATIONS if impl in available]
    x = np.arange(len(qubit_counts))
    width = 0.8 / max(len(impl_names), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(qubit_counts) * 1.6), 5.2))
    for i, impl in enumerate(impl_names):
        df = available[impl]
        means, stds = [], []
        for n_qubits in qubit_counts:
            sub = df[df["n_qubits"] == n_qubits]["rss_peak_mb"].dropna()
            means.append(sub.mean() if not sub.empty else np.nan)
            stds.append(sub.std() if len(sub) > 1 else 0)
        offsets = x + (i - (len(impl_names) - 1) / 2) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=4,
               label=impl, color=IMPL_COLOURS[impl], alpha=0.9,
               edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=1.1, capthick=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels([str(n_qubits) for n_qubits in qubit_counts], fontsize=12)
    ax.set_xlabel("n_qubits", fontsize=14)
    ax.set_ylabel("peak RSS (MB)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Stage 1 peak process RSS by qubit count", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save_fig(fig, output_dir, "stage1_peak_rss_comparison.png")


def plot_stage1_traced_vs_rss(stage1_dfs, output_dir):
    available = {impl: df for impl, df in stage1_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage1_traced_vs_rss_scatter: no Stage 1 data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for impl in IMPLEMENTATIONS:
        df = available.get(impl)
        if df is None:
            continue
        sub = df.dropna(subset=["peak_traced_mb", "rss_peak_mb"])
        if sub.empty:
            continue
        ax.scatter(
            sub["peak_traced_mb"], sub["rss_peak_mb"],
            alpha=0.6, label=impl, color=IMPL_COLOURS[impl], s=24,
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] stage1_traced_vs_rss_scatter: no plottable data")
        return

    ax.set_xlabel("peak traced memory (MB)")
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Stage 1 tracemalloc vs process RSS")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, "stage1_traced_vs_rss_scatter.png")


def plot_stage1_memory_vs_time(stage1_dfs, output_dir):
    available = {impl: df for impl, df in stage1_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage1_memory_vs_mining_time: no Stage 1 data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for impl in IMPLEMENTATIONS:
        df = available.get(impl)
        if df is None:
            continue
        sub = df.dropna(subset=["mining_time", "rss_peak_mb"])
        if sub.empty:
            continue
        ax.scatter(
            sub["mining_time"], sub["rss_peak_mb"],
            alpha=0.6, label=impl, color=IMPL_COLOURS[impl], s=24,
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] stage1_memory_vs_mining_time: no plottable data")
        return

    ax.set_xlabel("mining_time (s)")
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Stage 1 mining time vs peak RSS")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, "stage1_memory_vs_mining_time.png")


def plot_stage1_memory_aggregate_per_impl(stage1_dfs, output_dir):
    available = {impl: df for impl, df in stage1_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage1_memory_aggregate_per_impl: no Stage 1 data")
        return

    labels, data, colours = [], [], []
    for impl in IMPLEMENTATIONS:
        df = available.get(impl)
        if df is None:
            continue
        sub = df["rss_peak_mb"].dropna()
        if sub.empty:
            continue
        labels.append(impl)
        data.append(sub.values)
        colours.append(IMPL_COLOURS[impl])

    if not labels:
        print("[SKIP] stage1_memory_aggregate_per_impl: no RSS data")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, colour in zip(boxplot["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)

    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Stage 1 peak RSS distribution by implementation")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir, "stage1_memory_aggregate_per_impl.png")


def plot_memory_series_per_impl(series_dfs, output_dir):
    available = {impl: df for impl, df in series_dfs.items() if df is not None}
    if not available:
        print("[SKIP] memory_series_per_impl: no memory series data")
        return

    fig, axes = plt.subplots(len(IMPLEMENTATIONS), 1, figsize=(10, 4 * len(IMPLEMENTATIONS)),
                             sharex=True, squeeze=False)

    any_plotted = False
    for idx, impl in enumerate(IMPLEMENTATIONS):
        ax = axes[idx, 0]
        df = available.get(impl)
        if df is None:
            ax.set_title(f"{impl} (no data)")
            ax.axis("off")
            continue

        hams = df["ham_prefix"].unique()
        show_current = len(hams) <= 8
        colours, q_min, q_max = _qubit_colormap_values(df["n_qubits"])

        for ham in hams:
            sub = df[df["ham_prefix"] == ham].sort_values("iteration")
            if sub.empty:
                continue
            sub_c, _, _ = _qubit_colormap_values(sub["n_qubits"])
            n_qubits = sub["n_qubits"].iloc[0]
            color = plt.cm.viridis(sub_c[0] if len(sub_c) else 0)
            ax.plot(sub["iteration"], sub["peak_mb"], color=color, linewidth=1.5,
                    label=f"{ham} ({int(n_q)}q)" if pd.notna(n_qubits) else ham)
            if show_current:
                ax.plot(sub["iteration"], sub["current_mb"], color=color,
                        linewidth=0.8, alpha=0.35, linestyle="--")

        ax.set_title(impl)
        ax.set_ylabel("peak memory (MB)")
        ax.grid(alpha=0.3)
        if len(hams) <= 12:
            ax.legend(fontsize=7, loc="upper left")
        any_plotted = True

        if q_min != q_max:
            scalar_mappable = plt.cm.ScalarMappable(cmap="viridis",
                                       norm=plt.Normalize(vmin=q_min, vmax=q_max))
            scalar_mappable.set_array([])
            plt.colorbar(scalar_mappable, ax=ax, label="n_qubits")

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] memory_series_per_impl: no plottable data")
        return

    axes[-1, 0].set_xlabel("GA iteration")
    fig.suptitle("Memory series per Hamiltonian", y=1.01)
    _save_fig(fig, output_dir, "memory_series_per_impl.png")


def plot_memory_series_cross_impl(series_dfs, output_dir):
    available = {impl: df for impl, df in series_dfs.items() if df is not None}
    if len(available) < 2:
        print("[SKIP] memory_series_cross_impl: need at least 2 implementations")
        return

    ham_sets = [set(df["ham_prefix"].unique()) for df in available.values()]
    common_hams = sorted(set.intersection(*ham_sets))
    if not common_hams:
        print("[SKIP] memory_series_cross_impl: no shared ham_prefix across implementations")
        return

    n_cols = min(3, len(common_hams))
    n_rows = int(np.ceil(len(common_hams) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i, ham in enumerate(common_hams):
        ax = axes[i // n_cols, i % n_cols]
        for impl, df in available.items():
            sub = df[df["ham_prefix"] == ham].sort_values("iteration")
            if sub.empty:
                continue
            ax.plot(sub["iteration"], sub["peak_mb"], color=IMPL_COLOURS[impl],
                    linewidth=1.5, label=impl)
        ax.set_title(ham)
        ax.set_xlabel("GA iteration")
        ax.set_ylabel("peak memory (MB)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(common_hams), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle("Memory series — shared Hamiltonians across implementations", y=1.01)
    _save_fig(fig, output_dir, "memory_series_cross_impl.png")


def plot_memory_aggregate(series_dfs, output_dir):
    available = {impl: df for impl, df in series_dfs.items() if df is not None}
    if not available:
        print("[SKIP] memory_aggregate_per_impl: no memory series data")
        return

    fig, axes = plt.subplots(1, len(IMPLEMENTATIONS), figsize=(5 * len(IMPLEMENTATIONS), 4),
                             squeeze=False)
    any_plotted = False

    for idx, impl in enumerate(IMPLEMENTATIONS):
        ax = axes[0, idx]
        df = available.get(impl)
        if df is None:
            ax.set_title(f"{impl} (no data)")
            ax.axis("off")
            continue

        agg = df.groupby("iteration")["peak_mb"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)
        ax.plot(agg["iteration"], agg["mean"], color=IMPL_COLOURS[impl], linewidth=2)
        ax.fill_between(agg["iteration"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=IMPL_COLOURS[impl], alpha=0.25)
        ax.set_title(impl)
        ax.set_xlabel("GA iteration")
        ax.set_ylabel("peak memory (MB)")
        ax.grid(alpha=0.3)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] memory_aggregate_per_impl: no plottable data")
        return

    fig.suptitle("Aggregate peak memory per implementation", y=1.02)
    _save_fig(fig, output_dir, "memory_aggregate_per_impl.png")


def plot_peak_memory_comparison(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] peak_memory_comparison: no Stage 3 profiling data")
        return

    qubit_counts = sorted({
        int(n_qubits) for df in available.values()
        for n_qubits in df["n_qubits"].dropna().unique()
    })
    if not qubit_counts:
        print("[SKIP] peak_memory_comparison: no n_qubits values")
        return

    impl_names = [impl for impl in IMPLEMENTATIONS if impl in available]
    x = np.arange(len(qubit_counts))
    width = 0.8 / max(len(impl_names), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(qubit_counts) * 1.5), 5))

    for i, impl in enumerate(impl_names):
        df = available[impl]
        means, stds = [], []
        for n_qubits in qubit_counts:
            sub = df[df["n_qubits"] == n_qubits]["peak_memory_mb"]
            means.append(sub.mean() if not sub.empty else np.nan)
            stds.append(sub.std() if len(sub) > 1 else 0)
        offsets = x + (i - (len(impl_names) - 1) / 2) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=3,
               label=impl, color=IMPL_COLOURS[impl], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n_qubits) for n_qubits in qubit_counts])
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("peak memory (MB)")
    ax.set_title("Peak memory comparison by qubit count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir, "peak_memory_comparison.png")


def plot_stage2_time(stage2_dfs, stage2b_df, output_dir):
    labels, means, stds, colours = [], [], [], []

    for impl in IMPLEMENTATIONS:
        df = stage2_dfs.get(impl)
        if df is None or df["train_time_s"].dropna().empty:
            continue
        labels.append(impl)
        times = df["train_time_s"].dropna()
        means.append(times.mean())
        stds.append(times.std() if len(times) > 1 else 0)
        colours.append(IMPL_COLOURS[impl])

    if stage2b_df is not None and not stage2b_df["train_time_s"].dropna().empty:
        times = stage2b_df["train_time_s"].dropna()
        labels.append("NN_HEA (NN)")
        means.append(times.mean())
        stds.append(times.std() if len(times) > 1 else 0)
        colours.append(IMPL_COLOURS["NN_HEA"])

    if not labels:
        print("[SKIP] stage2_training_time: no Stage 2 profiling data")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours, alpha=0.85)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("train_time_s")
    ax.set_title("Stage 2 training time by implementation")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, output_dir, "stage2_training_time.png")


def plot_stage3_total_time(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage3_total_time_scaling: no Stage 3 profiling data")
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    markers = {"HEA": "o", "UCCSD": "s", "NN_HEA": "^"}
    n_impl = len(IMPLEMENTATIONS)
    any_plotted = False

    for i, impl in enumerate(IMPLEMENTATIONS):
        df = available.get(impl)
        if df is None:
            continue
        agg = df.groupby("n_qubits")["total_time_s"].agg(["mean", "std", "count"])
        agg = agg.reset_index().sort_values("n_qubits")
        if agg.empty:
            continue
        dodge = (i - (n_impl - 1) / 2) * 0.22        # cosmetic x offset
        ax.errorbar(agg["n_qubits"] + dodge, agg["mean"],
                    yerr=agg["std"].fillna(0),
                    color=IMPL_COLOURS[impl], marker=markers.get(impl, "o"),
                    markersize=7, markeredgecolor="k", markeredgewidth=0.5,
                    linewidth=1.6, capsize=5, capthick=1.2, elinewidth=1.2,
                    label=impl, zorder=3)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] stage3_total_time_scaling: no plottable data")
        return

    ax.set_xlabel("n_qubits", fontsize=14)
    ax.set_ylabel("total_time_s", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Stage 3 total time scaling", fontsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=12, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save_fig(fig, output_dir, "stage3_total_time_scaling.png")


def plot_final_vqe_time_per_iter_scaling(output_dir,
                                         csv_path="../data/O1/obj1_baseline_vs_opt.csv"):
    """Baseline per-iteration VQE cost vs qubit count (shared baseline, log y)."""
    if not _file_nonempty(csv_path):
        print("[SKIP] final_vqe_time_per_iter_scaling: no obj1_baseline_vs_opt.csv")
        return

    df = pd.read_csv(csv_path).drop_duplicates(subset="ham_prefix").copy()
    df["time_per_iter_baseline"] = (
        df["run_time_s_baseline"] / df["n_iterations_baseline"])
    df = df.dropna(subset=["time_per_iter_baseline", "n_qubits"])
    df = df.sort_values(["n_qubits", "ham_prefix"])
    if df.empty:
        print("[SKIP] final_vqe_time_per_iter_scaling: no usable rows")
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.set_yscale("log")
    ax.plot(df["n_qubits"], df["time_per_iter_baseline"],
            color="0.5", linewidth=1.2, zorder=2)
    ax.scatter(df["n_qubits"], df["time_per_iter_baseline"], s=90,
               facecolor="0.30", edgecolor="black", linewidths=0.7, zorder=3)

    ax.margins(x=0.20)
    lo = df["time_per_iter_baseline"].min()
    hi = df["time_per_iter_baseline"].max()
    ax.set_ylim(lo / 3.0, hi * 4.0)

    # Deterministic labelling: within a qubit group the lower point is
    # labelled below and the upper above; the side is chosen inward so no
    # label can run off the axes.
    x_lo, x_hi = df["n_qubits"].min(), df["n_qubits"].max()
    for xq, grp in df.groupby("n_qubits"):
        grp = grp.sort_values("time_per_iter_baseline")
        n = len(grp)
        frac = (xq - x_lo) / max(x_hi - x_lo, 1e-9)
        dx, ha = (12, "left") if frac < 0.5 else (-12, "right")
        for k, (_, row) in enumerate(grp.iterrows()):
            dy, va = (11, "bottom") if (n == 1 or k > 0) else (-11, "top")
            ax.annotate(row["ham_prefix"],
                        xy=(row["n_qubits"], row["time_per_iter_baseline"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=10, color="0.25", zorder=6,
                        arrowprops=dict(arrowstyle="-", color="0.72",
                                        linewidth=0.6, shrinkA=0, shrinkB=4))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("n_qubits", fontsize=14)
    ax.set_ylabel("baseline time per iteration (s, log scale)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Final VQE baseline per-iteration cost by qubit count",
                 fontsize=15)
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save_fig(fig, output_dir, "final_vqe_time_per_iter_scaling.png")


def plot_final_vqe_runtime_comparison(output_dir):
    """Side-by-side baseline vs optimised Final-VQE wall-clock time (HEA | NN_HEA)."""
    if not _file_nonempty(BASELINE_VS_OPT_CSV):
        print("[SKIP] final_vqe_runtime_comparison: missing obj1_baseline_vs_opt.csv")
        return

    df = pd.read_csv(BASELINE_VS_OPT_CSV)
    if df.empty:
        print("[SKIP] final_vqe_runtime_comparison: empty obj1_baseline_vs_opt.csv")
        return

    for col in ("run_time_s_baseline", "run_time_s_optimised", "n_qubits"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["implementation"].isin(["HEA", "NN_HEA"])]
    df = df.dropna(subset=[
        "run_time_s_baseline", "run_time_s_optimised", "n_qubits", "ham_prefix",
    ])
    df = df[df["run_time_s_baseline"] > 0]
    if df.empty:
        print("[SKIP] final_vqe_runtime_comparison: no plottable rows")
        return

    df = df.sort_values(["n_qubits", "ham_prefix"])
    opt_colours = {"HEA": "tab:blue", "NN_HEA": "tab:green"}
    impls = [impl for impl in ("HEA", "NN_HEA") if (df["implementation"] == impl).any()]
    if not impls:
        print("[SKIP] final_vqe_runtime_comparison: no HEA/NN_HEA rows")
        return

    fig, axes = plt.subplots(1, len(impls), figsize=(6 * len(impls), 5), sharey=True, squeeze=False)
    bar_width = 0.35
    legend_handles = None

    for col_idx, impl in enumerate(impls):
        ax = axes[0, col_idx]
        sub = df[df["implementation"] == impl].reset_index(drop=True)
        x = np.arange(len(sub))
        baseline = sub["run_time_s_baseline"].to_numpy()
        optimised = sub["run_time_s_optimised"].to_numpy()

        bars_bl = ax.bar(
            x - bar_width / 2, baseline, bar_width,
            color="lightgrey", edgecolor="0.4", linewidth=0.5, label="baseline",
        )
        bars_opt = ax.bar(
            x + bar_width / 2, optimised, bar_width,
            color=opt_colours[impl], edgecolor="0.3", linewidth=0.5, label="optimised",
        )

        for i, (b_opt, t_bl, t_opt) in enumerate(zip(bars_opt, baseline, optimised)):
            ratio = t_opt / t_bl
            ax.annotate(
                f"{ratio:.0f}x",
                (b_opt.get_x() + b_opt.get_width() / 2, t_opt),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=7, color="0.25",
            )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["ham_prefix"].tolist(), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("run time (s, log scale)")
        ax.set_title(impl)
        ax.grid(axis="y", alpha=0.3)
        if legend_handles is None:
            legend_handles = [bars_bl, bars_opt]

    fig.suptitle("Final VQE wall-clock time: baseline vs optimised", y=1.02)
    if legend_handles is not None:
        fig.legend(
            legend_handles, ["baseline", "optimised"],
            loc="upper center", ncol=2, fontsize=9,
            bbox_to_anchor=(0.5, 0.02), frameon=True,
        )
    fig.tight_layout()
    _save_fig(fig, output_dir, "final_vqe_runtime_comparison.png")


def plot_checkpoint_times(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] stage3_checkpoint_time_profiles: no Stage 3 profiling data")
        return

    fig, axes = plt.subplots(1, len(IMPLEMENTATIONS), figsize=(5 * len(IMPLEMENTATIONS), 4),
                             squeeze=False)
    any_plotted = False

    for idx, impl in enumerate(IMPLEMENTATIONS):
        ax = axes[0, idx]
        df = available.get(impl)
        if df is None:
            ax.set_title(f"{impl} (no data)")
            ax.axis("off")
            continue

        checkpoint_rows = []
        for _, row in df.iterrows():
            checkpoints = row["checkpoint_times_parsed"]
            if not checkpoints:
                continue
            colours, q_min, q_max = _qubit_colormap_values(pd.Series([row["n_qubits"]]))
            color = plt.cm.viridis(colours[0])
            iters = [point[0] for point in checkpoints]
            times = [point[1] for point in checkpoints]
            ax.plot(iters, times, color=color, linewidth=1, alpha=0.6)
            for iteration, elapsed_s in checkpoints:
                checkpoint_rows.append({"iteration": iteration, "elapsed_s": elapsed_s})

        if checkpoint_rows:
            cp_df = pd.DataFrame(checkpoint_rows)
            mean_line = cp_df.groupby("iteration")["elapsed_s"].mean()
            ax.plot(mean_line.index, mean_line.values, color="black",
                    linewidth=2.5, linestyle="--", label="mean")
            any_plotted = True

        ax.set_title(impl)
        ax.set_xlabel("checkpoint iteration")
        ax.set_ylabel("elapsed (s)")
        ax.grid(alpha=0.3)
        if checkpoint_rows:
            ax.legend(fontsize=8)

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] stage3_checkpoint_time_profiles: no checkpoint_times data")
        return

    fig.suptitle("Stage 3 checkpoint time profiles", y=1.02)
    _save_fig(fig, output_dir, "stage3_checkpoint_time_profiles.png")


def plot_time_vs_convergence_iter(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] time_vs_convergence_iter: no Stage 3 profiling data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    annotations = []
    legend_added = set()

    for i, impl in enumerate(IMPLEMENTATIONS):
        df = available.get(impl)
        if df is None:
            continue
        dodge = (i - (len(IMPLEMENTATIONS) - 1) / 2) * 0.35   # cosmetic offset
        early = df[df["converged_early"]]
        full = df[~df["converged_early"]]
        if not early.empty:
            ax.scatter(early["exit_iter"] + dodge, early["total_time_s"],
                       color=IMPL_COLOURS[impl], marker="o", s=60,
                       edgecolors="k", linewidths=0.4, alpha=0.85,
                       label=f"{impl} (early exit)")
            legend_added.add(impl)
        if not full.empty:
            ax.scatter(full["exit_iter"] + dodge, full["total_time_s"],
                       facecolors="none", edgecolors=IMPL_COLOURS[impl],
                       marker="o", s=60, linewidths=1.6,
                       label=f"{impl} (full run)")
            legend_added.add(impl)
        n_early = len(early)
        n_total = len(df)
        annotations.append(f"{impl}: {n_early}/{n_total} early exits")

    if not annotations:
        plt.close(fig)
        print("[SKIP] time_vs_convergence_iter: no plottable data")
        return

    ax.set_xlabel("exit_iter", fontsize=14)
    ax.set_ylabel("total_time_s", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Time vs convergence iteration", fontsize=15)
    ax.annotate("\n".join(annotations), xy=(0.98, 0.03), xycoords="axes fraction",
                va="bottom", ha="right", fontsize=11, zorder=6,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.92,
                          edgecolor="0.7"))
    if legend_added:
        ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=4, frameon=False, columnspacing=1.4, handletextpad=0.5)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))   # reserve bottom 8% for the legend row
    _save_fig(fig, output_dir, "time_vs_convergence_iter.png")


def plot_convergence_iter_distribution(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] convergence_iter_distribution: no Stage 3 profiling data")
        return

    bins = list(range(0, GA_MAX_ITER + 6, 5))
    n = len(IMPLEMENTATIONS)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.7 * n), squeeze=False,
                             sharex=True)
    any_plotted = False

    for idx, impl in enumerate(IMPLEMENTATIONS):
        ax = axes[idx, 0]
        df = available.get(impl)
        if df is None:
            ax.set_title(f"{impl} (no data)", fontsize=13)
            ax.axis("off")
            continue

        early = df[df["exit_iter"] < GA_MAX_ITER]["exit_iter"]
        full = df[df["exit_iter"] == GA_MAX_ITER]["exit_iter"]

        if not early.empty:
            ax.hist(early, bins=bins, color=IMPL_COLOURS[impl], alpha=0.9,
                    edgecolor="white", label="early exit")
        if not full.empty:
            ax.hist(full, bins=[GA_MAX_ITER - 2.5, GA_MAX_ITER + 2.5],
                    color="grey", alpha=0.9, edgecolor="white",
                    label=f"full run ({GA_MAX_ITER})")

        ax.set_title(impl, fontsize=13, loc="left")
        ax.set_ylabel("count", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # counts are integers
        ax.set_xlim(0, GA_MAX_ITER + 5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, frameon=False, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("[SKIP] convergence_iter_distribution: no plottable data")
        return

    axes[-1, 0].set_xlabel("exit_iter", fontsize=13)
    fig.suptitle("Convergence iteration distribution", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, output_dir, "convergence_iter_distribution.png")


def plot_final_corpus_size(stage3_dfs, output_dir):
    available = {impl: df for impl, df in stage3_dfs.items() if df is not None}
    if not available:
        print("[SKIP] final_corpus_size_by_qubits: no Stage 3 profiling data")
        return

    qubit_counts = sorted({
        int(n_qubits) for df in available.values()
        for n_qubits in df["n_qubits"].dropna().unique()
    })
    if not qubit_counts:
        print("[SKIP] final_corpus_size_by_qubits: no n_qubits values")
        return

    impl_names = [impl for impl in IMPLEMENTATIONS if impl in available]
    x = np.arange(len(qubit_counts))
    width = 0.8 / max(len(impl_names), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(qubit_counts) * 1.5), 5))

    for i, impl in enumerate(impl_names):
        df = available[impl]
        means, stds = [], []
        for n_qubits in qubit_counts:
            sub = df[df["n_qubits"] == n_qubits]["final_corpus_size"]
            means.append(sub.mean() if not sub.empty else np.nan)
            stds.append(sub.std() if len(sub) > 1 else 0)
        offsets = x + (i - (len(impl_names) - 1) / 2) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=4,
               label=impl, color=IMPL_COLOURS[impl], alpha=0.9,
               edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=1.1, capthick=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels([str(n_qubits) for n_qubits in qubit_counts], fontsize=12)
    ax.set_xlabel("n_qubits", fontsize=14)
    ax.set_ylabel("mean final corpus size", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Final corpus size by qubit count", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save_fig(fig, output_dir, "final_corpus_size_by_qubits.png")


def _fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return f"{float(val):.{decimals}f}"


def write_summary(stage1_dfs, stage2_dfs, stage2b_df, stage3_dfs, series_dfs, output_dir):
    lines = [
        "COMPUTATIONAL RESOURCE SUMMARY",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "=== STAGE 1: Mining Memory ===",
    ]
    stage1_stats = {}

    for impl in IMPLEMENTATIONS:
        df = stage1_dfs.get(impl)
        if df is None:
            lines.append(f"{impl}: N/A")
            continue

        n_failed = int(df.attrs.get("n_failed_records", 0))
        n_total = int(df.attrs.get("n_total_records", len(df)))
        qubits = sorted({int(n_qubits) for n_qubits in df["n_qubits"].dropna().unique()})
        traced = df["peak_traced_mb"].dropna()
        rss = df["rss_peak_mb"].dropna()
        times = df["mining_time"].dropna()

        stage1_stats[impl] = {
            "mean_rss_mb": rss.mean() if not rss.empty else np.nan,
            "mean_traced_mb": traced.mean() if not traced.empty else np.nan,
        }
        lines.extend([
            f"{impl}:",
            f"  n_success_records: {len(df)} ({n_failed} non-success of {n_total} total)",
            f"  n_qubits: {qubits if qubits else 'unknown'}",
            f"  mining_time_s: mean={_fmt(times.mean()) if not times.empty else 'N/A'}",
            f"  peak_traced_mb: mean={_fmt(traced.mean()) if not traced.empty else 'N/A'}, "
            f"max={_fmt(traced.max()) if not traced.empty else 'N/A'}",
            f"  rss_peak_mb: mean={_fmt(rss.mean()) if not rss.empty else 'N/A'}, "
            f"max={_fmt(rss.max()) if not rss.empty else 'N/A'}",
        ])

    lines.extend(["", "=== STAGE 1: Cross-implementation ==="])
    if stage1_stats:
        valid_rss = {k: v["mean_rss_mb"] for k, v in stage1_stats.items() if pd.notna(v["mean_rss_mb"])}
        valid_traced = {k: v["mean_traced_mb"] for k, v in stage1_stats.items() if pd.notna(v["mean_traced_mb"])}
        if valid_rss:
            lines.append(f"Lowest mean RSS peak (Stage 1): {min(valid_rss, key=valid_rss.get)}")
        if valid_traced:
            lines.append(f"Lowest mean tracemalloc peak (Stage 1): {min(valid_traced, key=valid_traced.get)}")
    else:
        lines.append("N/A (no Stage 1 profiling data)")

    lines.extend([
        "",
        "Stage 1 note: RSS is whole-process memory (includes Qiskit classical simulation). "
        "rss_peak_bytes = max(before, after) may miss transient mid-VQE peaks. "
        "tracemalloc captures Python heap only.",
        "",
        "=== STAGE 2: Training Time ===",
    ])

    for impl in IMPLEMENTATIONS:
        df = stage2_dfs.get(impl)
        if df is None or df["train_time_s"].dropna().empty:
            lines.append(f"{impl}: N/A")
        else:
            times = df["train_time_s"].dropna()
            lines.append(
                f"{impl}: mean={_fmt(times.mean())}s, n_runs={len(times)}"
            )

    if stage2b_df is None or stage2b_df["train_time_s"].dropna().empty:
        lines.append("NN_HEA (NN stage2b): N/A")
    else:
        times = stage2b_df["train_time_s"].dropna()
        lines.append(
            f"NN_HEA (NN stage2b): mean={_fmt(times.mean())}s, n_runs={len(times)}"
        )

    lines.extend(["", "=== STAGE 3: Total Time ==="])
    stage3_stats = {}

    for impl in IMPLEMENTATIONS:
        df = stage3_dfs.get(impl)
        if df is None:
            lines.append(f"{impl}: N/A")
            continue

        n_early = int(df["converged_early"].sum())
        n_total = len(df)
        rate = 100.0 * n_early / n_total if n_total else 0
        stage3_stats[impl] = {
            "mean_time": df["total_time_s"].mean(),
            "mean_peak_mb": df["peak_memory_mb"].mean(),
            "n_early": n_early,
        }
        lines.extend([
            f"{impl}:",
            f"  n_hamiltonians: {n_total}",
            f"  total_time_s: mean={_fmt(df['total_time_s'].mean())}, "
            f"min={_fmt(df['total_time_s'].min())}, max={_fmt(df['total_time_s'].max())}",
            f"  exit_iter: mean={_fmt(df['exit_iter'].mean())}, "
            f"min={_fmt(df['exit_iter'].min())}, max={_fmt(df['exit_iter'].max())}",
            f"  early exits: {n_early}/{n_total} ({_fmt(rate)}%)",
            f"  mean peak_memory_mb: {_fmt(df['peak_memory_mb'].mean())}",
        ])

    lines.extend(["", "=== STAGE 3: Memory ==="])
    for impl in IMPLEMENTATIONS:
        df = stage3_dfs.get(impl)
        if df is None:
            lines.append(f"{impl}: N/A")
            continue
        worst_idx = df["peak_memory_mb"].idxmax()
        worst = df.loc[worst_idx]
        lines.extend([
            f"{impl}:",
            f"  mean peak_memory_mb: {_fmt(df['peak_memory_mb'].mean())}",
            f"  max peak_memory_mb: {_fmt(df['peak_memory_mb'].max())} "
            f"({worst.get('ham_prefix', '?')}, {int(worst['n_qubits'])} qubits)",
            f"  mean final_corpus_size: {_fmt(df['final_corpus_size'].mean())}",
        ])

    lines.extend([
        "",
        "=== CROSS-IMPLEMENTATION COMPARISON ===",
    ])

    if stage3_stats:
        fastest = min(stage3_stats, key=lambda k: stage3_stats[k]["mean_time"])
        lowest_mem = min(stage3_stats, key=lambda k: stage3_stats[k]["mean_peak_mb"])
        most_early = max(stage3_stats, key=lambda k: stage3_stats[k]["n_early"])
        lines.extend([
            f"Fastest Stage 3 (lowest mean total_time_s): {fastest}",
            f"Lowest peak memory (mean): {lowest_mem}",
            f"Most early exits: {most_early}",
        ])
    else:
        lines.append("N/A (no Stage 3 profiling data)")

    lines.extend([
        "",
        "Note: Per-iteration corpus size (plot I3) is not logged in mined files; "
        "only final_corpus_size at run end is available.",
        "",
        "Note: NN-HEA Stage 1/2 mining and XGBoost training costs are not duplicated — "
        "NN-HEA reuses HEA's Stage 1 records and Stage 2 surrogate model. "
        "N/A for NN-HEA Stage 1/2 in this table means shared with HEA, not missing data.",
    ])

    out_path = os.path.join(output_dir, "resource_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stage1_dfs = {impl: load_stage1(impl) for impl in IMPLEMENTATIONS}
    stage2_dfs = {impl: load_stage2(impl) for impl in IMPLEMENTATIONS}
    stage2b_df = load_stage2b()
    stage3_dfs = {impl: load_stage3(impl) for impl in IMPLEMENTATIONS}
    series_dfs = {impl: load_memory_series(impl) for impl in IMPLEMENTATIONS}

    plot_stage1_peak_traced_comparison(stage1_dfs, OUTPUT_DIR)
    plot_stage1_peak_rss_comparison(stage1_dfs, OUTPUT_DIR)
    plot_stage1_traced_vs_rss(stage1_dfs, OUTPUT_DIR)
    plot_stage1_memory_vs_time(stage1_dfs, OUTPUT_DIR)
    plot_stage1_memory_aggregate_per_impl(stage1_dfs, OUTPUT_DIR)

    plot_memory_series_per_impl(series_dfs, OUTPUT_DIR)
    plot_memory_series_cross_impl(series_dfs, OUTPUT_DIR)
    plot_memory_aggregate(series_dfs, OUTPUT_DIR)
    plot_peak_memory_comparison(stage3_dfs, OUTPUT_DIR)

    plot_stage2_time(stage2_dfs, stage2b_df, OUTPUT_DIR)
    plot_stage3_total_time(stage3_dfs, OUTPUT_DIR)
    plot_final_vqe_time_per_iter_scaling(OUTPUT_DIR)
    plot_final_vqe_runtime_comparison(OUTPUT_DIR)
    plot_checkpoint_times(stage3_dfs, OUTPUT_DIR)
    plot_time_vs_convergence_iter(stage3_dfs, OUTPUT_DIR)

    plot_convergence_iter_distribution(stage3_dfs, OUTPUT_DIR)
    plot_final_corpus_size(stage3_dfs, OUTPUT_DIR)

    write_summary(stage1_dfs, stage2_dfs, stage2b_df, stage3_dfs, series_dfs, OUTPUT_DIR)

    print(f"\n── Objective 3 complete. Outputs in {OUTPUT_DIR} ──")


if __name__ == "__main__":
    main()
