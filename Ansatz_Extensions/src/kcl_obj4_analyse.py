"""
Objective 4 analysis: GA-optimal hyperparameter patterns + SHAP attributions.

Reads Stage 3 energies_stage3_*.csv for procedural/structural hyperparam tables,
and trained XGBoost models + records for SHAP feature attributions (HEA/UCCSD).

Writes (under ../data/O4/):
  obj4_procedural_hyperparams.csv
  obj4_structural_hyperparams.csv
  obj4_hyperparam_summary.txt
  {HEA,UCCSD}_shap_summary.png
  {HEA,UCCSD}_shap_feature_groups.png

SHAP is run only for HEA and UCCSD. NN-HEA reuses the HEA XGBoost model trained
on the same ansatz; a separate SHAP run would duplicate HEA feature attributions.

Usage:
    python3 kcl_obj4_analyse.py

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb

from kcl_util_analysis import (
    IMPL_DATA,
    PARAMS_Q_SIZE,
    STAGE3_ANSATZES,
    _default_x_vec,
    analyse_stage3_hams,
    fmt,
    parse_float,
    write_csv,
)

OUTPUT_DIR = '../data/O4'
NN_SEED = -1

# O4: HEA/NN-HEA structural IND:2–5; procedural IND:6–13 shared across all three.
STRUCTURAL_INDICES = [
    (2, 'reps'),
    (3, 'entanglement'),
    (4, 'rotation_blocks'),
    (5, 'skip_final_rotation'),
]
PROCEDURAL_INDICES = [
    (6, 'optimizer_choice'),
    (7, 'iter_max'),
    (8, 'coeff_cutoff'),
    (9, 'atol'),
    (10, 'shots'),
    (11, 'gradient_method'),
    (12, 'initial_point_seed'),
    (13, 'max_evals_grouped'),
]
FLOAT_PROCEDURAL = {'coeff_cutoff', 'atol'}

ENTANGLEMENT_LABELS = {0: 'linear', 1: 'full', 2: 'circular'}
ROTATION_LABELS = {0: 'ry', 1: 'ry_rz', 2: 'rx_ry'}
OPTIMIZER_LABELS = {0: 'COBYLA', 1: 'SPSA', 2: 'L_BFGS_B'}
GRADIENT_LABELS = {0: 'None', 1: 'ParamShift', 2: 'FiniteDiff'}
SKIP_FINAL_LABELS = {0: 'False', 1: 'True'}

CATEGORICAL_LABELS = {
    'entanglement': ENTANGLEMENT_LABELS,
    'rotation_blocks': ROTATION_LABELS,
    'skip_final_rotation': SKIP_FINAL_LABELS,
    'optimizer_choice': OPTIMIZER_LABELS,
    'gradient_method': GRADIENT_LABELS,
}

OBJ4_PROCEDURAL_FIELDS = (
    ['implementation', 'ham_prefix', 'n_qubits', 'dst']
    + [c for _, name in PROCEDURAL_INDICES
       for c in (f'{name}_default', name)]
)
OBJ4_STRUCTURAL_FIELDS = (
    ['implementation', 'ham_prefix', 'n_qubits', 'dst']
    + [name for _, name in STRUCTURAL_INDICES]
)

SUMMARY_NOTE_OBJ4 = (
    "Objective 4 — GA-optimal hyperparameter patterns (descriptive only).\n"
    "Source: Stage 3 optimal_x_vec (last successful row per ham_prefix).\n"
    "Procedural table (IND:6–13): solver knobs for HEA, UCCSD, and NN-HEA,\n"
    "with *_default columns (default_hea_x_vec / default_uccsd_x_vec) beside\n"
    "each GA-optimal value for side-by-side comparison.\n"
    "Structural table (IND:2–5): EfficientSU2 geometry for HEA vs NN-HEA only.\n"
    "UCCSD ansatz structure differs (initial_state, reps, electron_fraction,\n"
    "generalized) and is omitted from the structural table — not blank columns.\n"
    "Sample size is small (Stage 3 Hamiltonians); report value lists and modes,\n"
    "not statistical frequency claims.\n"
)

SHAP_IMPLS = ["HEA", "UCCSD"]
SHAP_IMPL_DATA_DIRS = {
    "HEA": "../data/HEA",
    "UCCSD": "../data/UCCSD",
}
SHAP_RECORDS_CSV = {
    impl: os.path.join(SHAP_IMPL_DATA_DIRS[impl], f"records_{impl}.csv")
    for impl in SHAP_IMPLS
}
SHAP_HAM_CSV = {
    impl: os.path.join(SHAP_IMPL_DATA_DIRS[impl], f"hamiltonian_{impl}.csv")
    for impl in SHAP_IMPLS
}
MODEL_FILES = {
    "HEA": "model_hea_pre_xgb_28.json",
    "UCCSD": "model_uccsd_pre_xgb_28.json",
}
SHAP_PARAMS_Q_SIZE = {
    "HEA": 14,
    "UCCSD": 14,
}
SHAP_SAMPLE_CAP = None
SHAP_RANDOM_STATE = 42


def _parse_optimal_x_vec(raw):
    """Parse JSON list of length >= 14 from Stage 3 optimal_x_vec; else None."""
    if raw is None or not str(raw).strip():
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list) or len(data) < PARAMS_Q_SIZE:
        return None
    return [float(v) for v in data[:PARAMS_Q_SIZE]]


def _last_success_by_prefix(stage3_rows, ansatz):
    """Last successful Stage 3 row per ham_prefix for one ansatz."""
    by_prefix = {}
    for r in stage3_rows:
        if r.get('ansatz') != ansatz or r.get('status') != 'success':
            continue
        prefix = r.get('ham_prefix', '').strip()
        if prefix:
            by_prefix[prefix] = r
    return by_prefix


def _decode_int_slot(name, value):
    """Round categorical/int slot; preserve NN_SEED sentinel for initial_point_seed."""
    rounded = int(round(value))
    if name == 'initial_point_seed' and int(round(value)) == NN_SEED:
        return NN_SEED
    return rounded


def _format_slot_value(name, value):
    if name in FLOAT_PROCEDURAL:
        return fmt(value, precision=8)
    return str(_decode_int_slot(name, value))


def _mode_note(name, values):
    """Mode label string for int/categorical slots (handles NN_SEED)."""
    if not values:
        return ''
    counts = Counter(values)
    mode_val, mode_n = counts.most_common(1)[0]
    label_map = CATEGORICAL_LABELS.get(name)
    if name == 'initial_point_seed' and mode_val == NN_SEED:
        label = 'NN_SEED'
    elif label_map is not None:
        label = label_map.get(mode_val, str(mode_val))
    else:
        label = str(mode_val)
    return f"{label} ({mode_val}) in {mode_n}/{len(values)}"


def analyse_hyperparameter_patterns(stage3_rows):
    """O4: decode Stage 3 optimal_x_vec into procedural (all 3) and structural (HEA/NN-HEA) tables plus a descriptive summary."""
    procedural_rows = []
    structural_rows = []
    proc_values = {impl: {name: [] for _, name in PROCEDURAL_INDICES} for impl in STAGE3_ANSATZES}
    struct_values = {impl: {name: [] for _, name in STRUCTURAL_INDICES} for impl in ('HEA', 'NN_HEA')}
    dst_by_impl = {impl: [] for impl in STAGE3_ANSATZES}
    n_skipped = 0

    for ansatz in STAGE3_ANSATZES:
        by_prefix = _last_success_by_prefix(stage3_rows, ansatz)
        for prefix in sorted(by_prefix.keys()):
            r = by_prefix[prefix]
            x_vec = _parse_optimal_x_vec(r.get('optimal_x_vec', ''))
            if x_vec is None:
                print(
                    f"  [WARN] O4: missing/invalid optimal_x_vec for "
                    f"{ansatz}/{prefix} — re-run Stage 3"
                )
                n_skipped += 1
                continue

            dst = parse_float(r.get('dst'))
            n_qubits = r.get('n_qubits', '')
            n_qubits_int = int(n_qubits) if str(n_qubits).strip() else 0
            default_x = _default_x_vec(ansatz, n_qubits_int)
            proc_row = {
                'implementation': ansatz,
                'ham_prefix': prefix,
                'n_qubits': n_qubits,
                'dst': fmt(dst, precision=6),
            }
            for idx, name in PROCEDURAL_INDICES:
                proc_row[f'{name}_default'] = _format_slot_value(name, default_x[idx])
                if name in FLOAT_PROCEDURAL:
                    val = float(x_vec[idx])
                    proc_row[name] = fmt(val, precision=8)
                    proc_values[ansatz][name].append(val)
                else:
                    val = _decode_int_slot(name, x_vec[idx])
                    proc_row[name] = str(val)
                    proc_values[ansatz][name].append(val)
            procedural_rows.append(proc_row)
            if dst is not None:
                dst_by_impl[ansatz].append((prefix, dst))

            if ansatz in ('HEA', 'NN_HEA'):
                struct_row = {
                    'implementation': ansatz,
                    'ham_prefix': prefix,
                    'n_qubits': n_qubits,
                    'dst': fmt(dst, precision=6),
                }
                for idx, name in STRUCTURAL_INDICES:
                    val = _decode_int_slot(name, x_vec[idx])
                    struct_row[name] = str(val)
                    struct_values[ansatz][name].append(val)
                structural_rows.append(struct_row)

    summary_lines = [
        "=" * 70,
        "Objective 4 — Hyperparameter Pattern Summary",
        "=" * 70,
        "",
        SUMMARY_NOTE_OBJ4,
    ]

    if not procedural_rows:
        summary_lines.extend([
            "No parseable optimal_x_vec rows found.",
            "Re-run Stage 3 for HEA / UCCSD / NN-HEA so energies_stage3_*.csv "
            "includes the optimal_x_vec column.",
            f"Skipped rows: {n_skipped}",
        ])
        return procedural_rows, structural_rows, summary_lines

    summary_lines.append("--- Procedural hyperparameters (IND:6–13) ---")
    summary_lines.append("")
    for ansatz in STAGE3_ANSATZES:
        cols = proc_values[ansatz]
        if not any(cols.values()):
            summary_lines.append(f"  {ansatz}: no rows")
            summary_lines.append("")
            continue
        n_ham = len(next(iter(cols.values())))
        summary_lines.append(f"  {ansatz} (n={n_ham} Hamiltonians):")
        # Defaults are constant across hams for IND:6–13; print once for comparison.
        default_x = _default_x_vec(ansatz, 4)
        default_parts = [
            f"{name}={_format_slot_value(name, default_x[idx])}"
            for idx, name in PROCEDURAL_INDICES
        ]
        summary_lines.append(f"    defaults: {', '.join(default_parts)}")
        for _, name in PROCEDURAL_INDICES:
            vals = cols[name]
            if name in FLOAT_PROCEDURAL:
                shown = [fmt(v, precision=8) for v in vals]
                summary_lines.append(f"    {name}: {shown}")
            else:
                note = _mode_note(name, vals)
                summary_lines.append(f"    {name}: {vals} — {note}")
                if name == 'initial_point_seed':
                    n_nn = sum(1 for v in vals if v == NN_SEED)
                    if n_nn:
                        summary_lines.append(
                            f"      NN_SEED (-1) present in {n_nn}/{len(vals)} rows"
                        )
        if dst_by_impl[ansatz]:
            ranked = sorted(dst_by_impl[ansatz], key=lambda t: t[1], reverse=True)
            top = ", ".join(f"{p} (dst={d:.4g})" for p, d in ranked[:3])
            summary_lines.append(f"    largest dst: {top}")
        summary_lines.append("")

    summary_lines.append("--- Structural hyperparameters (IND:2–5, HEA / NN-HEA) ---")
    summary_lines.append("")
    summary_lines.append(
        "  UCCSD omitted: ansatz structure uses initial_state / electron_fraction / "
        "generalized, not HEA EfficientSU2 slots."
    )
    summary_lines.append("")
    for ansatz in ('HEA', 'NN_HEA'):
        cols = struct_values[ansatz]
        if not any(cols.values()):
            summary_lines.append(f"  {ansatz}: no rows")
            summary_lines.append("")
            continue
        n_ham = len(next(iter(cols.values())))
        summary_lines.append(f"  {ansatz} (n={n_ham} Hamiltonians):")
        for _, name in STRUCTURAL_INDICES:
            vals = cols[name]
            note = _mode_note(name, vals)
            summary_lines.append(f"    {name}: {vals} — {note}")
        summary_lines.append("")

    if n_skipped:
        summary_lines.append(f"Skipped rows without optimal_x_vec: {n_skipped}")

    return procedural_rows, structural_rows, summary_lines


def write_obj4_summary(lines, path):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Written: {path}")


def _resolve_model_path(impl_name):
    data_dir = SHAP_IMPL_DATA_DIRS.get(impl_name, ".")
    patterns = {
        "HEA": "model_hea_pre_xgb_*.json",
        "UCCSD": "model_uccsd_pre_xgb_*.json",
    }
    pattern = patterns.get(impl_name, "")
    if pattern:
        matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if matches:
            return matches[-1]
    default_path = os.path.join(data_dir, MODEL_FILES[impl_name])
    if os.path.exists(default_path):
        return default_path
    return MODEL_FILES[impl_name]


def _load_npy_rows_for_records(records_path, npy_folder):
    """Raw X rows for successful records via (x_file, npy_index); impl-filtered."""
    from kcl_util import read_records_csv

    rows = read_records_csv(records_path)
    success_rows = [r for r in rows if r.get("status") == "success"]

    npy_cache = {}
    x_rows = []
    seen = set()

    for row in success_rows:
        x_file = row.get("x_file", "")
        npy_index = row.get("npy_index", "")
        if not x_file or npy_index == "":
            continue
        key = (x_file, str(npy_index))
        if key in seen:
            continue
        seen.add(key)

        if x_file not in npy_cache:
            npy_path = os.path.join(npy_folder, x_file + ".npy")
            if not os.path.exists(npy_path):
                print(f"[WARN] Missing npy file: {npy_path}")
                continue
            npy_cache[x_file] = np.load(npy_path, allow_pickle=True)

        data = npy_cache[x_file]
        idx = int(npy_index)
        if idx < 0 or idx >= len(data):
            print(f"[WARN] npy_index {idx} out of range for {x_file} (n={len(data)})")
            continue
        x_rows.append(np.asarray(data[idx], dtype=np.float64))

    return x_rows


def run_shap_analysis(impl_name, model_file, records_csv, ham_csv, output_dir, npy_folder):
    """SHAP beeswarm + hyperparam-vs-ham importance for one XGBoost model."""
    print(f"[{impl_name}] SHAP analysis — model={model_file}, records={records_csv}")
    _ = ham_csv  # retained for API consistency with main()

    from kcl_util import vec_to_fixed_size_vec

    model = xgb.XGBRegressor()
    model.load_model(model_file)
    n_features = model.n_features_in_
    print(f"[{impl_name}] Model expects {n_features} features")

    print(f"[{impl_name}] Reconstructing X from npy files in {npy_folder}")
    x_raw = _load_npy_rows_for_records(records_csv, npy_folder)
    if not x_raw:
        print(f"[SKIP] {impl_name} SHAP: no npy rows found for records")
        return

    print(f"[{impl_name}] Loaded {len(x_raw)} raw feature vectors")
    x_padded = np.array([
        vec_to_fixed_size_vec(n_features, row) for row in x_raw
    ])

    if SHAP_SAMPLE_CAP is not None and len(x_padded) > SHAP_SAMPLE_CAP:
        rng = np.random.default_rng(SHAP_RANDOM_STATE)
        idx = rng.choice(len(x_padded), size=SHAP_SAMPLE_CAP, replace=False)
        x_sample = x_padded[idx]
        print(f"[{impl_name}] Subsampled to {SHAP_SAMPLE_CAP} rows for SHAP")
    else:
        x_sample = x_padded

    print(f"[{impl_name}] Computing SHAP values for {len(x_sample)} samples")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    fig_summary = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_sample, show=False)
    summary_path = os.path.join(output_dir, f"{impl_name}_shap_summary.png")
    fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"[{impl_name}] Saved {summary_path}")

    n_hyper = SHAP_PARAMS_Q_SIZE[impl_name]
    hyper_block = float(np.abs(shap_values[:, :n_hyper]).mean())
    ham_block = float(np.abs(shap_values[:, n_hyper:]).mean())

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ["Hyperparameters", "Hamiltonian features"]
    values = [hyper_block, ham_block]
    bars = ax.barh(labels, values, color=["#4C72B0", "#DD8452"])
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f" {val:.4f}", va="center", ha="left", fontsize=10,
        )
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"{impl_name} — feature group importance")
    groups_path = os.path.join(output_dir, f"{impl_name}_shap_feature_groups.png")
    fig.savefig(groups_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{impl_name}] Saved {groups_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("── Objective 4: hyperparameter patterns ──")
    stage3_sources = [
        (os.path.join(IMPL_DATA[ansatz], f'energies_stage3_{ansatz}.csv'), ansatz)
        for ansatz in STAGE3_ANSATZES
    ]
    stage3_rows = analyse_stage3_hams(stage3_sources)
    procedural_rows, structural_rows, obj4_lines = analyse_hyperparameter_patterns(stage3_rows)
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj4_procedural_hyperparams.csv'),
        OBJ4_PROCEDURAL_FIELDS,
        procedural_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj4_structural_hyperparams.csv'),
        OBJ4_STRUCTURAL_FIELDS,
        structural_rows,
    )
    write_obj4_summary(obj4_lines, os.path.join(OUTPUT_DIR, 'obj4_hyperparam_summary.txt'))

    print("\n── Objective 4: SHAP attributions ──")
    for impl in SHAP_IMPLS:
        if not os.path.exists(SHAP_RECORDS_CSV[impl]):
            print(f"[SKIP] {impl} SHAP: records CSV not found at {SHAP_RECORDS_CSV[impl]}")
            continue
        model_path = _resolve_model_path(impl)
        if os.path.exists(model_path):
            run_shap_analysis(
                impl, model_path, SHAP_RECORDS_CSV[impl], SHAP_HAM_CSV[impl],
                OUTPUT_DIR, SHAP_IMPL_DATA_DIRS[impl],
            )
        else:
            print(f"[SKIP] {impl} SHAP: model file not found at {model_path}")

    print(f"\n── Objective 4 complete. Outputs in {OUTPUT_DIR} ──")


if __name__ == "__main__":
    main()
