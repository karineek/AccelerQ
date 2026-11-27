"""
Shared helpers for objective-scoped analysis scripts (O1–O4).

CSV I/O, Stage 3 loading, default x_vec construction, and common constants.
No matplotlib / SHAP / VQE dependencies.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import csv
import os

import numpy as np

IMPL_DATA = {
    'HEA': '../data/HEA',
    'UCCSD': '../data/UCCSD',
    'NN_HEA': '../data/NN_HEA',
}
STAGE3_ANSATZES = ('HEA', 'UCCSD', 'NN_HEA')
PARAMS_Q_SIZE = 14

# Default hyperparameter vectors (seed <= 1), mirroring kcl_util_HEA / kcl_util_UCCSD
# without importing VQE wrappers (avoids qiskit dependency for analysis-only runs).
_DEFAULT_HEA = [3, 0, 1, 0, 0, 500, 1e-5, 1e-6, 10**4, 0, 0, 5]
_DEFAULT_UCCSD_SOLVER = [0, 2, 0, 0, 500, 1e-5, 1e-6, 10**4, 0, 0, 5]

NEAR_ZERO_BASELINE_THRESHOLD = 1e-6


def load_csv(path):
    if not os.path.exists(path):
        print(f"  Warning: {path} not found — skipping.")
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def delta_opt(e_before, e_after):
    """Fractional improvement: positive = e_after is lower (better) than e_before."""
    if e_before is None or e_after is None:
        return None
    if abs(e_before) < NEAR_ZERO_BASELINE_THRESHOLD:
        return None
    return (e_before - e_after) / abs(e_before)


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}")


def fmt(val, precision=6):
    if val is None or val == '':
        return ''
    try:
        return f'{float(val):.{precision}f}'
    except (ValueError, TypeError):
        return str(val)


def _default_hea_x_vec(n_qubits):
    defaults = _DEFAULT_HEA
    return np.array([
        float(n_qubits), 1.0,
        float(defaults[0]), float(defaults[1]), float(defaults[2]), float(defaults[3]),
        float(defaults[4]), float(defaults[5]), float(defaults[6]), float(defaults[7]),
        float(defaults[8]), float(defaults[9]), float(defaults[10]), float(defaults[11]),
    ], dtype=float)


def _default_uccsd_x_vec(n_qubits):
    defaults = _DEFAULT_UCCSD_SOLVER
    return np.array([
        float(n_qubits), 1.0,
        float(defaults[0]), float(defaults[1]), 0.0, float(defaults[2]),
        float(defaults[3]), float(defaults[4]), float(defaults[5]), float(defaults[6]),
        float(defaults[7]), float(defaults[8]), float(defaults[9]), float(defaults[10]),
    ], dtype=float)


def _default_x_vec(ansatz_label, n_qubits):
    if ansatz_label in ('HEA', 'NN_HEA'):
        return _default_hea_x_vec(n_qubits)
    if ansatz_label == 'UCCSD':
        return _default_uccsd_x_vec(n_qubits)
    raise ValueError(f"Unknown ansatz: {ansatz_label}")


def analyse_stage3_hams(stage3_sources):
    """stage3_sources: iterable of (csv_path, ansatz_label) pairs."""
    rows_out = []
    for path, ansatz_label in stage3_sources:
        for r in load_csv(path):
            score_ml = parse_float(r.get('score_ml_only'))
            score_full = parse_float(r.get('score_full_accelerq'))
            dst = parse_float(r.get('dst'))
            delta = delta_opt(score_ml, score_full)
            rows_out.append({
                'ansatz': ansatz_label,
                'ham_prefix': r.get('ham_prefix', ''),
                'n_qubits': r.get('n_qubits', ''),
                'score_ml_only': fmt(score_ml),
                'score_full_accelerq': fmt(score_full),
                'delta_opt': fmt(delta),
                'dst': fmt(dst),
                'optimal_x_vec': r.get('optimal_x_vec', ''),
                'ml_only_x_vec': r.get('ml_only_x_vec', ''),
                'status': r.get('status', ''),
            })
    return rows_out
