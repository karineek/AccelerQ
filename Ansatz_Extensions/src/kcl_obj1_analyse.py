"""
Objective 1 (Layers 1+2) analysis script.

Reads:
  records_HEA.csv              — stage 1+2 output for HEA training Hamiltonians
  records_UCCSD.csv            — stage 1+2 output for UCCSD training Hamiltonians
  hamiltonian_HEA.csv          — Hamiltonian metadata for HEA (if available)
  hamiltonian_UCCSD.csv        — Hamiltonian metadata for UCCSD
  energies_stage3_HEA.csv      — stage 3 ML-Only and Full AccelerQ scores (HEA)
  energies_stage3_UCCSD.csv    — stage 3 ML-Only and Full AccelerQ scores (UCCSD)
  energies_stage3_NN_HEA.csv   — stage 3 ML-Only and Full AccelerQ scores (NN-HEA)
  Final_VQE_run/Final_VQE.csv  — true Stage 3 VQE (HEA + NN-HEA; is_baseline column)
  ham_reference_energies.csv   — reference ground-state energies

Writes (under ../data/O1/):
  obj1_training_hams.csv              — Comparison A: baseline vs ML-only proxy
  obj1_training_hams_near_zero_baseline.csv — Comp A rows with |E_baseline| < 1e-6
  obj1_stage3_hams.csv                — Comparison B: surrogate ML-only vs Full AccelerQ
  obj1_final_vqe.csv                  — Comparison C: optimised VQE vs reference
  obj1_baseline_vs_opt.csv            — Comparison D: baseline vs optimised real VQE
  obj1_comparisonC_by_reference_type.csv — Comp C abs_error by reference_type
  obj1_comparisonD_summary_stats.csv  — Comp D mean/std/median + outliers
  obj1_dst_vs_delta_opt.png           — Stage 3 dst vs delta_opt scatter
  obj1_hea_vs_nnhea_delta_opt.png     — HEA vs NN-HEA paired delta_opt dumbbell
  obj1_training_delta_opt_by_ansatz.png — Comparison A strip/box by ansatz
  obj1_baseline_vs_opt_energy.png     — Comparison D energy dumbbell
  obj1_stage3_ham_features.csv        — post-hoc Stage 3 sparsity/term_count/coeff_var
  obj1_structure_vs_abs_error.png     — Comp C structure vs abs_error (descriptive)
  obj1_structure_vs_delta_opt.png     — Comp D structure vs delta_opt (descriptive)
  obj1_summary.txt                    — human-readable summary of all comparisons

Usage:
    python3 kcl_obj1_analyse.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

from kcl_util_analysis import (
    IMPL_DATA,
    NEAR_ZERO_BASELINE_THRESHOLD,
    PARAMS_Q_SIZE,
    STAGE3_ANSATZES,
    _default_x_vec,
    analyse_stage3_hams,
    delta_opt,
    fmt,
    load_csv,
    parse_float,
    parse_int,
    write_csv,
)

FINAL_VQE_CSV = '../data/Final_VQE_run/Final_VQE.csv'
REFERENCE_ENERGIES_CSV = '../data/ham_reference_energies.csv'
OUTPUT_DIR = '../data/O1'
HAM_FEATURE_NAMES = ('sparsity', 'term_count', 'coefficient_variance')
_PREFIX_RE = re.compile(r'^(\d+)qubits_(\d+)$')

# Match kcl_obj3_analyse.IMPL_COLOURS for a coherent results chapter.
ANSATZ_COLOURS = {
    'HEA': '#4C72B0',
    'UCCSD': '#DD8452',
    'NN_HEA': '#55A868',
}
HAM_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<']

TRAINING_FIELDS = [
    'ansatz', 'ham_id', 'n_qubits', 'dataset',
    'n_baseline_records',
    'E_baseline_mean',
    'E_baseline_std',
    'n_test_records',
    'E_pred_min',
    'E_pred_mean',
    'delta_opt',
    'E_baseline_near_zero',
    'model_mse_mean',
]

NEAR_ZERO_BASELINE_FIELDS = [
    'ansatz', 'ham_id', 'n_qubits',
    'E_baseline_mean', 'E_pred_min', 'E_pred_mean',
]

COMPARISON_C_BY_REF_FIELDS = [
    'implementation', 'reference_type', 'n', 'mean_abs_error', 'std_abs_error',
]

COMPARISON_D_SUMMARY_FIELDS = [
    'implementation', 'n',
    'mean_delta_opt', 'std_delta_opt', 'median_delta_opt',
    'outlier_ham_prefixes',
]

STAGE3_FIELDS = [
    'ansatz', 'ham_prefix', 'n_qubits',
    'score_ml_only',
    'score_full_accelerq',
    'delta_opt',
    'dst',
    'optimal_x_vec',
    'ml_only_x_vec',
    'status',
]

FINAL_VQE_FIELDS = [
    'implementation', 'ham_prefix', 'n_qubits', 'dataset',
    'vqe_energy', 'reference_energy', 'reference_type',
    'signed_error', 'abs_error',
    'n_iterations', 'iter_max', 'converged', 'status',
    'sparsity', 'term_count', 'coefficient_variance',
]

BASELINE_VS_OPT_FIELDS = [
    'implementation', 'ham_prefix', 'n_qubits',
    'E_baseline', 'E_optimised', 'delta_opt',
    'n_iterations_baseline', 'n_iterations_optimised',
    'iter_max_baseline', 'iter_max_optimised',
    'converged_baseline', 'converged_optimised',
    'run_time_s_baseline', 'run_time_s_optimised',
    'sparsity', 'term_count', 'coefficient_variance',
]

STAGE3_HAM_FEATURE_FIELDS = [
    'ham_prefix', 'n_qubits', 'sparsity', 'term_count', 'coefficient_variance',
]

SUMMARY_NOTE_TRAINING = (
    "Comparison A — training Hamiltonians (<=16 qubits).\n"
    "Baseline = true VQE energy with default hyperparameters, identified by\n"
    "matching the hyperparam slice from .npy rows to default_x_vec(n_qubits).\n"
    "ML-Only proxy = min predicted_energy across test+test_extra split records per Hamiltonian.\n"
    "These are surrogate predictions on random draws, not a GA-optimised result.\n"
    "delta_opt = (E_baseline_mean - E_pred_min) / |E_baseline_mean|.\n"
    "Limitation: dst not available for training Hamiltonians "
    "(x_vec not stored in records CSV).\n"
    "NN-HEA is omitted from Comparison A: it shares the HEA ansatz and XGBoost\n"
    "surrogate (model_hea_pre_xgb_*), so training-Hamiltonian metrics are identical\n"
    "by construction.\n"
)

SUMMARY_NOTE_STAGE3 = (
    "Comparison B — stage 3 Hamiltonians (>=20 qubits).\n"
    "ML-Only = min surrogate score from 499-seed random sweep (before GA).\n"
    "Full AccelerQ = min surrogate score after 50 GA iterations.\n"
    "Both are surrogate predictions — no true VQE Baseline exists at this scale.\n"
    "delta_opt = (score_ml_only - score_full_accelerq) / |score_ml_only|.\n"
    "dst = Euclidean distance of GA-final x_vec from default x_vec.\n"
)

SUMMARY_NOTE_FINAL_VQE = (
    "Comparison C — optimised Stage 3 VQE vs reference (HEA + NN-HEA).\n"
    "Uses Final_VQE.csv rows with is_baseline=False (GA-optimal x_vec).\n"
    "Baseline rows are excluded here; see Comparison D for baseline vs optimised.\n"
    "Reference energy from ham_reference_energies.csv "
    "(exact for QunaSys, best-guess for Molecular).\n"
    "signed_error = vqe_energy - reference_energy "
    "(negative = VQE above reference / worse for ground state).\n"
    "abs_error = |signed_error|.\n"
    "UCCSD has no Final VQE row — remains surrogate-only (Comparison B).\n"
)

SUMMARY_NOTE_BASELINE_VS_OPT = (
    "Comparison D — real AccelerQ improvement on Stage 3 Hamiltonians.\n"
    "Pairs baseline (default_hea_x_vec, is_baseline=True) vs optimised\n"
    "(GA optimal_x_vec, is_baseline=False) from Final_VQE.csv per\n"
    "(implementation, ham_prefix). Both arms share iter_max cap 500.\n"
    "HEA and NN-HEA share the same physical baseline run (written once, tagged twice).\n"
    "Reports n_iterations, iter_max, converged, and run_time_s "
    "(baseline vs optimised).\n"
    "delta_opt = (E_baseline - E_optimised) / |E_baseline| "
    "(positive = AccelerQ lowered the energy).\n"
    "Answers: does AccelerQ improve real execution vs default hyperparameters?\n"
    "Distinct from Comparison C (distance to reference ground state).\n"
)


def _n_qubits_from_prefix(prefix):
    match = _PREFIX_RE.match(str(prefix).strip())
    return int(match.group(1)) if match else None


def _resolve_ham_dir(ham_dir=None):
    """Repo hamiltonian/ path without importing kcl_util_hamiltonian (avoids juliacall)."""
    if ham_dir is not None:
        return ham_dir
    # Same formula as kcl_util_hamiltonian.REPO_HAMILTONIAN_DIR
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'hamiltonian')
    )
    return candidate


def load_stage3_ham_features(ham_prefixes, ham_dir=None, cache_path=None):
    """Structural features for Stage 3 hams; lazy-imports juliacall path."""
    try:
        from kcl_util import process_file
        from kcl_util_hamiltonian import compute_hamiltonian_features
    except Exception as exc:
        print(f"  Warning: cannot load Hamiltonian tooling (juliacall?) — "
              f"skipping Stage 3 features: {exc}")
        return {}

    ham_dir = _resolve_ham_dir(ham_dir)
    if cache_path is None:
        cache_path = os.path.join(OUTPUT_DIR, 'obj1_stage3_ham_features.csv')

    features = {}
    prefixes = sorted({p.strip() for p in ham_prefixes if p and str(p).strip()})
    for prefix in prefixes:
        data_name = f"{prefix}.data"
        data_path = os.path.join(ham_dir, data_name)
        if not os.path.isfile(data_path):
            print(f"  Warning: missing Hamiltonian file {data_path} — skipping features")
            continue
        try:
            _, ham = process_file(ham_dir, data_name)
            sparsity, term_count, coeff_var = compute_hamiltonian_features(ham)
        except Exception as exc:
            print(f"  Warning: feature extraction failed for {prefix}: {exc}")
            continue
        n_qubits = _n_qubits_from_prefix(prefix)
        features[prefix] = {
            'sparsity': sparsity,
            'term_count': term_count,
            'coefficient_variance': coeff_var,
            'n_qubits': n_qubits if n_qubits is not None else '',
        }

    cache_rows = []
    for prefix in sorted(features.keys()):
        feat = features[prefix]
        cache_rows.append({
            'ham_prefix': prefix,
            'n_qubits': feat.get('n_qubits', ''),
            'sparsity': fmt(feat['sparsity'], precision=10),
            'term_count': feat['term_count'],
            'coefficient_variance': fmt(feat['coefficient_variance'], precision=10),
        })
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    write_csv(cache_path, STAGE3_HAM_FEATURE_FIELDS, cache_rows)
    return features


def _attach_ham_features(rows, features):
    """Join sparsity/term_count/coefficient_variance onto rows by ham_prefix."""
    for r in rows:
        prefix = (r.get('ham_prefix') or '').strip()
        feat = features.get(prefix) if prefix else None
        if feat is None:
            for name in HAM_FEATURE_NAMES:
                r[name] = ''
            continue
        r['sparsity'] = fmt(feat['sparsity'], precision=10)
        r['term_count'] = feat['term_count']
        r['coefficient_variance'] = fmt(feat['coefficient_variance'], precision=10)
    return rows


def collect_stage3_ham_prefixes(final_vqe_path, stage3_rows=None):
    """Union of ham_prefix from Final VQE CSV and successful Stage 3 rows."""
    prefixes = set()
    for r in load_csv(final_vqe_path):
        prefix = (r.get('ham_prefix') or '').strip()
        if prefix:
            prefixes.add(prefix)
    if stage3_rows:
        for r in stage3_rows:
            if r.get('status') != 'success':
                continue
            prefix = (r.get('ham_prefix') or '').strip()
            if prefix:
                prefixes.add(prefix)
    return sorted(prefixes)


def _coerce_is_baseline(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ('true', '1', 'yes')


def _is_near_zero_baseline_flag(val):
    """True for bool True or CSV string 'True' / 'true' / '1'."""
    if val is True:
        return True
    if val is False or val is None or val == '':
        return False
    return str(val).strip().lower() in ('true', '1', 'yes')


def _load_hyperparams_from_npy(x_file, npy_index, npy_folder):
    npy_path = os.path.join(npy_folder, x_file + '.npy')
    if not os.path.exists(npy_path):
        print(f"  Warning: missing npy file {npy_path}")
        return None
    data = np.load(npy_path, allow_pickle=True)
    idx = int(npy_index)
    if idx < 0 or idx >= len(data):
        print(f"  Warning: npy_index {idx} out of range for {x_file} (n={len(data)})")
        return None
    row = np.asarray(data[idx], dtype=np.float64)
    if len(row) < PARAMS_Q_SIZE:
        return None
    return row[:PARAMS_Q_SIZE]


def _is_baseline_record(record, ansatz_label, n_qubits, npy_cache, npy_folder):
    x_file = record.get('x_file', '')
    npy_index = record.get('npy_index', '')
    if not x_file or npy_index == '':
        return False
    key = (x_file, str(npy_index))
    if key not in npy_cache:
        npy_cache[key] = _load_hyperparams_from_npy(x_file, npy_index, npy_folder)
    hyperparams = npy_cache[key]
    if hyperparams is None:
        return False
    default = _default_x_vec(ansatz_label, n_qubits)
    return np.allclose(hyperparams, default, rtol=0, atol=0)


def analyse_training_hams(records_path, ham_path, ansatz_label, npy_folder):
    records = load_csv(records_path)
    hams = load_csv(ham_path)
    ham_meta = {r['ham_id']: r for r in hams}

    by_ham = defaultdict(list)
    for r in records:
        if r.get('status') == 'success':
            by_ham[r['ham_id']].append(r)

    npy_cache = {}
    rows = []
    for ham_id, recs in sorted(by_ham.items(), key=lambda x: int(x[0])):
        meta = ham_meta.get(ham_id, {})
        n_qubits = parse_int(meta.get('n_qubits'))
        if n_qubits is None:
            for r in recs:
                n_qubits = parse_int(r.get('n_qubits'))
                if n_qubits is not None:
                    break
        if n_qubits is None:
            print(f"  Warning: no n_qubits for ham_id={ham_id}, skipping baseline matching")
            continue

        baseline_energies = []
        for r in recs:
            e = parse_float(r.get('true_energy'))
            if e is not None and _is_baseline_record(r, ansatz_label, n_qubits, npy_cache, npy_folder):
                baseline_energies.append(e)

        test_preds = []
        test_errors = []
        for r in recs:
            split = r.get('split', '')
            if split in ('test', 'test_extra'):
                pred_energy = parse_float(r.get('predicted_energy'))
                sq_error = parse_float(r.get('squared_error'))
                if pred_energy is not None:
                    test_preds.append(pred_energy)
                if sq_error is not None:
                    test_errors.append(sq_error)

        E_bl_mean = float(np.mean(baseline_energies)) if baseline_energies else None
        E_bl_std = float(np.std(baseline_energies)) if baseline_energies else None
        E_pred_min = float(min(test_preds)) if test_preds else None
        E_pred_mean = float(np.mean(test_preds)) if test_preds else None
        mse_mean = float(np.mean(test_errors)) if test_errors else None
        e_bl_near_zero = (
            E_bl_mean is not None and abs(E_bl_mean) < NEAR_ZERO_BASELINE_THRESHOLD
        )
        delta = delta_opt(E_bl_mean, E_pred_min)

        rows.append({
            'ansatz': ansatz_label,
            'ham_id': ham_id,
            'n_qubits': meta.get('n_qubits', ''),
            'dataset': meta.get('dataset', ''),
            'n_baseline_records': len(baseline_energies),
            'E_baseline_mean': fmt(E_bl_mean),
            'E_baseline_std': fmt(E_bl_std),
            'n_test_records': len(test_preds),
            'E_pred_min': fmt(E_pred_min),
            'E_pred_mean': fmt(E_pred_mean),
            'delta_opt': fmt(delta),
            'E_baseline_near_zero': str(bool(e_bl_near_zero)),
            'model_mse_mean': fmt(mse_mean),
        })

    return rows


def load_reference_energies(path):
    refs = {}
    for r in load_csv(path):
        prefix = r.get('ham_prefix', '').strip()
        ref_energy = parse_float(r.get('reference_energy'))
        if prefix and ref_energy is not None:
            refs[prefix] = r
    return refs


def analyse_final_vqe(final_vqe_path, reference_path):
    """Comparison C: optimised (is_baseline=False) VQE vs reference energies."""
    refs = load_reference_energies(reference_path)
    rows_out = []

    for r in load_csv(final_vqe_path):
        if r.get('status') != 'success':
            continue
        # Older CSVs without is_baseline are treated as optimised.
        if 'is_baseline' in r and r.get('is_baseline', '') != '':
            if _coerce_is_baseline(r.get('is_baseline')):
                continue
        prefix = r.get('ham_prefix', '').strip()
        vqe_e = parse_float(r.get('vqe_energy'))
        if vqe_e is None:
            continue

        ref_row = refs.get(prefix)
        if ref_row is None:
            print(f"  Warning: no reference energy for {prefix}")
            ref_energy = None
            ref_type = ''
            dataset = ''
        else:
            ref_energy = parse_float(ref_row.get('reference_energy'))
            ref_type = ref_row.get('reference_type', '')
            dataset = ref_row.get('dataset', '')

        signed_error = (vqe_e - ref_energy) if ref_energy is not None else None
        abs_error = abs(signed_error) if signed_error is not None else None

        rows_out.append({
            'implementation': r.get('implementation', ''),
            'ham_prefix': prefix,
            'n_qubits': r.get('n_qubits', ''),
            'dataset': dataset,
            'vqe_energy': fmt(vqe_e, precision=10),
            'reference_energy': fmt(ref_energy, precision=10),
            'reference_type': ref_type,
            'signed_error': fmt(signed_error, precision=10),
            'abs_error': fmt(abs_error, precision=10),
            'n_iterations': r.get('n_iterations', ''),
            'iter_max': r.get('iter_max', ''),
            'converged': r.get('converged', ''),
            'status': r.get('status', ''),
        })

    return rows_out


def analyse_baseline_vs_optimised(final_vqe_path):
    """Comparison D: pair baseline vs optimised real VQE per (impl, ham_prefix)."""
    by_key = {}
    for r in load_csv(final_vqe_path):
        if r.get('status') != 'success':
            continue
        prefix = r.get('ham_prefix', '').strip()
        impl = r.get('implementation', '').strip()
        if not prefix or not impl:
            continue
        # Rows without is_baseline cannot form a D pair
        if 'is_baseline' not in r or r.get('is_baseline', '') == '':
            continue
        is_baseline = _coerce_is_baseline(r.get('is_baseline'))
        by_key.setdefault((impl, prefix), {})[is_baseline] = r

    rows_out = []
    for (impl, prefix), arms in sorted(by_key.items()):
        baseline = arms.get(True)
        optimised = arms.get(False)
        if baseline is None or optimised is None:
            missing = 'baseline' if baseline is None else 'optimised'
            print(f"  Warning: Comparison D incomplete for {impl}/{prefix} — missing {missing}")
            continue

        e_bl = parse_float(baseline.get('vqe_energy'))
        e_opt = parse_float(optimised.get('vqe_energy'))
        delta = delta_opt(e_bl, e_opt)

        rows_out.append({
            'implementation': impl,
            'ham_prefix': prefix,
            'n_qubits': baseline.get('n_qubits') or optimised.get('n_qubits', ''),
            'E_baseline': fmt(e_bl, precision=10),
            'E_optimised': fmt(e_opt, precision=10),
            'delta_opt': fmt(delta, precision=6),
            'n_iterations_baseline': baseline.get('n_iterations', ''),
            'n_iterations_optimised': optimised.get('n_iterations', ''),
            'iter_max_baseline': baseline.get('iter_max', ''),
            'iter_max_optimised': optimised.get('iter_max', ''),
            'converged_baseline': baseline.get('converged', ''),
            'converged_optimised': optimised.get('converged', ''),
            'run_time_s_baseline': baseline.get('run_time_s', ''),
            'run_time_s_optimised': optimised.get('run_time_s', ''),
        })

    return rows_out


def _pearson_corr(x, y):
    if len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _save_fig(fig, output_dir, filename):
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Written: {out_path}")


def plot_dst_vs_delta_opt(stage3_rows, output_dir):
    """Plot 1: Stage 3 Comparison B — dst vs delta_opt scatter by ansatz."""
    by_ansatz = {a: {'dst': [], 'delta_opt': []} for a in STAGE3_ANSATZES}
    for r in stage3_rows:
        if r.get('status') != 'success':
            continue
        ansatz = r.get('ansatz', '')
        if ansatz not in by_ansatz:
            continue
        delta = parse_float(r.get('delta_opt'))
        dst = parse_float(r.get('dst'))
        if delta is None or dst is None:
            continue
        by_ansatz[ansatz]['dst'].append(dst)
        by_ansatz[ansatz]['delta_opt'].append(delta)

    n_total = sum(len(v['dst']) for v in by_ansatz.values())
    if n_total == 0:
        print("[SKIP] obj1_dst_vs_delta_opt.png: no Stage 3 dst/delta_opt pairs")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for ansatz in STAGE3_ANSATZES:
        xs = by_ansatz[ansatz]['dst']
        ys = by_ansatz[ansatz]['delta_opt']
        if not xs:
            continue
        colour = ANSATZ_COLOURS[ansatz]
        corr = _pearson_corr(xs, ys)
        label = f"{ansatz} (n={len(xs)}"
        if corr is not None:
            label += f", r={corr:.3f}"
        label += ")"
        ax.scatter(xs, ys, c=colour, label=label, s=50, edgecolors='k', linewidths=0.3, zorder=3)
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=colour, linestyle='--',
                    linewidth=1.2, alpha=0.7)

    ax.set_xlabel('dst from default x_vec')
    ax.set_ylabel('delta_opt (surrogate)')
    ax.set_title(f'Stage 3 Comparison B — dst vs delta_opt (n={n_total})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, 'obj1_dst_vs_delta_opt.png')


def plot_hea_vs_nnhea_delta_opt(stage3_rows, output_dir):
    """Plot 2: horizontal dumbbell of HEA vs NN-HEA Stage 3 surrogate delta_opt."""
    hea = {}
    nn_hea = {}
    for r in stage3_rows:
        if r.get('status') != 'success':
            continue
        delta = parse_float(r.get('delta_opt'))
        if delta is None:
            continue
        prefix = r.get('ham_prefix', '').strip()
        if not prefix:
            continue
        if r.get('ansatz') == 'HEA':
            hea[prefix] = delta
        elif r.get('ansatz') == 'NN_HEA':
            nn_hea[prefix] = delta

    common = sorted(set(hea.keys()) & set(nn_hea.keys()))
    if not common:
        print("[SKIP] obj1_hea_vs_nnhea_delta_opt.png: no shared HEA/NN_HEA ham_prefix pairs")
        return

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(common) + 1.5)))
    y_pos = np.arange(len(common))
    for i, prefix in enumerate(common):
        ax.plot([hea[prefix], nn_hea[prefix]], [i, i], color='0.6', linewidth=1.5, zorder=1)
        ax.scatter([hea[prefix]], [i], c=ANSATZ_COLOURS['HEA'], s=60, zorder=2,
                   edgecolors='k', linewidths=0.3, label='HEA' if i == 0 else None)
        ax.scatter([nn_hea[prefix]], [i], c=ANSATZ_COLOURS['NN_HEA'], s=60, zorder=2,
                   edgecolors='k', linewidths=0.3, label='NN_HEA' if i == 0 else None)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(common)
    ax.set_xlabel('delta_opt (surrogate: ML-only → Full AccelerQ)')
    ax.set_title(
        f'Stage 3 HEA vs NN-HEA delta_opt (n={len(common)} paired Hamiltonians)\n'
        'Surrogate Comparison B — not Final VQE'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    _save_fig(fig, output_dir, 'obj1_hea_vs_nnhea_delta_opt.png')


def plot_training_delta_opt_by_ansatz(training_rows, output_dir):
    """Plot 3: strip + light box of Comparison A delta_opt by ansatz."""
    groups = {}
    excluded_by_ansatz = {}
    for ansatz in ('HEA', 'UCCSD'):
        vals = []
        n_excluded = 0
        for r in training_rows:
            if r.get('ansatz') != ansatz:
                continue
            if _is_near_zero_baseline_flag(r.get('E_baseline_near_zero')):
                n_excluded += 1
                continue
            delta = parse_float(r.get('delta_opt'))
            if delta is None:
                continue
            vals.append(delta)
        excluded_by_ansatz[ansatz] = n_excluded
        if vals:
            groups[ansatz] = vals

    if not groups and not any(excluded_by_ansatz.values()):
        print("[SKIP] obj1_training_delta_opt_by_ansatz.png: no Comparison A delta_opt values")
        return

    if not groups:
        # All points excluded — still write an annotated empty-axes figure.
        fig, ax = plt.subplots(figsize=(6, 5))
        notes = [
            f"{a}: {n} excluded (|E_baseline| < {NEAR_ZERO_BASELINE_THRESHOLD:g})"
            for a, n in excluded_by_ansatz.items() if n
        ]
        ax.text(0.5, 0.5, 'No defined delta_opt to plot\n' + '\n'.join(notes),
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        ax.set_title('Comparison A — training delta_opt by ansatz')
        _save_fig(fig, output_dir, 'obj1_training_delta_opt_by_ansatz.png')
        return

    labels = list(groups.keys())
    data = [groups[a] for a in labels]
    colours = [ANSATZ_COLOURS[a] for a in labels]
    n_total = sum(len(v) for v in data)

    fig, ax = plt.subplots(figsize=(6, 5))
    boxplot = ax.boxplot(
        data, positions=np.arange(len(labels)), widths=0.45,
        patch_artist=True, showfliers=False, zorder=1,
    )
    for patch, colour in zip(boxplot['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.25)
        patch.set_edgecolor(colour)

    rng = np.random.default_rng(42)
    for i, (ansatz, vals) in enumerate(groups.items()):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            c=ANSATZ_COLOURS[ansatz], s=36, edgecolors='k', linewidths=0.3,
            zorder=3, label=f"{ansatz} (n={len(vals)})",
        )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('delta_opt')
    ax.set_title(f'Comparison A — training delta_opt by ansatz (n={n_total})')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    excl_notes = [
        f"{a}: {n} excluded (|E_baseline| < {NEAR_ZERO_BASELINE_THRESHOLD:g})"
        for a, n in excluded_by_ansatz.items() if n
    ]
    if excl_notes:
        ax.text(
            0.02, 0.98, 'Near-zero baseline:\n' + '\n'.join(excl_notes),
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        )
    _save_fig(fig, output_dir, 'obj1_training_delta_opt_by_ansatz.png')


def _stagger_labels(ys, span, min_gap_frac=0.055):
    """Nudge near-coincident label y-positions apart, preserving order."""
    order = np.argsort(ys)
    staggered = np.array(ys, dtype=float)
    min_gap = min_gap_frac * span
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if staggered[j] - staggered[i] < min_gap:
            staggered[j] = staggered[i] + min_gap
    return staggered


def plot_baseline_vs_opt_energy(baseline_vs_opt_rows, output_dir):
    """Plot 4: two-panel energy dumbbell (HEA | NN_HEA) for Comparison D."""
    by_impl = {'HEA': [], 'NN_HEA': []}
    for r in baseline_vs_opt_rows:
        impl = r.get('implementation', '')
        if impl not in by_impl:
            continue
        e_bl = parse_float(r.get('E_baseline'))
        e_opt = parse_float(r.get('E_optimised'))
        prefix = r.get('ham_prefix', '').strip()
        if e_bl is None or e_opt is None or not prefix:
            continue
        by_impl[impl].append((prefix, e_bl, e_opt))

    panels = [(impl, rows) for impl, rows in by_impl.items() if rows]
    if not panels:
        print("[SKIP] obj1_baseline_vs_opt_energy.png: no Comparison D baseline/optimised pairs")
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.2 * len(panels), 5.6),
        sharey=False, squeeze=False,
    )

    for col, (impl, rows) in enumerate(panels):
        ax = axes[0, col]
        rows = sorted(rows, key=lambda t: t[0])
        colour = ANSATZ_COLOURS[impl]
        e_bl = np.array([t[1] for t in rows], dtype=float)
        e_opt = np.array([t[2] for t in rows], dtype=float)
        span = max(float(np.ptp(np.concatenate([e_bl, e_opt]))), 1e-9)
        label_y = _stagger_labels(e_bl, span)

        for i, (prefix, _, _) in enumerate(rows):
            ax.plot([0, 1], [e_bl[i], e_opt[i]], color='0.6', linewidth=1.5, zorder=1)
            ax.scatter([0], [e_bl[i]], c=colour, s=60, edgecolors='k', linewidths=0.3,
                       zorder=3, marker='o', label='Baseline' if i == 0 else None)
            ax.scatter([1], [e_opt[i]], c=colour, s=60, edgecolors='k', linewidths=0.3,
                       zorder=3, marker='s', label='Optimised' if i == 0 else None)
            # label lives INSIDE the axes, in a reserved left gutter
            ax.annotate(prefix, xy=(0, e_bl[i]), xytext=(-0.10, label_y[i]),
                        textcoords='data', ha='right', va='center',
                        fontsize=9, color='0.30',
                        arrowprops=dict(arrowstyle='-', color='0.75',
                                        linewidth=0.6, shrinkA=0, shrinkB=3))

        ax.set_xlim(-0.62, 1.18)          # <- the gutter; stops all bleed
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Baseline', 'Optimised'], fontsize=11)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylabel('VQE energy', fontsize=12)
        ax.set_title(f'{impl} (n={len(rows)})', fontsize=13)
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.09),
                  ncol=2, frameon=False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Comparison D — real VQE baseline vs optimised energy\n'
        'Shared iter_max=500; HEA/NN-HEA share the physical baseline run',
        fontsize=13, y=1.00,
    )
    fig.subplots_adjust(wspace=0.34)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, output_dir, 'obj1_baseline_vs_opt_energy.png')


def _thousands(x, _pos):
    return f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:g}"


def _format_feature_axis(ax, feat):
    if feat == 'term_count':
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        f = ScalarFormatter(useOffset=False)
        f.set_scientific(False)
        ax.xaxis.set_major_formatter(f)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha('right')


def _plot_structure_scatter(rows, y_key, y_label, title, filename, output_dir):
    """1x3 HEA/NN-HEA structure scatters; colour=impl, marker=ham (no point labels)."""
    points = []
    for r in rows:
        y = parse_float(r.get(y_key))
        if y is None:
            continue
        impl = (r.get('implementation') or '').strip()
        if impl not in ('HEA', 'NN_HEA'):
            continue
        prefix = (r.get('ham_prefix') or '').strip()
        feat_vals = {name: parse_float(r.get(name)) for name in HAM_FEATURE_NAMES}
        if any(v is None for v in feat_vals.values()):
            continue
        points.append((impl, prefix, feat_vals, y))

    if not points:
        print(f"[SKIP] {filename}: no enriched rows with features and {y_key}")
        return

    hams = sorted({p[1] for p in points})
    marker_of = {h: HAM_MARKERS[i % len(HAM_MARKERS)] for i, h in enumerate(hams)}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)

    for ax, feat_name in zip(axes, HAM_FEATURE_NAMES):
        corr_lines = []
        for impl in ('HEA', 'NN_HEA'):
            xs, ys = [], []
            for imp, prefix, feat_vals, y in points:
                if imp != impl:
                    continue
                xs.append(feat_vals[feat_name])
                ys.append(y)
                ax.scatter(feat_vals[feat_name], y, c=ANSATZ_COLOURS[impl],
                           marker=marker_of[prefix], s=95, edgecolors='k',
                           linewidths=0.5, alpha=0.85, zorder=3)
            if len(xs) >= 2:
                corr = _pearson_corr(xs, ys)
                if corr is not None:
                    corr_lines.append(
                        f"{impl.replace('_', '-')}: n={len(xs)}, r={corr:.3f}")

        ax.set_xlabel(feat_name, fontsize=14)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(x=0.14, y=0.16)
        _format_feature_axis(ax, feat_name)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        if corr_lines:
            ax.annotate('\n'.join(corr_lines), xy=(0.02, 0.98),
                        xycoords='axes fraction', va='top', ha='left',
                        fontsize=10, zorder=6,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.92, edgecolor='0.7'))

    axes[0].set_ylabel(y_label, fontsize=14)

    impl_handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=9,
               markerfacecolor=ANSATZ_COLOURS[i], markeredgecolor='k',
               label=i.replace('_', '-'))
        for i in ('HEA', 'NN_HEA')
    ]
    ham_handles = [
        Line2D([], [], marker=marker_of[h], linestyle='none', markersize=9,
               markerfacecolor='0.65', markeredgecolor='k', label=h)
        for h in hams
    ]
    leg1 = fig.legend(handles=impl_handles, title='implementation (colour)',
                      loc='upper left', bbox_to_anchor=(0.06, 0.02), ncol=2,
                      fontsize=12, title_fontsize=12, frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=ham_handles, title='Hamiltonian (marker)',
               loc='upper left', bbox_to_anchor=(0.34, 0.02), ncol=len(hams),
               fontsize=12, title_fontsize=12, frameon=False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _save_fig(fig, output_dir, filename)


def plot_structure_vs_abs_error(final_vqe_rows, output_dir):
    """Comparison C — structure features vs abs_error (labelled, descriptive only)."""
    _plot_structure_scatter(
        final_vqe_rows,
        y_key='abs_error',
        y_label='abs_error (|VQE − reference|)',
        title=(
            'Comparison C — Hamiltonian structure vs abs_error\n'
            '(descriptive / small-n; no significance claim)'
        ),
        filename='obj1_structure_vs_abs_error.png',
        output_dir=output_dir,
    )


def plot_structure_vs_delta_opt(baseline_vs_opt_rows, output_dir):
    """Comparison D — structure features vs delta_opt (labelled, descriptive only)."""
    _plot_structure_scatter(
        baseline_vs_opt_rows,
        y_key='delta_opt',
        y_label='delta_opt (baseline → optimised)',
        title=(
            'Comparison D — Hamiltonian structure vs delta_opt\n'
            '(descriptive / small-n; no significance claim)'
        ),
        filename='obj1_structure_vs_delta_opt.png',
        output_dir=output_dir,
    )


def write_obj1_plots(
    training_rows, stage3_rows, baseline_vs_opt_rows, output_dir,
    final_vqe_rows=None,
):
    """Write Objective 1 figures matched to sample size (scatter/dumbbell vs strip/box)."""
    os.makedirs(output_dir, exist_ok=True)
    plot_dst_vs_delta_opt(stage3_rows, output_dir)
    plot_hea_vs_nnhea_delta_opt(stage3_rows, output_dir)
    plot_training_delta_opt_by_ansatz(training_rows, output_dir)
    plot_baseline_vs_opt_energy(baseline_vs_opt_rows, output_dir)
    if final_vqe_rows is not None:
        plot_structure_vs_abs_error(final_vqe_rows, output_dir)
    plot_structure_vs_delta_opt(baseline_vs_opt_rows, output_dir)


def _reference_type_display_label(ref_type):
    """Short print label only; grouping always uses the raw reference_type string."""
    lower = (ref_type or '').strip().lower()
    if 'exact' in lower:
        return 'exact'
    if 'best_guess' in lower:
        return 'best_guess'
    return (ref_type or '').strip() or 'unknown'


def comparison_c_by_reference_type(final_vqe_rows):
    """Comparison C abs_error stats by raw reference_type; returns (csv_rows, summary_lines)."""
    csv_rows = []
    summary_lines = []
    for impl in ('HEA', 'NN_HEA'):
        impl_rows = [
            r for r in final_vqe_rows
            if r.get('implementation') == impl and r.get('abs_error', '') != ''
        ]
        abs_errors = [float(r['abs_error']) for r in impl_rows]
        if not abs_errors:
            continue

        mean_p = float(np.mean(abs_errors))
        std_p = float(np.std(abs_errors))
        csv_rows.append({
            'implementation': impl,
            'reference_type': 'pooled',
            'n': len(abs_errors),
            'mean_abs_error': fmt(mean_p, precision=6),
            'std_abs_error': fmt(std_p, precision=6),
        })
        summary_lines.append(
            f"    {impl}: mean abs_error = {mean_p:.6f}"
            f" +/- {std_p:.6f}"
            f"  (n={len(abs_errors)}, all reference types, pooled)"
        )

        by_type = defaultdict(list)
        for r in impl_rows:
            ref_type = (r.get('reference_type') or '').strip()
            if ref_type:
                by_type[ref_type].append(float(r['abs_error']))

        for ref_type in sorted(by_type.keys()):
            vals = by_type[ref_type]
            mean_t = float(np.mean(vals))
            std_t = float(np.std(vals))
            csv_rows.append({
                'implementation': impl,
                'reference_type': ref_type,
                'n': len(vals),
                'mean_abs_error': fmt(mean_t, precision=6),
                'std_abs_error': fmt(std_t, precision=6),
            })
            label = _reference_type_display_label(ref_type)
            summary_lines.append(
                f"    {impl}: mean abs_error = {mean_t:.6f}"
                f" +/- {std_t:.6f}"
                f"  (n={len(vals)}, {label} reference only)"
            )
    return csv_rows, summary_lines


def comparison_d_summary_stats(baseline_vs_opt_rows):
    """Comparison D delta_opt summary + 3σ flags (outliers kept); returns (csv_rows, summary_lines)."""
    csv_rows = []
    summary_lines = []
    for impl in ('HEA', 'NN_HEA'):
        pairs = []
        for r in baseline_vs_opt_rows:
            if r.get('implementation') != impl:
                continue
            delta = parse_float(r.get('delta_opt'))
            if delta is None:
                continue
            pairs.append((r.get('ham_prefix', '').strip(), delta))
        if not pairs:
            continue

        deltas = [d for _, d in pairs]
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        median_d = float(np.median(deltas))
        outliers = []
        if std_d > 0:
            for prefix, d in pairs:
                if abs(d - mean_d) > 3.0 * std_d:
                    outliers.append(prefix)

        csv_rows.append({
            'implementation': impl,
            'n': len(deltas),
            'mean_delta_opt': fmt(mean_d, precision=4),
            'std_delta_opt': fmt(std_d, precision=4),
            'median_delta_opt': fmt(median_d, precision=4),
            'outlier_ham_prefixes': ';'.join(outliers),
        })
        line = (
            f"    {impl}: mean delta_opt = {mean_d:.4f}"
            f" +/- {std_d:.4f}"
            f"  (n={len(deltas)}); median delta_opt = {median_d:.4f}"
        )
        if outliers:
            line += f"; candidate outliers (>3 std): {', '.join(outliers)}"
        summary_lines.append(line)
    return csv_rows, summary_lines


def write_summary(training_rows, stage3_rows, path, final_vqe_rows=None, baseline_vs_opt_rows=None):
    if final_vqe_rows is None:
        final_vqe_rows = []
    if baseline_vs_opt_rows is None:
        baseline_vs_opt_rows = []
    lines = []
    comparison_c_csv_rows = []
    comparison_d_csv_rows = []

    lines.append("=" * 70)
    lines.append("Objective 1 — Generalisation Across Ansatz Classes")
    lines.append("=" * 70)
    lines.append("")
    lines.append(SUMMARY_NOTE_TRAINING)

    lines.append("--- Comparison A: Training Hamiltonians ---")
    lines.append("")
    current_ansatz = None
    for r in training_rows:
        if r['ansatz'] != current_ansatz:
            current_ansatz = r['ansatz']
            lines.append(f"  Ansatz: {current_ansatz}")
        lines.append(
            f"    ham_id={r['ham_id']} | {r['n_qubits']} qubits | {r['dataset']}"
        )
        lines.append(
            f"      Baseline energy:   {r['E_baseline_mean']} +/- {r['E_baseline_std']}"
            f"  (n={r['n_baseline_records']})"
        )
        lines.append(
            f"      ML-Only pred min:  {r['E_pred_min']}"
            f"  (from {r['n_test_records']} test records)"
        )
        lines.append(f"      delta_opt:         {r['delta_opt']}")
        lines.append(f"      Model MSE (test):  {r['model_mse_mean']}")
        lines.append("")

    # Absolute-energy view for near-zero-baseline training rows
    for ansatz in ('HEA', 'UCCSD'):
        near_zero_rows = [
            r for r in training_rows
            if r.get('ansatz') == ansatz
            and _is_near_zero_baseline_flag(r.get('E_baseline_near_zero'))
        ]
        if not near_zero_rows:
            continue
        lines.append(
            f"  Comparison A — near-zero-baseline Hamiltonians ({ansatz}): "
            f"reported as absolute energies, not delta_opt, since "
            f"|E_baseline| < {NEAR_ZERO_BASELINE_THRESHOLD:g} makes the ratio undefined."
        )
        for r in near_zero_rows:
            lines.append(
                f"    ham_id={r['ham_id']}: E_baseline_mean={r['E_baseline_mean']},"
                f" E_pred_min={r['E_pred_min']}"
            )
        lines.append("")

    lines.append("")
    lines.append("=" * 70)
    lines.append(SUMMARY_NOTE_STAGE3)

    lines.append("--- Comparison B: Stage 3 Hamiltonians ---")
    lines.append("")
    current_ansatz = None
    for r in stage3_rows:
        if r.get('status') != 'success':
            continue
        if r['ansatz'] != current_ansatz:
            current_ansatz = r['ansatz']
            lines.append(f"  Ansatz: {current_ansatz}")
        lines.append(f"    {r['ham_prefix']} ({r['n_qubits']} qubits)")
        lines.append(f"      ML-Only score:        {r['score_ml_only']}")
        lines.append(f"      Full AccelerQ score:  {r['score_full_accelerq']}")
        lines.append(f"      delta_opt:            {r['delta_opt']}")
        lines.append(f"      dst from default:     {r['dst']}")
        lines.append("")

    if final_vqe_rows:
        lines.append("")
        lines.append("=" * 70)
        lines.append(SUMMARY_NOTE_FINAL_VQE)
        lines.append("--- Comparison C: Final VQE vs reference ---")
        lines.append("")
        current_impl = None
        for r in final_vqe_rows:
            if r['implementation'] != current_impl:
                current_impl = r['implementation']
                lines.append(f"  Implementation: {current_impl}")
            lines.append(f"    {r['ham_prefix']} ({r['n_qubits']} qubits, {r['dataset']})")
            lines.append(f"      VQE energy:         {r['vqe_energy']}")
            lines.append(f"      Reference energy:   {r['reference_energy']} ({r['reference_type']})")
            lines.append(f"      abs_error:          {r['abs_error']}")
            lines.append(f"      signed_error:       {r['signed_error']}")
            lines.append(
                f"      iterations:         {r['n_iterations']}/{r['iter_max']}"
                f"  converged={r['converged']}"
            )
            lines.append("")

        lines.append("  Comparison C cross-implementation summary:")
        comparison_c_csv_rows, c_lines = comparison_c_by_reference_type(final_vqe_rows)
        lines.extend(c_lines)

    if baseline_vs_opt_rows:
        lines.append("")
        lines.append("=" * 70)
        lines.append(SUMMARY_NOTE_BASELINE_VS_OPT)
        lines.append("--- Comparison D: Baseline vs optimised Final VQE ---")
        lines.append("")
        current_impl = None
        for r in baseline_vs_opt_rows:
            if r['implementation'] != current_impl:
                current_impl = r['implementation']
                lines.append(f"  Implementation: {current_impl}")
            lines.append(f"    {r['ham_prefix']} ({r['n_qubits']} qubits)")
            lines.append(f"      E_baseline:          {r['E_baseline']}")
            lines.append(f"      E_optimised:         {r['E_optimised']}")
            lines.append(f"      delta_opt:           {r['delta_opt']}")
            lines.append(
                f"      iters baseline/opt:  {r['n_iterations_baseline']}/"
                f"{r['iter_max_baseline']}  vs  {r['n_iterations_optimised']}/"
                f"{r['iter_max_optimised']}"
            )
            lines.append(
                f"      run_time_s baseline/opt:  {r['run_time_s_baseline']}"
                f"  vs  {r['run_time_s_optimised']}"
            )
            lines.append("")

        lines.append("  Comparison D cross-implementation summary:")
        comparison_d_csv_rows, d_lines = comparison_d_summary_stats(baseline_vs_opt_rows)
        lines.extend(d_lines)

    lines.append("")
    lines.append("=" * 70)
    lines.append("Cross-ansatz delta_opt summary (Layer 1)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Comparison A (training Hamiltonians):")
    training_deltas_by_ansatz = {}
    for ansatz in ('HEA', 'UCCSD'):
        deltas = []
        n_near_zero = 0
        for r in training_rows:
            if r.get('ansatz') != ansatz:
                continue
            if _is_near_zero_baseline_flag(r.get('E_baseline_near_zero')):
                n_near_zero += 1
                continue
            delta = parse_float(r.get('delta_opt'))
            if delta is not None:
                deltas.append(delta)
        training_deltas_by_ansatz[ansatz] = deltas
        if deltas or n_near_zero:
            if deltas:
                lines.append(
                    f"    {ansatz}: mean delta_opt = {np.mean(deltas):.4f}"
                    f" +/- {np.std(deltas):.4f}"
                    f"  (n Hamiltonians = {len(deltas)},"
                    f" {n_near_zero}_near_zero_baseline_excluded)"
                )
            else:
                lines.append(
                    f"    {ansatz}: mean delta_opt = undefined"
                    f"  (n Hamiltonians = 0,"
                    f" {n_near_zero}_near_zero_baseline_excluded)"
                )

    lines.append("")
    lines.append("  Comparison B (stage 3 Hamiltonians):")
    stage3_deltas_by_ansatz = {}
    stage3_dsts_by_ansatz = {}
    for ansatz in STAGE3_ANSATZES:
        deltas = [
            float(r['delta_opt'])
            for r in stage3_rows
            if r['ansatz'] == ansatz
            and r['delta_opt'] != ''
            and r.get('status') == 'success'
        ]
        dsts = [
            float(r['dst'])
            for r in stage3_rows
            if r['ansatz'] == ansatz
            and r['dst'] != ''
            and r.get('status') == 'success'
        ]
        stage3_deltas_by_ansatz[ansatz] = deltas
        stage3_dsts_by_ansatz[ansatz] = dsts
        if deltas:
            lines.append(
                f"    {ansatz}: mean delta_opt = {np.mean(deltas):.4f}"
                f" +/- {np.std(deltas):.4f}"
                f"  (n Hamiltonians = {len(deltas)})"
            )
        if dsts:
            lines.append(
                f"    {ansatz}: mean dst       = {np.mean(dsts):.4f}"
                f" +/- {np.std(dsts):.4f}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("Layer 2 — variance and dst vs delta_opt (stage 3)")
    lines.append("=" * 70)
    lines.append("")
    for ansatz in STAGE3_ANSATZES:
        deltas = stage3_deltas_by_ansatz.get(ansatz, [])
        dsts = stage3_dsts_by_ansatz.get(ansatz, [])
        if len(deltas) >= 2:
            lines.append(f"  {ansatz}: std(delta_opt) = {np.std(deltas):.4f}  (n={len(deltas)})")
        paired_dsts = []
        paired_deltas = []
        for r in stage3_rows:
            if r['ansatz'] != ansatz or r.get('status') != 'success':
                continue
            delta = parse_float(r.get('delta_opt'))
            dst = parse_float(r.get('dst'))
            if delta is not None and dst is not None:
                paired_deltas.append(delta)
                paired_dsts.append(dst)
        corr = _pearson_corr(paired_dsts, paired_deltas)
        if corr is not None:
            lines.append(
                f"  {ansatz}: Pearson corr(dst, delta_opt) = {corr:.4f}  (n={len(paired_dsts)})"
            )
        elif len(paired_dsts) >= 1:
            lines.append(f"  {ansatz}: Pearson corr(dst, delta_opt) — insufficient variance (n={len(paired_dsts)})")

    hea_std = np.std(training_deltas_by_ansatz.get('HEA', [])) if training_deltas_by_ansatz.get('HEA') else None
    uccsd_std = np.std(training_deltas_by_ansatz.get('UCCSD', [])) if training_deltas_by_ansatz.get('UCCSD') else None
    if hea_std is not None and uccsd_std is not None:
        lines.append("")
        lines.append("  Comparison A cross-ansatz variance:")
        lines.append(f"    HEA std(delta_opt)   = {hea_std:.4f}")
        lines.append(f"    UCCSD std(delta_opt) = {uccsd_std:.4f}")
    elif hea_std is not None:
        lines.append("")
        lines.append("  Comparison A cross-ansatz variance:")
        lines.append(f"    HEA std(delta_opt)   = {hea_std:.4f}")
        lines.append("    UCCSD std(delta_opt) = undefined (no valid delta_opt)")

    lines.append("")
    lines.append(
        "  Note: Comparison A and B cover different Hamiltonian scales and\n"
        "  different reference points (true VQE vs surrogate). Do not\n"
        "  interpret the two delta_opt values as directly comparable.\n"
        "  Cross-ansatz comparisons within each comparison are indicative only:\n"
        "  HEA and UCCSD differ in circuit depth, GA budget, and Hamiltonian coverage."
    )

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {path}")
    return comparison_c_csv_rows, comparison_d_csv_rows


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_rows = []
    for ansatz in ('HEA', 'UCCSD'):
        data_dir = IMPL_DATA[ansatz]
        records_path = os.path.join(data_dir, f'records_{ansatz}.csv')
        ham_path = os.path.join(data_dir, f'hamiltonian_{ansatz}.csv')
        print(f"Analysing training Hamiltonians: {ansatz}")
        training_rows.extend(
            analyse_training_hams(records_path, ham_path, ansatz, data_dir)
        )

    stage3_sources = [
        (os.path.join(IMPL_DATA[ansatz], f'energies_stage3_{ansatz}.csv'), ansatz)
        for ansatz in STAGE3_ANSATZES
    ]
    print("Analysing Stage 3 Hamiltonians")
    stage3_rows = analyse_stage3_hams(stage3_sources)

    print("Loading Stage 3 Hamiltonian structural features")
    ham_prefixes = collect_stage3_ham_prefixes(FINAL_VQE_CSV, stage3_rows)
    ham_features = load_stage3_ham_features(ham_prefixes)
    _attach_ham_features(stage3_rows, ham_features)

    print("Analysing Final VQE vs reference energies")
    final_vqe_rows = analyse_final_vqe(FINAL_VQE_CSV, REFERENCE_ENERGIES_CSV)
    _attach_ham_features(final_vqe_rows, ham_features)

    print("Analysing baseline vs optimised Final VQE")
    baseline_vs_opt_rows = analyse_baseline_vs_optimised(FINAL_VQE_CSV)
    _attach_ham_features(baseline_vs_opt_rows, ham_features)

    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_training_hams.csv'),
        TRAINING_FIELDS,
        training_rows,
    )
    near_zero_rows = [
        {
            'ansatz': r['ansatz'],
            'ham_id': r['ham_id'],
            'n_qubits': r['n_qubits'],
            'E_baseline_mean': r['E_baseline_mean'],
            'E_pred_min': r['E_pred_min'],
            'E_pred_mean': r['E_pred_mean'],
        }
        for r in training_rows
        if _is_near_zero_baseline_flag(r.get('E_baseline_near_zero'))
    ]
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_training_hams_near_zero_baseline.csv'),
        NEAR_ZERO_BASELINE_FIELDS,
        near_zero_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_stage3_hams.csv'),
        STAGE3_FIELDS,
        stage3_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_final_vqe.csv'),
        FINAL_VQE_FIELDS,
        final_vqe_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_baseline_vs_opt.csv'),
        BASELINE_VS_OPT_FIELDS,
        baseline_vs_opt_rows,
    )

    comparison_c_csv_rows, comparison_d_csv_rows = write_summary(
        training_rows, stage3_rows,
        os.path.join(OUTPUT_DIR, 'obj1_summary.txt'),
        final_vqe_rows, baseline_vs_opt_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_comparisonC_by_reference_type.csv'),
        COMPARISON_C_BY_REF_FIELDS,
        comparison_c_csv_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, 'obj1_comparisonD_summary_stats.csv'),
        COMPARISON_D_SUMMARY_FIELDS,
        comparison_d_csv_rows,
    )
    write_obj1_plots(
        training_rows, stage3_rows, baseline_vs_opt_rows, OUTPUT_DIR,
        final_vqe_rows=final_vqe_rows,
    )
