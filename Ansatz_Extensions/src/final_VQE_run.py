"""
Execute real VQE on Stage 3 Hamiltonians for HEA and NN-HEA.

Two arms land in the same Final_VQE.csv (tagged with is_baseline):
  - Baseline: default_hea_x_vec (iter_max=500), run once per Hamiltonian and
    reused for both HEA and NN_HEA rows (shared ansatz / default vector).
  - Optimised: GA optimal_x_vec from energies_stage3_{HEA,NN_HEA}.csv.

Partial runs upsert by (ham_prefix, implementation, is_baseline) so
--mode baseline does not wipe optimised rows (and vice versa).

Run from src/:
    python3 final_VQE_run.py
    python3 final_VQE_run.py --mode baseline --ham-prefix 24qubits_05
    python3 final_VQE_run.py --mode optimised --impl HEA

Or via the Artifact_Experiments wrapper (from repo root):
    ./Artifact_Experiments/final_VQE_run.sh [--mode ...] [--ham-prefix ...] [--impl ...]
"""

import argparse
import csv
import json
import os
import re
import time

import numpy as np
from openfermion import FermionOperator as FO

from kcl_HEA_VQE import NN_SEED, Wrapper_HEA
from kcl_util import process_file
from kcl_util_HEA import default_hea_x_vec, sanitize_x_vec_params
from kcl_util_hamiltonian import REPO_HAMILTONIAN_DIR
from kcl_util_NN_HEA import extract_ham_features_for_nn, sanitize_x_vec_params_nn_hea

HAM_DIR = REPO_HAMILTONIAN_DIR
OUTPUT_DIR = os.path.join("..", "data", "Final_VQE_run")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Final_VQE.csv")

STAGE3_SOURCES = {
    "HEA": os.path.join("..", "data", "HEA", "energies_stage3_HEA.csv"),
    "NN_HEA": os.path.join("..", "data", "NN_HEA", "energies_stage3_NN_HEA.csv"),
}

FINAL_VQE_FIELDS = [
    "ham_prefix",
    "n_qubits",
    "implementation",
    "is_baseline",
    "vqe_energy",
    "n_iterations",
    "iter_max",
    "converged",
    "status",
    "error_message",
    "run_time_s",
]

_PREFIX_RE = re.compile(r"^(\d+)qubits_(\d+)$")


def parse_ham_seed(ham_prefix):
    match = _PREFIX_RE.match(ham_prefix.strip())
    if not match:
        raise ValueError(f"Cannot parse ham_prefix: {ham_prefix!r}")
    return int(match.group(2))


def _coerce_is_baseline(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes")


def row_key(row):
    return (
        str(row.get("ham_prefix", "")).strip(),
        str(row.get("implementation", "")).strip(),
        _coerce_is_baseline(row.get("is_baseline", False)),
    )


def load_last_success_rows(csv_path):
    """Return dict ham_prefix -> last successful row (preserves CSV order)."""
    if not os.path.exists(csv_path):
        print(f"  Warning: {csv_path} not found — skipping.")
        return {}

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        prefix = row.get("ham_prefix", "").strip()
        if prefix:
            out[prefix] = row
    return out


def parse_optimal_x_vec(row):
    raw = row.get("optimal_x_vec", "")
    if not raw or not str(raw).strip():
        return None
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) < 14:
        raise ValueError(f"optimal_x_vec must be JSON list of length >= 14, got {data!r}")
    return np.array(data[:14], dtype=float)


def build_hea_wrapper(x, n_qubits):
    return Wrapper_HEA(
        n_qubits=n_qubits,
        ham=None,
        is_classical=True,
        reps=int(round(x[2])),
        entanglement=int(round(x[3])),
        rotation_blocks=int(round(x[4])),
        skip_final_rot=int(round(x[5])),
        optimizer_choice=int(round(x[6])),
        iter_max=int(round(x[7])),
        coeff_cutoff=float(x[8]),
        atol=float(x[9]),
        shots=int(round(x[10])),
        gradient_method=int(round(x[11])),
        initial_point_seed=int(round(x[12])),
        max_evals_grouped=int(round(x[13])),
    )


def run_hea_vqe(x_vec, n_qubits, ham_seed):
    x = sanitize_x_vec_params(x_vec)
    wrapper = build_hea_wrapper(x, n_qubits)
    wrapper.get_result(seed=ham_seed, ham_dir=HAM_DIR)
    result = wrapper.result_for_evaluation()
    iter_max = int(round(x[7]))
    return {
        "vqe_energy": float(result.energy),
        "n_iterations": int(result.n_iterations),
        "iter_max": iter_max,
        "converged": result.n_iterations < iter_max,
    }


def run_nn_hea_vqe(x_vec, n_qubits, ham_seed, ham_prefix):
    x = sanitize_x_vec_params_nn_hea(x_vec)
    nn_ham_features = None
    if int(round(x[12])) == NN_SEED:
        file_name = ham_prefix + ".data"
        result = process_file(HAM_DIR, file_name)
        ham = result[1]
        if not isinstance(ham, FO):
            raise TypeError(f"Expected FermionOperator for {ham_prefix}")
        nn_ham_features = extract_ham_features_for_nn(ham)

    wrapper = build_hea_wrapper(x, n_qubits)
    wrapper.get_result(
        seed=ham_seed,
        ham_dir=HAM_DIR,
        nn_ham_features=nn_ham_features,
    )
    result = wrapper.result_for_evaluation()
    iter_max = int(round(x[7]))
    return {
        "vqe_energy": float(result.energy),
        "n_iterations": int(result.n_iterations),
        "iter_max": iter_max,
        "converged": result.n_iterations < iter_max,
    }


def collect_stage3_by_impl(ham_prefix_filter=None, impl_filter=None):
    """impl -> {ham_prefix: stage3_row} for filtered Stage 3 success rows."""
    by_impl = {}
    for impl, csv_path in STAGE3_SOURCES.items():
        if impl_filter and impl != impl_filter:
            continue
        rows = load_last_success_rows(csv_path)
        filtered = {}
        for prefix, row in rows.items():
            if ham_prefix_filter and prefix != ham_prefix_filter:
                continue
            filtered[prefix] = row
        by_impl[impl] = filtered
    return by_impl


def collect_optimised_jobs(by_impl):
    jobs = []
    for impl, rows in by_impl.items():
        for prefix, row in sorted(rows.items()):
            jobs.append((impl, prefix, row))
    return jobs


def collect_baseline_hams(by_impl):
    """Map unique ham_prefix → (n_qubits, impls); baseline runs once per prefix."""
    hams = {}
    for impl, rows in by_impl.items():
        for prefix, row in rows.items():
            n_qubits = int(row["n_qubits"])
            if prefix not in hams:
                hams[prefix] = {"n_qubits": n_qubits, "impls": []}
            if impl not in hams[prefix]["impls"]:
                hams[prefix]["impls"].append(impl)
    return hams


def _success_row(prefix, n_qubits, impl, is_baseline, metrics, elapsed):
    return {
        "ham_prefix": prefix,
        "n_qubits": n_qubits,
        "implementation": impl,
        "is_baseline": is_baseline,
        "vqe_energy": metrics["vqe_energy"],
        "n_iterations": metrics["n_iterations"],
        "iter_max": metrics["iter_max"],
        "converged": metrics["converged"],
        "status": "success",
        "error_message": "",
        "run_time_s": elapsed,
    }


def _failed_row(prefix, n_qubits, impl, is_baseline, error_message, status="failed"):
    return {
        "ham_prefix": prefix,
        "n_qubits": n_qubits,
        "implementation": impl,
        "is_baseline": is_baseline,
        "status": status,
        "error_message": error_message,
    }


def load_existing_results(output_path):
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return {}
    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        # Normalise is_baseline for older rows that lack the column
        if "is_baseline" not in row or row.get("is_baseline", "") == "":
            row = dict(row)
            row["is_baseline"] = False
        out[row_key(row)] = row
    return out


def upsert_results(new_rows, output_path):
    """Merge new_rows into existing CSV by (ham_prefix, implementation, is_baseline)."""
    existing = load_existing_results(output_path)
    for row in new_rows:
        existing[row_key(row)] = row

    merged = sorted(
        existing.values(),
        key=lambda r: (
            str(r.get("ham_prefix", "")),
            str(r.get("implementation", "")),
            _coerce_is_baseline(r.get("is_baseline", False)),
        ),
    )
    # Normalise boolean for CSV
    for row in merged:
        row["is_baseline"] = _coerce_is_baseline(row.get("is_baseline", False))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_VQE_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"Written: {output_path} ({len(merged)} rows after upsert)")


def run_baseline_arm(by_impl):
    """Run shared HEA default baseline once per ham; emit one row per impl."""
    hams = collect_baseline_hams(by_impl)
    results = []
    for prefix in sorted(hams.keys()):
        info = hams[prefix]
        n_qubits = info["n_qubits"]
        impls = info["impls"]
        print(f"\n>> BASELINE (shared HEA default) | {prefix} ({n_qubits} qubits) -> {impls}")
        try:
            x_vec = default_hea_x_vec(n_qubits)
            ham_seed = parse_ham_seed(prefix)
            t0 = time.perf_counter()
            metrics = run_hea_vqe(x_vec, n_qubits, ham_seed)
            elapsed = time.perf_counter() - t0
            print(
                f"  energy={metrics['vqe_energy']:.10f}  "
                f"iters={metrics['n_iterations']}/{metrics['iter_max']}  "
                f"converged={metrics['converged']}  time={elapsed:.1f}s"
            )
            for impl in impls:
                row = _success_row(prefix, n_qubits, impl, True, metrics, elapsed)
                results.append(row)
                upsert_results([row], OUTPUT_CSV)          # <-- write immediately
        except Exception as exc:
            print(f"  FAILED: {exc}")
            for impl in impls:
                row = _failed_row(prefix, n_qubits, impl, True, str(exc))
                results.append(row)
                upsert_results([row], OUTPUT_CSV)          # <-- write immediately
    return results


def run_optimised_arm(by_impl):
    results = []
    for impl, prefix, row in collect_optimised_jobs(by_impl):
        n_qubits = int(row["n_qubits"])
        print(f"\n>> OPTIMISED | {impl} | {prefix} ({n_qubits} qubits)")
        try:
            x_vec = parse_optimal_x_vec(row)
            if x_vec is None:
                print(f"  SKIP: missing optimal_x_vec — re-run Stage 3 for {prefix}")
                skip_row = _failed_row(
                    prefix, n_qubits, impl, False,
                    "missing optimal_x_vec", status="skipped",
                )
                results.append(skip_row)
                upsert_results([skip_row], OUTPUT_CSV)     # <-- write immediately
                continue

            ham_seed = parse_ham_seed(prefix)
            t0 = time.perf_counter()
            if impl == "HEA":
                metrics = run_hea_vqe(x_vec, n_qubits, ham_seed)
            else:
                metrics = run_nn_hea_vqe(x_vec, n_qubits, ham_seed, prefix)
            elapsed = time.perf_counter() - t0

            print(
                f"  energy={metrics['vqe_energy']:.10f}  "
                f"iters={metrics['n_iterations']}/{metrics['iter_max']}  "
                f"converged={metrics['converged']}  time={elapsed:.1f}s"
            )
            success_row = _success_row(prefix, n_qubits, impl, False, metrics, elapsed)
            results.append(success_row)
            upsert_results([success_row], OUTPUT_CSV)      # <-- write immediately
        except Exception as exc:
            print(f"  FAILED: {exc}")
            fail_row = _failed_row(prefix, n_qubits, impl, False, str(exc))
            results.append(fail_row)
            upsert_results([fail_row], OUTPUT_CSV)         # <-- write immediately
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run final VQE (baseline default_hea_x_vec and/or GA-optimal)"
    )
    parser.add_argument("--ham-prefix", help="Run only this ham_prefix (e.g. 24qubits_05)")
    parser.add_argument("--impl", choices=["HEA", "NN_HEA"], help="Run only this implementation")
    parser.add_argument(
        "--mode",
        choices=["all", "baseline", "optimised"],
        default="all",
        help="Which arm(s) to run (default: all). Partial modes upsert into Final_VQE.csv.",
    )
    args = parser.parse_args()

    by_impl = collect_stage3_by_impl(
        ham_prefix_filter=args.ham_prefix, impl_filter=args.impl,
    )
    if not any(by_impl.values()):
        print("No Stage 3 jobs found matching filters.")
        return 1

    results = []
    if args.mode in ("all", "baseline"):
        results.extend(run_baseline_arm(by_impl))
    if args.mode in ("all", "optimised"):
        results.extend(run_optimised_arm(by_impl))

    if not results:
        print("No runs produced.")
        return 1

    upsert_results(results, OUTPUT_CSV)
    n_ok = sum(1 for r in results if r.get("status") == "success")
    print(f"\nDone: {n_ok}/{len(results)} successful runs this invocation.")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
