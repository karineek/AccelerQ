
"""
Stage 3 GA optimisation for NN-HEA with NN warm-start.

Uses the shared HEA XGBoost surrogate (model_hea_pre_xgb_*.json). NN-HEA differs
from HEA only in GA warm-start via the trained NN (nn_generator, NN_SEED).

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import csv
import json
import os

import numpy as np
import xgboost as xgb
from openfermion import FermionOperator as FO
from scipy.spatial import distance

from kcl_opt_xgb import opt_hyperparams
from kcl_tests_NN_HEA import test_semi_dynamic_nn_hea, test_static_nn_hea
from kcl_util import ham_to_vector, load_model, process_file
from kcl_util_NN_HEA import (
  NN_SEED,
  default_hea_x_vec,
  extract_ham_features_for_nn,
  generate_hyper_params_nn_hea,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DATA_DIR = "../data/NN_HEA"
HEA_DATA_DIR = "../data/HEA"
stage3_csv = os.path.join(DATA_DIR, "energies_stage3_NN_HEA.csv")
memory_series_csv = os.path.join(DATA_DIR, "stage3_memory_series_NN_HEA.csv")

STAGE3_FIELD_NAMES = [
  "ham_prefix",
  "n_qubits",
  "score_ml_only",
  "score_full_accelerq",
  "dst",
  "optimal_x_vec",
  "ml_only_x_vec",
  "status",
  "total_time_s",
  "checkpoint_times",
  "convergence_iter",
  "converged_early",
  "peak_memory_bytes",
  "final_corpus_size",
]

MEMORY_SERIES_FIELD_NAMES = [
  "ham_prefix",
  "n_qubits",
  "iteration",
  "current_bytes",
  "peak_bytes",
]


def append_stage3_result(csv_path, row_dict):
  file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
  mode = "a" if file_exists else "w"
  with open(csv_path, mode, newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=STAGE3_FIELD_NAMES,
      restval="",
      extrasaction="ignore",
    )
    if not file_exists:
      writer.writeheader()
    writer.writerow(row_dict)


def append_memory_series(csv_path, ham_prefix, n_qubits, memory_series):
  file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
  mode = "a" if file_exists else "w"
  with open(csv_path, mode, newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=MEMORY_SERIES_FIELD_NAMES,
      restval="",
      extrasaction="ignore",
    )
    if not file_exists:
      writer.writeheader()
    for iteration, current_bytes, peak_bytes in memory_series:
      writer.writerow({
        "ham_prefix": ham_prefix,
        "n_qubits": n_qubits,
        "iteration": iteration,
        "current_bytes": current_bytes,
        "peak_bytes": peak_bytes,
      })


def _profiling_row_fields(prof):
  if not prof:
    return {}
  return {
    "total_time_s": prof.get("total_time_s"),
    "checkpoint_times": json.dumps(prof.get("checkpoint_times")),
    "convergence_iter": prof.get("convergence_iter"),
    "converged_early": prof.get("converged_early"),
    "peak_memory_bytes": prof.get("peak_memory_bytes"),
    "final_corpus_size": prof.get("final_corpus_size"),
  }


if __name__ == "__main__":
  np.set_printoptions(
    precision=17,
    floatmode="fixed",
    suppress=True,
    formatter={"float": lambda x: f"{x:.17e}"},
  )

  print(">> Start Stage 3 (NN-HEA)")

  ham_dir = "../hamiltonian/"
  prefix = "14qubits_06"  #  >>>>>>>>>>>>>>>> Change only this!
  file_name = prefix + ".data"
  os.makedirs(DATA_DIR, exist_ok=True)
  model_file = os.path.join(HEA_DATA_DIR, "model_hea_pre_xgb_24.json")
  max_size = load_model(model_file, xgb.XGBRegressor).n_features_in_

  print(">> Read ham " + ham_dir + ", " + file_name)
  n_qubits, ham = process_file(ham_dir, file_name)
  n_qubits = int(n_qubits)

  print(">> Start processing: " + file_name + " with qubits " + str(n_qubits))

  if not isinstance(ham, FO):
    print("Error type. Exit")
  else:
    ham_vec = ham_to_vector(ham, 50)
    ham_features = extract_ham_features_for_nn(ham)

    nn_generator = lambda seed, n_qubits: generate_hyper_params_nn_hea(
      seed,
      n_qubits,
      initial_point_seed=NN_SEED,
      ham_features=ham_features,
    )

    print(">> Running Opt. Hyperparameters")
    ml_only_x, score_ml_only, opt_params, score_full_accelerq, prof = opt_hyperparams(
      model_file,
      nn_generator,
      xgb.XGBRegressor,
      n_qubits,
      ham,
      ham_vec,
      max_size,
      test_static_nn_hea,
      test_semi_dynamic_nn_hea,
      None,
    )
    default_x = default_hea_x_vec(n_qubits)
    if opt_params is not None:
      dst = distance.euclidean(default_x, opt_params)
      formatted = [f"{num:.17e}" for num in opt_params]
      lines = [", ".join(formatted[i : i + 3]) for i in range(0, len(formatted), 3)]
      print("Pre-opt succeeded! Result: [")
      for i, line in enumerate(lines):
        if i == len(lines) - 1:
          print("  " + line + " ]")
        else:
          print("  " + line + ",")
      print("Euclidean dist:", dst)
      print("Number of max iteration: " + str(round(opt_params[7])))
      ml_only_json = (
        json.dumps([float(v) for v in ml_only_x])
        if ml_only_x is not None else ""
      )
      append_stage3_result(
        stage3_csv,
        {
          "ham_prefix": prefix,
          "n_qubits": n_qubits,
          "score_ml_only": score_ml_only,
          "score_full_accelerq": score_full_accelerq,
          "dst": dst,
          "optimal_x_vec": json.dumps([float(v) for v in opt_params]),
          "ml_only_x_vec": ml_only_json,
          "status": "success",
          **_profiling_row_fields(prof),
        },
      )
      if prof and prof.get("memory_series"):
        append_memory_series(memory_series_csv, prefix, n_qubits, prof["memory_series"])
    else:
      print("FAILED.")
      append_stage3_result(
        stage3_csv,
        {
          "ham_prefix": prefix,
          "n_qubits": n_qubits,
          "status": "failed",
        },
      )
