
"""
Utility functions for the NN-HEA pipeline (learned warm-start for Stage 3 GA).

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import os
import sys

sys.path.append("../")

import numpy as np

from kcl_HEA_VQE import Wrapper_HEA
from kcl_NN_model import (
  NN_SEED,
  load_nn_model,
  predict_x_vec,
)
from kcl_util import csv_length, init_or_append_csv
from kcl_util_HEA import (
  _flatten_opt_angles,
  default_hea_x_vec,
  generate_hyper_params_hea,
  sanitize_x_vec_params,
)
from kcl_util_hamiltonian import get_hamiltonian_features

OPTIMAL_NN_HEA_FIELD_NAMES = [
  "record_id",
  "ham_id",
  "npy_index",
  "x_file",
  "num_params",
  "optimal_angles",
]


def extract_ham_features_for_nn(ham) -> np.ndarray:
  """
  Fixed-length Hamiltonian coefficient vector for NN training and inference.

  Uses get_hamiltonian_features (ml_cutoff=0.05, length HAMILTONIAN_FEATURE_LENGTH).
  """
  return np.asarray(get_hamiltonian_features(ham), dtype=np.float64)


def sanitize_x_vec_params_nn_hea(x_vec_params):
  """Like HEA sanitize but preserves NN_SEED sentinel at IND:12."""
  x = sanitize_x_vec_params(x_vec_params)
  orig = np.array(x_vec_params, dtype=float)
  if round(orig[12]) == NN_SEED:
    x[12] = float(NN_SEED)
  return x


_NN_MODEL_PATH = os.path.join(
  os.path.dirname(__file__), "..", "data", "NN_HEA", "nn_hea_model.pt"
)
_nn_load_error = None
try:
  _nn_model, _scaler_mean, _scaler_std = load_nn_model(_NN_MODEL_PATH)
except Exception as exc:
  _nn_model = None
  _scaler_mean = None
  _scaler_std = None
  _nn_load_error = f"{type(exc).__name__}: {exc}"


def generate_hyper_params_nn_hea(
  seed,
  n_qubits,
  initial_point_seed=None,
  ham_features=None,
):
  """HEA x_vec, or Stage-3 NN warm-start (IND:12=-1; seed==1 uses NN predict)."""
  if initial_point_seed is None:
    return generate_hyper_params_hea(seed, n_qubits)

  if int(round(initial_point_seed)) != NN_SEED:
    raise ValueError(f"initial_point_seed must be None or NN_SEED ({NN_SEED})")

  if seed == 1:
    if ham_features is None:
      raise ValueError("ham_features required for NN warm-start (seed == 1)")
    if _nn_model is None:
      detail = f" ({_nn_load_error})" if _nn_load_error else ""
      raise RuntimeError(f"NN model not loaded from {_NN_MODEL_PATH}{detail}")
    return predict_x_vec(_nn_model, _scaler_mean, _scaler_std, ham_features, n_qubits)

  x = generate_hyper_params_hea(seed, n_qubits)
  x = np.array(x, dtype=float, copy=True)
  x[12] = float(NN_SEED)
  return x


def _build_wrapper_from_x(x, n_qubits, ham):
  return Wrapper_HEA(
    n_qubits=n_qubits,
    ham=ham,
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


def wrapper_nn_hea_miner(
  x_vec_params,
  n_qubits,
  ham,
  ham_id,
  records_csv_file,
  optimal_csv_file,
  ham_seed=5,
  ham_dir="../hamiltonian",
):
  """
  Stage 1 data-collection wrapper: run VQE, return energy, append optimal angles sidecar.
  """
  x = sanitize_x_vec_params(x_vec_params)
  wrapper = _build_wrapper_from_x(x, n_qubits, ham)
  energy = wrapper.get_result(seed=ham_seed, ham_dir=ham_dir)
  result = wrapper.result_for_evaluation()

  record_id = csv_length(records_csv_file) + 1 if csv_length(records_csv_file) > 0 else 1
  num_params, padded = _flatten_opt_angles(result, n_qubits, x)
  angles_str = ",".join(f"{v:.17e}" for v in padded)

  init_or_append_csv(
    optimal_csv_file,
    OPTIMAL_NN_HEA_FIELD_NAMES,
    {
      "record_id": record_id,
      "ham_id": ham_id,
      "npy_index": "",
      "x_file": "",
      "num_params": num_params,
      "optimal_angles": angles_str,
    },
  )
  return energy


def wrapper_nn_hea(
  x_vec_params,
  n_qubits,
  ham,
  ham_seed=5,
  ham_dir="../hamiltonian",
  nn_ham_features=None,
):
  """Stage 3 VQE execution with NN sentinel support and feature passthrough."""
  x = sanitize_x_vec_params_nn_hea(x_vec_params)
  wrapper = _build_wrapper_from_x(x, n_qubits, ham)
  energy = wrapper.get_result(
    seed=ham_seed,
    ham_dir=ham_dir,
    nn_ham_features=nn_ham_features,
  )
  return energy
