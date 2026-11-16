
"""
Validation tests for NN-HEA Stage 3 hyperparameter filtering.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import sys

sys.path.append("../")
from kcl_NN_model import NN_SEED
from kcl_util_NN_HEA import sanitize_x_vec_params_nn_hea
from kcl_util_ansatz_cache import cached_hea_num_params, INVALID_NUM_PARAMS


def _as_sanitized(x_vec_params):
  return sanitize_x_vec_params_nn_hea(x_vec_params)


def test_nn_seed_sentinel(x_vec_params):
  """IND:12 must be NN_SEED (-1) in Stage 3 NN mode."""
  return round(x_vec_params[12]) == NN_SEED


def test_iter_max_bounds(x_vec_params):
  x_vec = _as_sanitized(x_vec_params)
  iter_max = int(round(x_vec[7]))
  if iter_max < 10:
    return False
  if iter_max > 500:
    return False
  return True


def test_reps_bounds(x_vec_params):
  x_vec = _as_sanitized(x_vec_params)
  reps = int(round(x_vec[2]))
  return 1 <= reps <= 6


def test_categorical_indices(x_vec_params):
  x_vec = _as_sanitized(x_vec_params)
  if not (0 <= round(x_vec[3]) <= 2):
    return False
  if not (0 <= round(x_vec[4]) <= 2):
    return False
  if not (0 <= round(x_vec[5]) <= 1):
    return False
  if not (0 <= round(x_vec[6]) <= 2):
    return False
  if not (0 <= round(x_vec[11]) <= 2):
    return False
  if not (1 <= round(x_vec[13]) <= 20):
    return False
  return True


def test_cobyla_gradient(x_vec_params):
  x_vec = _as_sanitized(x_vec_params)
  if round(x_vec[6]) == 0 and round(x_vec[11]) != 0:
    return False
  return True


def test_float_bounds(x_vec_params):
  x_vec = _as_sanitized(x_vec_params)
  if not (1e-8 <= x_vec[8] <= 1e-2):
    return False
  if not (1e-8 <= x_vec[9] <= 1e-4):
    return False
  return True


def test_ansatz_parameter_count(x_vec_params, n_qubits):
  x_vec = _as_sanitized(x_vec_params)
  n_qubits = int(round(x_vec[0])) if n_qubits is None else int(n_qubits)
  reps = int(round(x_vec[2]))
  num_params = cached_hea_num_params(
    n_qubits,
    reps,
    int(round(x_vec[3])),
    int(round(x_vec[4])),
    int(round(x_vec[5])),
  )
  if num_params == INVALID_NUM_PARAMS:
    return False

  if n_qubits >= 20 and reps * n_qubits > 120:
    return False
  if num_params <= 0:
    return False
  if n_qubits >= 20 and num_params > 5000:
    return False
  return True


def test_static_nn_hea(x_vec_params, hamiltonian):
  del hamiltonian
  if not test_iter_max_bounds(x_vec_params):
    return False
  if not test_reps_bounds(x_vec_params):
    return False
  if not test_categorical_indices(x_vec_params):
    return False
  if not test_cobyla_gradient(x_vec_params):
    return False
  if not test_float_bounds(x_vec_params):
    return False
  return True


def test_semi_dynamic_nn_hea(x_vec_params, hamiltonian):
  del hamiltonian
  x_vec = _as_sanitized(x_vec_params)
  n_qubits = int(round(x_vec[0]))
  return test_ansatz_parameter_count(x_vec, n_qubits)
