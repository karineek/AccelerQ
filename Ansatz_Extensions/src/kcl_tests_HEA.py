
"""
Validation tests for filtering HEA-VQE hyperparameter configurations in AccelerQ Stage 3.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import sys

sys.path.append("../")
from kcl_util_HEA import sanitize_x_vec_params
from kcl_util_ansatz_cache import cached_hea_num_params, INVALID_NUM_PARAMS


def _as_sanitized(x_vec_params):
    return sanitize_x_vec_params(x_vec_params)


def test_iter_max_bounds(x_vec_params):
    """IND:7 iter_max must be in [10, 500] after sanitization."""
    x_vec = _as_sanitized(x_vec_params)
    iter_max = int(round(x_vec[7]))
    if iter_max < 10:
        return False
    if iter_max > 500:
        return False
    return True


def test_reps_bounds(x_vec_params):
    """IND:2 reps in [1, 6]."""
    x_vec = _as_sanitized(x_vec_params)
    reps = int(round(x_vec[2]))
    return 1 <= reps <= 6


def test_categorical_indices(x_vec_params):
    """IND:3-6, 11, 13 within valid categorical ranges."""
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
    """COBYLA (IND:6==0) requires gradient_method (IND:11)==0."""
    x_vec = _as_sanitized(x_vec_params)
    if round(x_vec[6]) == 0 and round(x_vec[11]) != 0:
        return False
    return True


def test_float_bounds(x_vec_params):
    """IND:8 coeff_cutoff and IND:9 atol within training ranges."""
    x_vec = _as_sanitized(x_vec_params)
    if not (1e-8 <= x_vec[8] <= 1e-2):
        return False
    if not (1e-8 <= x_vec[9] <= 1e-4):
        return False
    return True


def test_ansatz_parameter_count(x_vec_params, n_qubits):
    """
    Semi-dynamic: build Ansatz_HEA only; reject extreme parameter counts on large systems.
    """
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


def test_static_hea(x_vec_params, hamiltonian):
    """Static filters before ML prediction (no VQE execution)."""
    del hamiltonian  # unused; signature matches opt_hyperparams contract
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


def test_semi_dynamic_hea(x_vec_params, hamiltonian):
    """Ansatz structure checks without running VQE."""
    del hamiltonian
    x_vec = _as_sanitized(x_vec_params)
    n_qubits = int(round(x_vec[0]))
    return test_ansatz_parameter_count(x_vec, n_qubits)
