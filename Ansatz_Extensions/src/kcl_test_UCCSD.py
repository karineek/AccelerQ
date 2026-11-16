
"""
Validation tests for filtering UCCSD-VQE hyperparameter configurations in AccelerQ Stage 3.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import sys
from kcl_util_UCCSD import sanitize_x_vec_params
from kcl_util_ansatz_cache import cached_uccsd_num_params, INVALID_NUM_PARAMS

sys.path.append("../")

def test_iter_max_bounds(x_vec_params):
    """IND:7 iter_max must be in [10, 500] after sanitization."""
    x_vec = sanitize_x_vec_params(x_vec_params)
    iter_max = int(round(x_vec[7]))
    if iter_max < 10:
        return False
    if iter_max > 500:
        return False
    return True


def test_reps_bounds(x_vec_params):
    """IND:3 reps in [1, 3]."""
    x_vec = sanitize_x_vec_params(x_vec_params)
    reps = int(round(x_vec[3]))
    return 1 <= reps <= 3

def test_categorical_indices(x_vec_params):
    """IND:2-6, 11 within valid categorical ranges."""
    x_vec = sanitize_x_vec_params(x_vec_params)

    if not (0 <= round(x_vec[2]) <= 2):
        return False
    if not (1 <= round(x_vec[3]) <= 3):
        return False
    if not (0 <= round(x_vec[4]) <= 1):
        return False
    if not (0 <= round(x_vec[5]) <= 1):
        return False
    if not (0 <= round(x_vec[6]) <= 2):
        return False
    if not (0 <= round(x_vec[11]) <= 2):
        return False
    return True

def test_cobyla_gradient(x_vec_params):
    """COBYLA (IND:6==0) requires gradient_method (IND:11)==0."""
    x_vec = sanitize_x_vec_params(x_vec_params)
    if round(x_vec[6]) == 0 and round(x_vec[11]) != 0:
        return False
    return True


def test_float_bounds(x_vec_params):
    """IND:8 coeff_cutoff and IND:9 atol within training ranges."""
    x_vec = sanitize_x_vec_params(x_vec_params)
    if not (1e-8 <= x_vec[8] <= 1e-2):
        return False
    if not (1e-8 <= x_vec[9] <= 1e-4):
        return False
    return True

def test_ansatz_parameter_count(x_vec_params, n_qubits):
    """
    Semi-dynamic: build Ansatz_UCCSD only; reject extreme parameter counts on large systems.
    """
    x_vec = sanitize_x_vec_params(x_vec_params)
    n_qubits = int(round(x_vec[0])) if n_qubits is None else int(n_qubits)

    initial_state    = int(round(x_vec[2]))
    reps             = int(round(x_vec[3]))
    electron_fraction = int(round(x_vec[4]))
    generalized      = int(round(x_vec[5]))

    # Generalized excitations disabled at the sampling stage - circuit depth 
    # exceeds classical simulation tractability
    # Generalised == 1 still exists at 4 and 6 qubit systems (covering this dimension)
    if n_qubits == 8 and generalized == 1:
        return False
    if n_qubits >= 8 and reps > 2 and generalized == 1:
        return False

    # Pre-construction guard: sd + half-filling is the worst-case excitation pool.
    # At n_qubits >= 20 this combination produces circuits too deep to train within budget.
    if n_qubits >= 20 and initial_state == 2 and electron_fraction == 1:
        return False

    num_params = cached_uccsd_num_params(
        n_qubits, initial_state, reps, electron_fraction, generalized,
    )
    if num_params == INVALID_NUM_PARAMS:
        return False

    # Degenerate circuit: no parameters means VQE cannot optimise anything.
    if num_params <= 0:
        return False

    # Absolute ceiling: mirrors HEA guard; UCCSD can hit this faster than HEA
    # due to combinatorial excitation pool growth, so the same threshold still applies.
    if n_qubits >= 20 and num_params > 5000:
        return False

    return True

def test_static_uccsd(x_vec_params, hamiltonian):
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


def test_semi_dynamic_uccsd(x_vec_params, hamiltonian):
    """Ansatz structure checks without running VQE."""
    del hamiltonian
    x_vec = sanitize_x_vec_params(x_vec_params)
    n_qubits = int(round(x_vec[0]))
    return test_ansatz_parameter_count(x_vec, n_qubits)