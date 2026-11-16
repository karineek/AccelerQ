
"""
This script contains a set of utility functions for the UCCSD-VQE QE implementation
used in the AccelerQ pipeline.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import sys

sys.path.append("../")
from kcl_UCCSD_VQE import Wrapper_UCCSD
from kcl_util_hamiltonian import compress_hamiltonian

import numpy as np
import random

# Stage 1 augmentation threshold (ML features); distinct from per-run IND:8 coeff_cutoff.
STAGE1_CUTOFF = 0.05

ITER_MAX_MIN = 10
ITER_MAX_CAP = 500

# UCCSD x_vec_params index table  (matches Wrapper_UCCSD._build_x_vec exactly)
#
# IND:0  — n_qubits              float  system size; fixed per run, not optimised
# IND:1  — is_classical          int    {1} fixed to 1 (classical sim); not optimised
# IND:2  — initial_state         int    {0=None, 1=HartreeFock, 2=fixed Pauli rotation state}
# IND:3  — reps                  int    [1, 3]  repetitions of UCCSD operator pool
# IND:4  — electron_fraction     int    {0=quarter-filling, 1=half-filling}
# IND:5  — generalized           int    {0=False, 1=True}  non-particle-conserving excitations
# IND:6  — optimizer_choice      int    {0=COBYLA, 1=SPSA, 2=L_BFGS_B}
# IND:7  — iter_max              int    [10, 500]  max optimizer iterations
# IND:8  — coeff_cutoff          float  [1e-8, 1e-2]  Hamiltonian compression threshold
# IND:9  — atol                  float  [1e-8, 1e-4]  convergence tolerance
# IND:10 — shots                 int    [10**2, 10**7]  ignored when is_classical=True
# IND:11 — gradient_method       int    {0=None, 1=ParamShift, 2=FiniteDiff}
# IND:12 — initial_point_seed    int    [0, 10**4]  RNG seed for initial parameters
# IND:13 — max_evals_grouped     int    [1, 20]  patience counter for early stopping

# DEFAULT hyperparameter vector (seed == 1)
# Wrapper_UCCSD(n_qubits, ham, is_classical=True,
#               initial_state=0, reps=2, electron_fraction=0, generalized=0,
#               optimizer_choice=0, iter_max=500, coeff_cutoff=1e-5,
#               atol=1e-6, shots=10**4, gradient_method=0,
#               initial_point_seed=0, max_evals_grouped=5)
#
# Rationale for defaults:
#   initial_state=0      no defined start state, more optimiser impact but greater danger of local minimum
#   reps=2               middle of [1,3]; balances expressibility vs circuit depth
#   electron_fraction=0  quarter-filling in Ansatz_UCCSD naming; yields total
#                        electrons = n_qubits // 2 on benchmark Hamiltonians
#                        (closed-shell molecular occupation). Must be chosen at
#                        call time together with n_qubits — see generate_hyper_params_uccsd.
#   generalized=0        particle-conserving UCCSD (standard chemistry ansatz)
#   optimizer=0          COBYLA; gradient-free, robust baseline, requires gradient_method=0
#   iter_max=500         global cap; shared by default, GA sampling, and sanitize
#   coeff_cutoff=1e-5    standard Hamiltonian compression; well below chemical accuracy
#   atol=1e-6            matches Qiskit VQE internal default convergence tolerance
#   shots=10**4          commonly used NISQ simulation benchmark value (inactive here)
#   gradient_method=0    None; required when optimizer=COBYLA
#   initial_point_seed=0 canonical reproducible starting point
#   max_evals_grouped=5  moderate patience; avoids premature stopping on oscillating landscapes
#
# Shared optimizer / solver defaults (indices IND:6–13); ansatz-specific defaults
# (electron_fraction) are resolved in generate_hyper_params_uccsd from n_qubits.
_DEFAULT_SOLVER = [
    0,
    2,
    0,
    0,
    ITER_MAX_CAP,
    1e-5,
    1e-6,
    10**4,
    0,
    0,
    5,
]


def _default_e_frac(n_qubits):
    """Default IND:4=0 (quarter-filling) matches closed-shell n_e = n_qubits // 2."""
    _ = n_qubits  # explicit: default is categorical but tied to physical convention
    return 0


def _sample_iter_max():
    """Multi-tier sample for iter_max in [ITER_MAX_MIN, ITER_MAX_CAP]."""
    tier = random.randint(1, 3)
    if tier == 1:
        return random.randint(ITER_MAX_MIN, 50)
    elif tier == 2:
        return random.randint(50, 200)
    return random.randint(200, ITER_MAX_CAP)


def _sample_shots():
    """Multi-tier log-uniform shots in [1e2, 1e7]; ignored under classical sim but kept in x_vec."""
    tier = random.randint(1, 4)
    if tier == 1:
        return random.randint(10**2,   10**3)
    elif tier == 2:
        return random.randint(10**3,   10**5)
    elif tier == 3:
        return random.randint(10**5,   10**6)
    else:
        return random.randint(10**6,   10**7)


def _log_uniform(lo, hi):
    """Uniform draw in log10-space so fine tolerances aren't starved by a flat uniform."""
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    return 10 ** random.uniform(log_lo, log_hi)


def _sample_grad(optimizer_choice):
    """Sample gradient_method; force 0 when optimizer is COBYLA (gradient-free)."""
    if optimizer_choice == 0:
        return 0  # COBYLA is gradient-free
    return random.randint(0, 2)


def _generate_integers(optimizer_choice, n_qubits):
    """Random int hyperparams in x_vec order; generalized forced 0 for n_qubits>=8."""
    rvi = []
    rvi.append(random.randint(0, 2))  # initial_state

    # Ceiling of 3 (not HEA's 6): UCCSD depth scales as O(n^4); reps > 2 on
    # systems above 8 qubits exceeds the classical training budget.
    rvi.append(random.randint(1, 3))

    # Uniform categorical over quarter- vs half-filling (both physically meaningful).
    rvi.append(random.randint(0, 1))

    # generalized=1 at 8q+ (~112 excitation terms * reps) is too deep to simulate.
    if n_qubits >= 8:
        rvi.append(0)
    else:
        rvi.append(random.randint(0, 1))

    rvi.append(_sample_iter_max())
    rvi.append(_sample_shots())
    rvi.append(_sample_grad(optimizer_choice))
    rvi.append(random.randint(0, 10**4))
    rvi.append(random.randint(1, 20))
    return rvi


def _generate_floats():
    """Log-uniform [coeff_cutoff, atol] in x_vec order."""
    rvf = []
    rvf.append(_log_uniform(1e-8, 1e-2))
    rvf.append(_log_uniform(1e-8, 1e-4))
    return rvf


def generate_hyper_params_uccsd(seed, n_qubits):
    """14-float UCCSD x_vec; seed<=1 is default (e_frac from n_qubits), else random."""
    np.set_printoptions(precision=17)

    if seed <= 1:
        initial_state            = _DEFAULT_SOLVER[0]
        reps                     = _DEFAULT_SOLVER[1]
        electron_fraction        = _default_e_frac(n_qubits)
        generalized              = _DEFAULT_SOLVER[2]
        optimizer_choice         = _DEFAULT_SOLVER[3]
        iter_max                 = _DEFAULT_SOLVER[4]
        coeff_cutoff             = _DEFAULT_SOLVER[5]
        atol                     = _DEFAULT_SOLVER[6]
        shots                    = _DEFAULT_SOLVER[7]
        gradient_method          = _DEFAULT_SOLVER[8]
        initial_point_seed       = _DEFAULT_SOLVER[9]
        max_evals_grouped        = _DEFAULT_SOLVER[10]

    else:
        optimizer_choice = random.randint(0, 2)

        rvi = _generate_integers(optimizer_choice, n_qubits)
        rvf = _generate_floats()

        initial_state       = rvi[0]
        reps                = rvi[1]
        electron_fraction   = rvi[2]
        generalized         = rvi[3]
        iter_max            = rvi[4]
        shots               = rvi[5]
        gradient_method     = rvi[6]
        initial_point_seed  = rvi[7]
        max_evals_grouped   = rvi[8]

        coeff_cutoff = rvf[0]
        atol         = rvf[1]

    x_vec_params = np.array(
        [
            float(n_qubits),
            1.0,
            float(initial_state),
            float(reps),
            float(electron_fraction),
            float(generalized),
            float(optimizer_choice),
            float(iter_max),
            float(coeff_cutoff),
            float(atol),
            float(shots),
            float(gradient_method),
            float(initial_point_seed),
            float(max_evals_grouped),
        ],
        dtype=float,
    )

    return x_vec_params


def sanitize_x_vec_params(x_vec_params):
    """Clamp GA-mutated UCCSD x_vec; generalized=0 for n>=8; COBYLA forces gradient_method=0."""
    x = np.array(x_vec_params, dtype=float, copy=True)
    n_q = int(round(x[0]))
    x[2] = max(0, min(2, round(x[2])))
    x[3] = max(1, min(3, round(x[3])))
    x[4] = max(0, min(1, round(x[4])))
    x[5] = 0 if n_q >= 8 else max(0, min(1, round(x[5])))
    x[6] = max(0, min(2, round(x[6])))
    x[7] = max(ITER_MAX_MIN, min(ITER_MAX_CAP, round(x[7])))
    x[8] = float(np.clip(x[8], 1e-8, 1e-2))
    x[9] = float(np.clip(x[9], 1e-8, 1e-4))
    x[10] = max(10**2, min(10**7, round(x[10])))
    x[11] = max(0, min(2, round(x[11])))
    if round(x[6]) == 0:
        x[11] = 0.0
    x[12] = max(0, min(10**4, round(x[12])))
    x[13] = max(1, min(20, round(x[13])))
    return x


def default_uccsd_x_vec(n_qubits):
    """14-element default hyperparameter vector for distance reporting in Stage 3."""
    return generate_hyper_params_uccsd(1, n_qubits)


def wrapper_uccsd(x_vec_params, n_qubits, ham, ham_seed=5, ham_dir="../hamiltonian"):
    """Run classical UCCSD-VQE via Wrapper_UCCSD; returns energy Y (ham arg is signature parity only)."""
    x = sanitize_x_vec_params(x_vec_params)
    wrapper = Wrapper_UCCSD(
        n_qubits           = n_qubits,
        ham                = ham,
        is_classical       = True,
        initial_state      = round(x[2]),
        reps               = round(x[3]),
        electron_fraction  = round(x[4]),
        generalized        = round(x[5]),
        optimizer_choice   = round(x[6]),
        iter_max           = round(x[7]),
        coeff_cutoff       = x[8],
        atol               = x[9],
        shots              = round(x[10]),
        gradient_method    = round(x[11]),
        initial_point_seed = round(x[12]),
        max_evals_grouped  = round(x[13]),
    )

    energy = wrapper.get_result(seed=ham_seed, ham_dir=ham_dir)
    return energy


def compress_uccsd(ham):
    """Stage 1 ML-feature compress at fixed 0.05; VQE uses IND:8 coeff_cutoff instead."""
    return compress_hamiltonian(ham, STAGE1_CUTOFF)
