
"""
This script contains a set of utility functions for the HEA-VQE QE implementation
used in the AccelerQ pipeline.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import sys

sys.path.append("../")
from kcl_HEA_VQE import Ansatz_HEA, Wrapper_HEA
from kcl_util import csv_length, init_or_append_csv
from kcl_util_hamiltonian import compress_hamiltonian

import numpy as np
import random

# Maximum ansatz.num_parameters across valid HEA configs (n_qubits<=24, reps<=6).
ANGLE_DIM = 336

OPTIMAL_HEA_FIELD_NAMES = [
    "record_id",
    "ham_id",
    "npy_index",
    "x_file",
    "num_params",
    "optimal_angles",
]

# Stage 1 augmentation threshold (ML features); distinct from per-run IND:8 coeff_cutoff.
STAGE1_CUTOFF = 0.05

ITER_MAX_MIN = 10
ITER_MAX_CAP = 500

# HEA x_vec_params index table  (matches Wrapper_HEA._build_x_vec exactly)
#
# IND:0  — n_qubits              float  system size; fixed per run, not optimised
# IND:1  — is_classical          int    {1} fixed to 1 (classical sim); not optimised
# IND:2  — reps                  int    [1, 6]  number of EfficientSU2 repetition layers
# IND:3  — entanglement          int    {0='linear', 1='full', 2='circular'}
# IND:4  — rotation_blocks       int    {0=['ry'], 1=['ry','rz'], 2=['rx','ry']}
# IND:5  — skip_final_rotation   int    {0=False, 1=True}  omit final rotation layer
# IND:6  — optimizer_choice      int    {0=COBYLA, 1=SPSA, 2=L_BFGS_B}
# IND:7  — iter_max              int    [10, 500]  max optimizer iterations
# IND:8  — coeff_cutoff          float  [1e-8, 1e-2]  Hamiltonian compression threshold
# IND:9  — atol                  float  [1e-8, 1e-4]  convergence tolerance
# IND:10 — shots                 int    [10**2, 10**7]  ignored when is_classical=True
# IND:11 — gradient_method       int    {0=None, 1=ParamShift, 2=FiniteDiff}
# IND:12 — initial_point_seed    int    [0, 10**4]  RNG seed for initial parameters
# IND:13 — max_evals_grouped     int    [1, 20]  patience counter for early stopping

# DEFAULT hyperparameter vector (seed == 1)
# Wrapper_HEA(n_qubits, ham, is_classical=True,
#             reps=3, entanglement=0, rotation_blocks=1, skip_final_rotation_layer=0,
#             optimizer_choice=0, iter_max=500, coeff_cutoff=1e-5,
#             atol=1e-6, shots=10**4, gradient_method=0,
#             initial_point_seed=0, max_evals_grouped=5)
#
# Rationale for defaults:
#   reps=3               middle of [1,6]; balances expressibility vs barren-plateau risk
#   entanglement=0       linear; most NISQ-realistic and Qiskit default
#   rotation_blocks=1    ['ry','rz']; standard EfficientSU2 full-SU2 rotation
#   skip_final=0         include final rotation layer (Qiskit default)
#   optimizer=0          COBYLA; gradient-free, robust baseline, requires gradient_method=0
#   iter_max=500         global cap; shared by default, GA sampling, and sanitize
#   coeff_cutoff=1e-5    standard Hamiltonian compression; well below chemical accuracy
#   atol=1e-6            matches Qiskit VQE internal default convergence tolerance
#   shots=10**4          commonly used NISQ simulation benchmark value (inactive here)
#   gradient_method=0    None; required when optimizer=COBYLA
#   initial_point_seed=0 canonical reproducible starting point
#   max_evals_grouped=5  moderate patience; avoids premature stopping on oscillating landscapes
_DEFAULT_HEA = [
    3,
    0,
    1,
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


def _generate_integers(optimizer_choice):
    """Random int hyperparams in x_vec order (indices 2–7, 10–13); needs optimizer for COBYLA rule."""
    rvi = []

    # Range ceiling of 6 balances expressibility against barren-plateau risk
    # and classical simulation cost on 4-16 qubit training systems.
    rvi.append(random.randint(1, 6))

    # Uniform categorical: all three topologies are equally valid a priori;
    # the ML model learns which correlates with lower energy per Hamiltonian class.
    rvi.append(random.randint(0, 2))

    # Uniform categorical over ['ry'] / ['ry','rz'] / ['rx','ry'].
    rvi.append(random.randint(0, 2))

    # Binary flag; uniform sampling exposes both structural variants to the model.
    rvi.append(random.randint(0, 1))

    rvi.append(_sample_iter_max())
    rvi.append(_sample_shots())
    rvi.append(_sample_grad(optimizer_choice))

    # 10,000 seeds provides ample landscape diversity across 400-500 training
    # samples. The ML model can learn which starting regions escape barren
    # plateaus for each Hamiltonian structure.
    rvi.append(random.randint(0, 10**4))

    # Upper bound of 20 is strict enough to ensure genuine convergence detection
    # while remaining below the threshold where patience never triggers.
    rvi.append(random.randint(1, 20))

    return rvi


def _generate_floats():
    """Log-uniform [coeff_cutoff, atol] in x_vec order."""
    rvf = []

    # 1e-8 avoids cutting physically relevant terms (sub-µHartree contributions).
    # 1e-2 is the aggressive-but-defensible ceiling matching the ADAPT-VQE range;
    # for dense QunaSys Hamiltonians it produces noticeably sparse operators.
    rvf.append(_log_uniform(1e-8, 1e-2))

    # 1e-8 approaches machine precision (meaningful for L-BFGS-B on smooth landscapes).
    # 1e-4 is the loose-convergence ceiling: ~0.063 kcal/mol, on the edge of
    # chemical accuracy. Looser tolerances produce premature termination.
    rvf.append(_log_uniform(1e-8, 1e-4))

    return rvf


def generate_hyper_params_hea(seed, n_qubits):
    """14-float HEA x_vec; seed<=1 is default, else random (optimizer sampled before gradient)."""
    np.set_printoptions(precision=17)

    if seed <= 1:
        reps                     = _DEFAULT_HEA[0]
        entanglement             = _DEFAULT_HEA[1]
        rotation_blocks          = _DEFAULT_HEA[2]
        skip_final_rot      = _DEFAULT_HEA[3]
        optimizer_choice         = _DEFAULT_HEA[4]
        iter_max                 = _DEFAULT_HEA[5]
        coeff_cutoff             = _DEFAULT_HEA[6]
        atol                     = _DEFAULT_HEA[7]
        shots                    = _DEFAULT_HEA[8]
        gradient_method          = _DEFAULT_HEA[9]
        initial_point_seed       = _DEFAULT_HEA[10]
        max_evals_grouped        = _DEFAULT_HEA[11]

    else:
        optimizer_choice = random.randint(0, 2)

        rvi = _generate_integers(optimizer_choice)
        rvf = _generate_floats()

        reps                = rvi[0]
        entanglement        = rvi[1]
        rotation_blocks     = rvi[2]
        skip_final_rot = rvi[3]
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
            float(reps),
            float(entanglement),
            float(rotation_blocks),
            float(skip_final_rot),
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
    """Clamp GA-mutated HEA x_vec into valid ranges; COBYLA forces gradient_method=0."""
    x = np.array(x_vec_params, dtype=float, copy=True)
    x[2] = max(1, min(6, round(x[2])))
    x[3] = max(0, min(2, round(x[3])))
    x[4] = max(0, min(2, round(x[4])))
    x[5] = max(0, min(1, round(x[5])))
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


def default_hea_x_vec(n_qubits):
    """14-element default hyperparameter vector for distance reporting in Stage 3."""
    return generate_hyper_params_hea(1, n_qubits)


def _flatten_opt_angles(result, n_qubits, x):
    """Flatten converged parameters in ansatz order; zero-pad to ANGLE_DIM."""
    ansatz = Ansatz_HEA(
        n_qubits,
        int(round(x[2])),
        int(round(x[3])),
        int(round(x[4])),
        int(round(x[5])),
    )
    param_names = [p.name for p in ansatz.circuit.parameters]
    # Keys may be Parameter objects or names; normalize for string lookup.
    opt_raw = result.optimal_parameters or {}
    opt = {getattr(k, "name", str(k)): v for k, v in opt_raw.items()}
    values = [float(opt.get(name, 0.0)) for name in param_names]
    padded = np.zeros(ANGLE_DIM, dtype=np.float64)
    n = min(len(values), ANGLE_DIM)
    padded[:n] = values[:n]
    return len(values), padded


def _build_wrapper_hea(x, n_qubits, ham):
    """Construct Wrapper_HEA from a sanitized x_vec (shared by wrapper_hea / miner)."""
    return Wrapper_HEA(
        n_qubits=n_qubits,
        ham=ham,
        is_classical=True,
        reps=round(x[2]),
        entanglement=round(x[3]),
        rotation_blocks=round(x[4]),
        skip_final_rot=round(x[5]),
        optimizer_choice=round(x[6]),
        iter_max=round(x[7]),
        coeff_cutoff=x[8],
        atol=x[9],
        shots=round(x[10]),
        gradient_method=round(x[11]),
        initial_point_seed=round(x[12]),
        max_evals_grouped=round(x[13]),
    )


def wrapper_hea(x_vec_params, n_qubits, ham, ham_seed=5, ham_dir="../hamiltonian"):
    """Run classical HEA-VQE via Wrapper_HEA; returns energy Y (ham arg is signature parity only)."""
    x = sanitize_x_vec_params(x_vec_params)
    wrapper = _build_wrapper_hea(x, n_qubits, ham)
    energy = wrapper.get_result(seed=ham_seed, ham_dir=ham_dir)
    return energy


def wrapper_hea_miner(
    x_vec_params,
    n_qubits,
    ham,
    ham_id,
    records_csv_file,
    optimal_csv_file,
    ham_seed=5,
    ham_dir="../hamiltonian",
):
    """Stage 1 miner: run VQE, return energy, append optimal angles sidecar."""
    x = sanitize_x_vec_params(x_vec_params)
    wrapper = _build_wrapper_hea(x, n_qubits, ham)
    energy = wrapper.get_result(seed=ham_seed, ham_dir=ham_dir)
    result = wrapper.result_for_evaluation()

    record_id = csv_length(records_csv_file) + 1 if csv_length(records_csv_file) > 0 else 1
    num_params, padded = _flatten_opt_angles(result, n_qubits, x)
    angles_str = ",".join(f"{v:.17e}" for v in padded)

    init_or_append_csv(
        optimal_csv_file,
        OPTIMAL_HEA_FIELD_NAMES,
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


def compress_hea(ham):
    """Stage 1 ML-feature compress at fixed 0.05; VQE uses IND:8 coeff_cutoff instead."""
    return compress_hamiltonian(ham, STAGE1_CUTOFF)
