
"""
Process-level cache for ansatz parameter counts used in Stage 3 semi-dynamic tests.

Building UCCSD/HEA ansätze is a pure function of structural hyperparameters; Stage 3
calls these checks tens of thousands of times per run with only dozens of unique keys.
"""

from functools import lru_cache

from kcl_HEA_VQE import Ansatz_HEA
from kcl_UCCSD_VQE import Ansatz_UCCSD

INVALID_NUM_PARAMS = -1


@lru_cache(maxsize=4096)
def cached_hea_num_params(
    n_qubits, reps, entanglement, rotation_blocks, skip_final_rot,
):
    try:
        return Ansatz_HEA(
            int(n_qubits),
            int(reps),
            int(entanglement),
            int(rotation_blocks),
            int(skip_final_rot),
        ).num_parameters
    except (ValueError, KeyError):
        return INVALID_NUM_PARAMS


@lru_cache(maxsize=4096)
def cached_uccsd_num_params(
    n_qubits, initial_state, reps, electron_fraction, generalized,
):
    try:
        return Ansatz_UCCSD(
            int(n_qubits),
            int(initial_state),
            int(reps),
            int(electron_fraction),
            int(generalized),
        ).num_parameters
    except (ValueError, KeyError):
        return INVALID_NUM_PARAMS
