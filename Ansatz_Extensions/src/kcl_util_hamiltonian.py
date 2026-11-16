
"""
Shared Hamiltonian utilities for HEA, UCCSD, and NN-VQE implementations
in the AccelerQ pipeline.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import os
import sys
import warnings

import numpy as np
from openfermion import count_qubits, jordan_wigner
from openfermion.ops.operators import FermionOperator
from qiskit.quantum_info import SparsePauliOp

sys.path.append("../")
from utils.challenge_2024 import problem_hamiltonian

# Fixed length for Hamiltonian coefficient features (AccelerQ Section 4.1 / XGBoost x_i prefix).
# Shared by HEA, UCCSD, and NN-VQE; revisit when max_size is calibrated against training data.
HAMILTONIAN_FEATURE_LENGTH = 1000

# Repo-root hamiltonian/ (stable regardless of os.chdir in stage scripts).
REPO_HAMILTONIAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hamiltonian"))


def _resolve_ham_dir(ham_dir):
    """Resolve ham_dir after stage ``os.chdir``; fall back to repo ``hamiltonian/``."""
    if os.path.isabs(ham_dir) and os.path.isdir(ham_dir):
        return ham_dir
    cwd_candidate = os.path.abspath(ham_dir)
    if os.path.isdir(cwd_candidate):
        return cwd_candidate
    if os.path.isdir(REPO_HAMILTONIAN_DIR):
        return REPO_HAMILTONIAN_DIR
    return cwd_candidate


def _ham_data_path(n_qubits, seed, ham_dir):
    """Build the conventional on-disk path for a challenge Hamiltonian file."""
    nq = f"{int(n_qubits):02d}"
    if isinstance(seed, int):
        seed_str = f"{seed:02d}"
    else:
        seed_str = str(seed).zfill(2) if len(str(seed)) < 2 else str(seed)
    return os.path.join(ham_dir, f"{nq}qubits_{seed_str}.data")


def _jw_to_sparse_pauli(jw_ham):
    """Term-by-term OpenFermion→SparsePauliOp conversion for pinned Qiskit 0.41.x."""
    n_qubits = count_qubits(jw_ham)
    pauli_list = []
    for pauli_term, coeff in jw_ham.terms.items():
        pauli_arr = ["I"] * n_qubits
        for qubit_idx, action in pauli_term:
            pauli_arr[qubit_idx] = action
        # Qiskit little-endian: rightmost character is qubit 0.
        pauli_str = "".join(reversed(pauli_arr))
        pauli_list.append((pauli_str, coeff))
    if not pauli_list:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(pauli_list)


def load_hamiltonian(n_qubits, seed, ham_dir):
    """Loads a Hamiltonian .data file; centralised here so callers don't import challenge_2024 directly."""
    resolved_dir = _resolve_ham_dir(ham_dir)
    path = _ham_data_path(n_qubits, seed, resolved_dir)
    try:
        return problem_hamiltonian(n_qubits, seed, resolved_dir)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load Hamiltonian for n_qubits={n_qubits}, seed={seed} "
            f"from directory '{ham_dir}'. "
            f"Attempted path: '{path}'. Original error: {exc}"
        ) from exc


def compress_hamiltonian(ham, coeff_cutoff):
    """Drop terms with |c| < coeff_cutoff in place (VQE IND:8; distinct from ML ml_cutoff)."""
    ham.compress(coeff_cutoff)
    if len(ham.terms) == 0:
        warnings.warn(
            f"compress_hamiltonian removed all terms (coeff_cutoff={coeff_cutoff}). "
            "Check VQE hyperparameter configuration.",
            stacklevel=2,
        )
    return ham


def to_qubit_operator(ham):
    """Jordan–Wigner map to a real SparsePauliOp; rejects large imaginary parts."""
    jw_ham = jordan_wigner(ham)
    sparse_op = _jw_to_sparse_pauli(jw_ham)
    max_imag = np.max(np.abs(sparse_op.coeffs.imag))
    if max_imag >= 1e-4:
        raise ValueError(
            f"Hamiltonian has large imaginary part (max |Im| = {max_imag}); "
            "likely non-Hermitian input."
        )
    if max_imag >= 1e-10:
        import warnings
        warnings.warn(
            f"Discarding small imaginary components (max |Im| = {max_imag})",
            UserWarning, stacklevel=2
        )
    return SparsePauliOp.from_list(
        [(p, float(c.real)) for p, c in zip(sparse_op.paulis.to_labels(), sparse_op.coeffs)]
    )


def get_hamiltonian_features(ham, ml_cutoff=0.05, target_length=HAMILTONIAN_FEATURE_LENGTH):
    """Fixed-length |c|-sorted coeff vector for XGBoost (ml_cutoff ≠ VQE coeff_cutoff)."""
    coeffs = [float(coefficient.real) for coefficient in ham.terms.values()]
    filtered = [c for c in coeffs if abs(c) >= ml_cutoff]
    filtered.sort(key=abs, reverse=True)
    if len(filtered) >= target_length:
        out = filtered[:target_length]
    else:
        out = filtered + [0.0] * (target_length - len(filtered))

    return np.array(out, dtype=np.float64)


def compute_hamiltonian_features(ham):
    """Structural [sparsity, term_count, coeff_variance] from raw (uncompressed) ham."""
    coeffs = [abs(float(c.real)) for c in ham.terms.values()]

    n_qubits = count_qubits(ham)
    term_count = len(ham.terms)

    # Conservative upper bound: 4^n Pauli terms after JW mapping.
    max_possible_terms = 4 ** n_qubits
    sparsity = 1.0 - (term_count / max_possible_terms)

    coeff_variance = float(np.var(coeffs)) if coeffs else 0.0

    return [sparsity, term_count, coeff_variance]