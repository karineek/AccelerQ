
"""
HEA-VQE computation logic for the AccelerQ pipeline.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from qiskit.algorithms.gradients import (
    FiniteDiffEstimatorGradient,
    ParamShiftEstimatorGradient,
)
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from kcl_util_hamiltonian import (
    compress_hamiltonian,
    load_hamiltonian,
    to_qubit_operator,
)

ENTANGLEMENT_MAP = {0: "linear", 1: "full", 2: "circular"}
ROTATION_MAP = {0: ["ry"], 1: ["ry", "rz"], 2: ["rx", "ry"]}
OPTIMIZER_MAP = {0: COBYLA, 1: SPSA, 2: L_BFGS_B}
NN_SEED = -1

# HEA x_vec_params index table
# Wrapper_HEA(n_qubits, ham, is_classical, reps, entanglement, rotation_blocks,
#             skip_final_rot, optimizer_choice, iter_max, coeff_cutoff,
#             atol, shots, gradient_method, initial_point_seed, max_evals_grouped)
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



class NotRunError(RuntimeError):
    """Raised when results are accessed before VQE has been run via get_result()."""


def _opt_params_from_vqe(res, ansatz: QuantumCircuit) -> dict:
    """Map VQEResult.optimal_parameters Parameter keys to name strings for CSV lookups."""
    if hasattr(res, "optimal_parameters") and res.optimal_parameters is not None:
        return {
            getattr(k, "name", str(k)): float(v)
            for k, v in dict(res.optimal_parameters).items()
        }
    optimal_point = getattr(res, "optimal_point", None)
    if optimal_point is None:
        return {}
    param_names = [p.name for p in ansatz.parameters]
    return dict(zip(param_names, (float(v) for v in optimal_point)))


@dataclass
class Result_HEA:
    """VQE run output for training and evaluation."""

    energy: float
    optimal_parameters: dict
    convergence_history: List[Tuple[int, float]] = field(default_factory=list)
    n_iterations: int = 0
    n_qubits: int = 0
    x_vec_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


@lru_cache(maxsize=256)
def _build_hea_circuit(
    n_qubits: int,
    reps: int,
    entanglement: int,
    rotation_blocks: int,
    skip_final_rot: int,
):
    """Build EfficientSU2 + num_parameters; callers must .copy() (VQE binds in-place)."""
    circuit = EfficientSU2(
        num_qubits=n_qubits,
        reps=reps,
        entanglement=ENTANGLEMENT_MAP[entanglement],
        su2_gates=ROTATION_MAP[rotation_blocks],
        skip_final_rotation_layer=bool(skip_final_rot),
        skip_unentangled_qubits=False,
    )
    return circuit, circuit.num_parameters


class Ansatz_HEA:
    """Hardware-efficient SU(2) ansatz from hyperparameters."""

    def __init__(
        self,
        n_qubits: int,
        reps: int,
        entanglement: int,
        rotation_blocks: int,
        skip_final_rot: int,
    ):
        if entanglement not in (0, 1, 2):
            raise ValueError(f"entanglement must be 0, 1, or 2; got {entanglement}")
        if rotation_blocks not in (0, 1, 2):
            raise ValueError(f"rotation_blocks must be 0, 1, or 2; got {rotation_blocks}")
        if skip_final_rot not in (0, 1):
            raise ValueError(
                f"skip_final_rot must be 0 or 1; got {skip_final_rot}"
            )
        if reps < 1:
            raise ValueError(f"reps must be >= 1; got {reps}")

        n_qubits = int(n_qubits)
        reps = int(reps)
        entanglement = int(entanglement)
        rotation_blocks = int(rotation_blocks)
        skip_final_rot = int(skip_final_rot)

        cached_circuit, num_params = _build_hea_circuit(
            n_qubits, reps, entanglement, rotation_blocks, skip_final_rot,
        )
        self.circuit = cached_circuit.copy()
        self.num_parameters = num_params


def _make_optimizer(optimizer_choice: int, iter_max: int, atol: float):
    if optimizer_choice == 0:
        return COBYLA(maxiter=iter_max, tol=atol)
    if optimizer_choice == 1:
        return SPSA(maxiter=iter_max)
    return L_BFGS_B(maxiter=iter_max, ftol=atol)


def _make_gradient(gradient_method: int, estimator):
    if gradient_method == 0:
        return None
    if gradient_method == 1:
        return ParamShiftEstimatorGradient(estimator)
    if gradient_method == 2:
        return FiniteDiffEstimatorGradient(estimator, epsilon=0.01)
    raise ValueError(f"gradient_method must be 0, 1, or 2; got {gradient_method}")


class Solver_HEA:
    """Configures and runs Qiskit VQE; produces Result_HEA."""

    def __init__(
        self,
        operator: SparsePauliOp,
        ansatz: QuantumCircuit,
        is_classical: bool,
        optimizer_choice: int,
        iter_max: int,
        atol: float,
        shots: int,
        gradient_method: int,
        initial_point_seed: int,
        max_evals_grouped: int,
        x_vec_params: np.ndarray,
        n_qubits: int,
        nn_ham_features: Optional[np.ndarray] = None,
    ):
        self.operator = operator
        self.ansatz = ansatz
        self.is_classical = is_classical
        self.optimizer_choice = optimizer_choice
        self.iter_max = iter_max
        self.atol = atol
        self.shots = shots  # ignored when is_classical=True (Estimator uses exact simulation)
        self.gradient_method = gradient_method
        self.initial_point_seed = initial_point_seed
        self.max_evals_grouped = max_evals_grouped
        self.x_vec_params = x_vec_params
        self.n_qubits = n_qubits

        self.convergence_history: List[Tuple[int, float]] = []
        self._n_converged = 0
        self._stop_flag = False

        if gradient_method != 0 and optimizer_choice == 0:
            raise ValueError("COBYLA is gradient-free; gradient_method must be 0")

        if is_classical:
            self._estimator = Estimator()
        else:
            raise NotImplementedError(
                "Noisy simulation (is_classical=False) is not implemented for HEA-VQE"
            )

        self._optimizer = _make_optimizer(optimizer_choice, iter_max, atol)
        self._gradient = _make_gradient(gradient_method, self._estimator)
        
        if int(round(initial_point_seed)) == NN_SEED:
            try:
                from kcl_NN_model import load_nn_model, predict_angles

                _nn_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "NN_HEA", "nn_hea_model.pt"
                )
                _m, _sm, _ss = load_nn_model(_nn_path)
                if nn_ham_features is None:
                    raise ValueError("nn_ham_features required on NN sentinel path")
                self._initial_point = predict_angles(
                    _m, _sm, _ss, nn_ham_features, ansatz.num_parameters
                )
            except Exception as e:
                print(f"[NN-HEA] Warning: NN angle prediction failed ({e}), using seed=42")
                rng = np.random.default_rng(42)
                self._initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        else:
            rng = np.random.default_rng(initial_point_seed)
            self._initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

        self._vqe = VQE(
            estimator=self._estimator,
            ansatz=ansatz,
            optimizer=self._optimizer,
            gradient=self._gradient,
            initial_point=self._initial_point,
            callback=self._callback,
        )

    def _callback(self, eval_count, params, energy, metadata):
        # Qiskit VQE does not stop the optimizer when the callback returns; _stop_flag only
        # suppresses further history appends after patience is met.
        if self._stop_flag:
            return
        self.convergence_history.append((eval_count, float(energy)))
        if len(self.convergence_history) >= 2:
            delta = abs(
                self.convergence_history[-1][1] - self.convergence_history[-2][1]
            )
            if delta < self.atol:
                self._n_converged += 1
            else:
                self._n_converged = 0
            if self._n_converged >= self.max_evals_grouped:
                self._stop_flag = True

    def run(self) -> Result_HEA:
        if len(self.convergence_history) > 0:
            raise RuntimeError(
                "Solver_HEA.run() has already been called. "
                "Discard this instance and create a new Solver_HEA for a new run."
            )

        res = self._vqe.compute_minimum_eigenvalue(self.operator)
        energy = float(res.eigenvalue.real)
        opt_params = _opt_params_from_vqe(res, self.ansatz)
        n_iterations = int(res.cost_function_evals)

        return Result_HEA(
            energy=energy,
            optimal_parameters=opt_params,
            convergence_history=self.convergence_history,
            n_iterations=n_iterations,
            n_qubits=self.n_qubits,
            x_vec_params=self.x_vec_params,
        )


class Wrapper_HEA:
    """Orchestrates Hamiltonian prep and VQE; run-before-read contract."""

    def __init__(
        self,
        n_qubits,
        ham,
        is_classical,
        reps,
        entanglement,
        rotation_blocks,
        skip_final_rot,
        optimizer_choice,
        iter_max,
        coeff_cutoff,
        atol,
        shots,
        gradient_method,
        initial_point_seed,
        max_evals_grouped,
    ):
        self.n_qubits = n_qubits
        # ham is retained for kcl_util_HEA signature parity; get_result loads by seed from disk.
        self.ham = ham
        self.is_classical = bool(is_classical)
        self.reps = reps
        self.entanglement = entanglement
        self.rotation_blocks = rotation_blocks
        self.skip_final_rot = skip_final_rot
        self.optimizer_choice = optimizer_choice
        self.iter_max = iter_max
        self.coeff_cutoff = coeff_cutoff
        self.atol = atol
        self.shots = shots
        self.gradient_method = gradient_method
        self.initial_point_seed = initial_point_seed
        self.max_evals_grouped = max_evals_grouped
        self._result: Optional[Result_HEA] = None

    def _prepare_operator(self, seed, ham_dir):
        """Load, compress, and Jordan–Wigner-map the Hamiltonian for this run."""
        ham = load_hamiltonian(self.n_qubits, seed, ham_dir)
        ham = compress_hamiltonian(ham, self.coeff_cutoff)
        return to_qubit_operator(ham)

    def _make_ansatz(self) -> Ansatz_HEA:
        """Build Ansatz_HEA from stored structural hyperparameters."""
        return Ansatz_HEA(
            self.n_qubits,
            self.reps,
            self.entanglement,
            self.rotation_blocks,
            self.skip_final_rot,
        )

    def _make_solver(self, operator, ansatz, nn_ham_features=None) -> Solver_HEA:
        """Build Solver_HEA from stored procedural knobs and optional NN features."""
        return Solver_HEA(
            operator,
            ansatz.circuit,
            self.is_classical,
            self.optimizer_choice,
            self.iter_max,
            self.atol,
            self.shots,
            self.gradient_method,
            self.initial_point_seed,
            self.max_evals_grouped,
            self._build_x_vec(),
            self.n_qubits,
            nn_ham_features=nn_ham_features,
        )

    def get_result(self, seed, ham_dir, nn_ham_features=None) -> float:
        """Run VQE once (run-before-read); raises if called twice on the same instance."""
        if self._result is not None:
            raise RuntimeError(
                "get_result() has already been called on this Wrapper instance. "
                "Create a new Wrapper for a new run."
            )

        operator = self._prepare_operator(seed, ham_dir)
        ansatz = self._make_ansatz()
        solver = self._make_solver(operator, ansatz, nn_ham_features)
        self._result = solver.run()
        return self._result.energy

    def _build_x_vec(self) -> np.ndarray:
        return np.array(
            [
                float(self.n_qubits),
                float(self.is_classical),
                float(self.reps),
                float(self.entanglement),
                float(self.rotation_blocks),
                float(self.skip_final_rot),
                float(self.optimizer_choice),
                float(self.iter_max),
                float(self.coeff_cutoff),
                float(self.atol),
                float(self.shots),
                float(self.gradient_method),
                float(self.initial_point_seed),
                float(self.max_evals_grouped),
            ],
            dtype=float,
        )

    def result_for_training(self) -> Tuple[float, np.ndarray]:
        if self._result is None:
            raise NotRunError(
                "get_result() must be called before accessing results. "
                "VQE execution is deliberate and not triggered automatically."
            )
        return self._result.energy, self._result.x_vec_params

    def result_for_evaluation(self) -> Result_HEA:
        if self._result is None:
            raise NotRunError(
                "get_result() must be called before accessing results. "
                "VQE execution is deliberate and not triggered automatically."
            )
        return self._result
