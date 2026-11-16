"""
UCCSD-VQE computation logic for the AccelerQ pipeline.
 
This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
 """

"""
Structure:
    Result_UCCSD  — dataclass
    Ansatz_UCCSD  — builds UCCSD circuit from hyperparameters
    Solver_UCCSD  — operates on any pre-built QuantumCircuit
    Wrapper_UCCSD — orchestrates Hamiltonian prep + ansatz build + solver
 
UCCSD-specific ansatz hyperparameters:
    excitations        — {0=singles_only, 1=doubles_only, 2=singles_and_doubles}
    reps               — number of times the UCCSD operator pool is repeated [1, 3]
    electron_fraction  — fraction of n_qubits treated as alpha electrons {0=quarter, 1=half}
    generalized        — {0=False, 1=True}  allow non-particle-conserving excitations
 
Qubit mapping is fixed to Jordan-Wigner to stay consistent with to_qubit_operator()
in kcl_util_hamiltonian.py, which also applies Jordan-Wigner. Exposing a second mapper
would silently produce operator/ansatz mismatches.
"""
 

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
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

from kcl_util_hamiltonian import (
    compress_hamiltonian,
    load_hamiltonian,
    to_qubit_operator,
)

OPTIMIZER_MAP = {0: COBYLA, 1: SPSA, 2: L_BFGS_B}

# initial_state type: what classes of UCC excitation operators are included.
# 0 → computational zero state   (default starting point)
# 1 → HartreeFock   
# 2 → Pauli rotation state (fixed π/4 angle rotation)
def _init_state(indx, n_orb, num_particles, qubit_mapper, n_qubits):
    if indx == 0:
        return None
    elif indx == 1:
        return HartreeFock(n_orb, num_particles, qubit_mapper)
    else:
        state = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            state.ry(np.pi/4, i)
        return state

# UCCSD x_vec_params index table
# Wrapper_UCCSD(n_qubits, ham, is_classical, excitations, reps, electron_fraction, generalized,
#               optimizer_choice, iter_max, coeff_cutoff, atol, shots,
#               gradient_method, initial_point_seed, max_evals_grouped)
#
# IND:0  — n_qubits              float  system size; fixed per run, not optimised
# IND:1  — is_classical          int    {1} fixed to 1 (classical sim); not optimised
# IND:2  — initial_state         int    {0=None, 1=Hartree-Fock state, 2=fixed Pauli rotation state}
# IND:3  — reps                  int    [1, 3]  repetitions of UCCSD operator pool
# IND:4  — electron_fraction     int    {0=quarter-filling, 1=half-filling}
# IND:5  — generalized           int    {0=False, 1=True}  allow non-particle-conserving excitations
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
class Result_UCCSD:
    """VQE run output for training and evaluation. Identical fields to Result_HEA."""

    energy: float
    optimal_parameters: dict
    convergence_history: List[Tuple[int, float]] = field(default_factory=list)
    n_iterations: int = 0
    n_qubits: int = 0
    x_vec_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


@lru_cache(maxsize=256)
def _build_uccsd_circuit(
    n_qubits: int,
    initial_state: int,
    reps: int,
    electron_fraction: int,
    generalized: int,
):
    """Build decomposed UCCSD + num_parameters; callers must .copy() (VQE binds in-place)."""
    n_orb = n_qubits // 2
    if electron_fraction == 0:
        n_e = max(1, n_orb // 2)
    else:
        n_e = min(n_orb, n_orb - 1)

    num_particles = (n_e, n_e)
    mapper = JordanWignerMapper()
    state = _init_state(
        initial_state, n_orb, num_particles, mapper, n_qubits,
    )

    circuit = UCCSD(
        num_spatial_orbitals=n_orb,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=state,
        reps=reps,
        generalized=bool(generalized),
    ).decompose()

    return circuit, circuit.num_parameters


class Ansatz_UCCSD:
    """
    Chemistry-inspired UCCSD ansatz built from AccelerQ hyperparameters.

    Particle count derivation
    -------------------------
    UCCSD requires n_orb and num_particles = (n_alpha, n_beta).
    The Hamiltonians in this pipeline use a spin-orbital basis:
        n_qubits = 2 * n_orb
    so n_orb = n_qubits // 2.

    electron_fraction then controls occupation:
        quarter-filling (0): n_alpha = n_beta = n_orb // 2
        half-filling    (1): n_alpha = n_beta = n_orb

    Quarter-filling (sparse occupation) is typical for many challenge Hamiltonians;
    half-filling (every spatial orbital singly occupied) gives the largest excitation
    pool. Both extremes are physically meaningful and produce different circuit depths,
    making electron_fraction a genuinely informative AccelerQ hyperparameter.

    Minimum qubit requirement: n_qubits >= 4 (so n_orb >= 2 and
    n_alpha >= 1 at quarter-filling). The data-generation scripts already enforce
    n_qubits in {4, 8, 12, 16}, so this is always satisfied.
    """

    def __init__(
        self,
        n_qubits: int,
        initial_state: int,
        reps: int,
        electron_fraction: int,
        generalized: int,
    ):
        if initial_state not in (0, 1, 2):
            raise ValueError(f"initial_state must be 0, 1, or 2; got {initial_state}")
        if reps < 1 or reps > 3:
            raise ValueError(f"reps must be in [1, 3]; got {reps}")
        if electron_fraction not in (0, 1):
            raise ValueError(f"electron_fraction must be 0 or 1; got {electron_fraction}")
        if generalized not in (0, 1):
            raise ValueError(f"generalized must be 0 or 1; got {generalized}")
        if n_qubits < 4 or n_qubits % 2 != 0:
            raise ValueError(
                f"n_qubits must be a positive even integer >= 4; got {n_qubits}"
            )

        n_qubits = int(n_qubits)
        initial_state = int(initial_state)
        reps = int(reps)
        electron_fraction = int(electron_fraction)
        generalized = int(generalized)

        cached_circuit, num_params = _build_uccsd_circuit(
            n_qubits, initial_state, reps, electron_fraction, generalized,
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


class Solver_UCCSD:
    """
    Configures and runs Qiskit VQE on a pre-built UCCSD circuit; produces Result_UCCSD.

    """

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
                "Noisy simulation (is_classical=False) is not implemented for UCCSD-VQE"
            )

        self._optimizer = _make_optimizer(optimizer_choice, iter_max, atol)
        self._gradient = _make_gradient(gradient_method, self._estimator)

        # initial_point_seed controls the RNG used to draw starting rotation angles.
        # Consistent seeding across runs makes AccelerQ's hyperparameter comparisons
        # interpretable: different seeds map to distinct starting landscapes.
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

    def run(self) -> Result_UCCSD:
        if len(self.convergence_history) > 0:
            raise RuntimeError(
                "Solver_UCCSD.run() has already been called. "
                "Discard this instance and create a new Solver_UCCSD for a new run."
            )

        res = self._vqe.compute_minimum_eigenvalue(self.operator)
        energy = float(res.eigenvalue.real)
        opt_params = _opt_params_from_vqe(res, self.ansatz)
        n_iterations = int(res.cost_function_evals)

        return Result_UCCSD(
            energy=energy,
            optimal_parameters=opt_params,
            convergence_history=self.convergence_history,
            n_iterations=n_iterations,
            n_qubits=self.n_qubits,
            x_vec_params=self.x_vec_params,
        )


class Wrapper_UCCSD:
    """Orchestrates Hamiltonian prep and VQE; run-before-read contract."""

    def __init__(
        self,
        n_qubits,
        ham,
        is_classical,
        initial_state,
        reps,
        electron_fraction,
        generalized,
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
        # ham is retained for kcl_util_UCCSD signature parity; get_result loads by seed from disk.
        self.ham = ham
        self.is_classical = bool(is_classical)
        self.initial_state = initial_state
        self.reps = reps
        self.electron_fraction = electron_fraction
        self.generalized = generalized
        self.optimizer_choice = optimizer_choice
        self.iter_max = iter_max
        self.coeff_cutoff = coeff_cutoff
        self.atol = atol
        self.shots = shots
        self.gradient_method = gradient_method
        self.initial_point_seed = initial_point_seed
        self.max_evals_grouped = max_evals_grouped
        self._result: Optional[Result_UCCSD] = None

    def _prepare_operator(self, seed, ham_dir):
        """Load, compress, and Jordan–Wigner-map the Hamiltonian for this run."""
        ham = load_hamiltonian(self.n_qubits, seed, ham_dir)
        ham = compress_hamiltonian(ham, self.coeff_cutoff)
        return to_qubit_operator(ham)

    def _make_ansatz(self) -> Ansatz_UCCSD:
        """Build Ansatz_UCCSD from stored structural hyperparameters."""
        return Ansatz_UCCSD(
            self.n_qubits,
            self.initial_state,
            self.reps,
            self.electron_fraction,
            self.generalized,
        )

    def _make_solver(self, operator, ansatz) -> Solver_UCCSD:
        """Build Solver_UCCSD from stored procedural knobs."""
        return Solver_UCCSD(
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
        )

    def get_result(self, seed, ham_dir) -> float:
        """Run VQE once (run-before-read); raises if called twice on the same instance."""
        if self._result is not None:
            raise RuntimeError(
                "get_result() has already been called on this Wrapper instance. "
                "Create a new Wrapper for a new run."
            )

        operator = self._prepare_operator(seed, ham_dir)
        ansatz = self._make_ansatz()
        solver = self._make_solver(operator, ansatz)
        self._result = solver.run()
        return self._result.energy

    def _build_x_vec(self) -> np.ndarray:
        return np.array(
            [
                float(self.n_qubits),
                float(self.is_classical),
                float(self.initial_state),
                float(self.reps),
                float(self.electron_fraction),
                float(self.generalized),
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

    def result_for_evaluation(self) -> Result_UCCSD:
        if self._result is None:
            raise NotRunError(
                "get_result() must be called before accessing results. "
                "VQE execution is deliberate and not triggered automatically."
            )
        return self._result