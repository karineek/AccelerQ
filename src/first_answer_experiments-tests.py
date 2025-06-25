# FOR EVALUATION ONLY
import sys
from datetime import datetime

from typing import Any
from itertools import combinations
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.circuit.transpile import RZSetTranspiler
from quri_parts.core.operator import (
    pauli_label,
    Operator,
    PauliLabel,
    pauli_product,
    PAULI_IDENTITY,
)
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    pauli_label_to_bsv,
    transition_amp_representation,
    transition_amp_comp_basis,
)
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from random import randint

from qiskit import quantum_info

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian

challenge_sampling = ChallengeSampling()

def print_to_file(message):
    with open("logger-final.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def print_to_file_itr(message):
    with open("logger-itr.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

class Solver:
    def __init__(
        self,
        is_classical,
        hamiltonian: Operator,
        pool: list[Operator],
        n_qubits: int,
        n_ele_cas: int,
        sampler,
        iter_max: int = 10,
        sampling_shots: int = 10**4,
        post_selected: bool = True,
        atol: float = 1e-5,
        round_op_config: Union[dict, Tuple[Optional[int], Optional[float]]] = (None, 1e-2),
        num_precise_gradient=None,
        max_num_converged: int = 1,
        final_sampling_shots_coeff: float = 1.0,
        check_duplicate: bool = True,
        reset_ignored_inx_mode: int = 10,
    ):
        round_ham = round_hamiltonian(hamiltonian, 100, 0.001)
        self.is_classical = is_classical
        self.hamiltonian: Operator = hamiltonian
        self.pool: list[Operator] = pool
        self.n_qubits: int = n_qubits
        self.n_ele_cas: int = n_ele_cas
        self.iter_max: int = iter_max
        self.sampling_shots: int = sampling_shots
        self.atol: float = atol
        self.sampler = sampler
        self.sv_sampler = create_qulacs_vector_sampler()
        self.post_selected: bool = post_selected
        self.check_duplicate: bool = check_duplicate
        # initialization
        hf_state = ComputationalBasisState(self.n_qubits, bits=2 ** self.n_ele_cas - 1)
        self.hf_state = hf_state
        self.comp_basis = [hf_state]
        # gradient
        if round_op_config is None:
            round_op_config = (None, None)
        num_pickup: int = round_op_config["num_pickup"] if isinstance(round_op_config, dict) else round_op_config[0]
        coeff_cutoff: float = round_op_config["cutoff"] if isinstance(round_op_config, dict) else round_op_config[1]
        self.num_pickup = num_pickup
        self.coeff_cutoff = coeff_cutoff
        round_ham = round_hamiltonian(hamiltonian, num_pickup=num_pickup, coeff_cutoff=coeff_cutoff)
        self.round_hamiltonian = round_ham
        self._is_grad_round: bool = not (num_pickup is None and coeff_cutoff is None)
        self.gradient_pool: List[Operator] = [commutator(round_ham, op) for op in pool]
        self.precise_grad_vals_mem: dict = {}
        self.gradient_vector_history = []
        self.num_precise_gradient: int = len(pool) if num_precise_gradient is None else num_precise_gradient
        self.pauli_rotation_circuit_qsci = PauliRotationCircuit([], [], [], n_qubits)
        self.ignored_gen_inx = []
        self.reset_ignored_inx_mode: int = reset_ignored_inx_mode if reset_ignored_inx_mode > 0 else iter_max
        # convergence
        assert max_num_converged >= 1
        self.final_sampling_shots: int = int(final_sampling_shots_coeff * sampling_shots)
        self.max_num_converged: int = max_num_converged
        self.num_converged: int = 0
        # results
        self.qsci_energy_history: list = []
        self.opt_energy_history: list = []
        self.operator_index_history: list = []
        self.gradient_history: list = []
        self.param_values: list = []
        self.raw_energy_history = []
        self.sampling_results_history = []
        self.comp_basis_history = []
        self.opt_param_value_history = []
        self.generator_history: list = []
        self.generator_qubit_indices_history: list = []

        if self.num_precise_gradient > len(pool):
            self.num_precise_gradient = len(pool)



    # Modify ADD VQE
    def _get_optimized_parameter(
        self, vec_qsci: np.ndarray, comp_basis: list[ComputationalBasisState]
    ) -> float:
        generator_qp = self.pool[self.operator_index_history[-1]]
        ham_sparse = generate_truncated_hamiltonian(self.hamiltonian, comp_basis)
        commutator_sparse = generate_truncated_hamiltonian(
            1j * self.precise_grad_vals_mem[self.operator_index_history[-1]], comp_basis
        )
        exp_h = (vec_qsci.T.conj() @ ham_sparse @ vec_qsci).item().real
        exp_commutator = (
            (vec_qsci.T.conj() @ commutator_sparse @ vec_qsci).item().real
        )
        php = generator_qp * self.hamiltonian * generator_qp
        php_sparse = generate_truncated_hamiltonian(php, comp_basis)
        exp_php = (vec_qsci.T.conj() @ php_sparse @ vec_qsci).item().real
        cost_e2 = (
            lambda x: exp_h * np.cos(x[0]) ** 2
            + exp_php * np.sin(x[0]) ** 2
            + exp_commutator * np.cos(x[0]) * np.sin(x[0])
        )
        result_qsci = scipy.optimize.minimize(
            cost_e2, np.array([0.0]), method="BFGS", options={"disp": False, "gtol": 1e-6}
        )
        try:
            assert result_qsci.success
        except:
            print("try optimization again...")
            result_qsci = scipy.optimize.minimize(
                cost_e2, np.array([0.1]), method="BFGS", options={"disp": False, "gtol": 1e-6}
            )
            if not result_qsci.success:
                print("*** Optimization failed, but we continue calculation. ***")
        print(f"θ: {result_qsci.x}")
        return float(result_qsci.x)

    def run(self) -> float:
        vec_qsci, val_qsci = diagonalize_effective_ham(self.hamiltonian, self.comp_basis)
        self.qsci_energy_history.append(val_qsci)
        for itr in range(1, self.iter_max + 1):
            print(f"iteration: {itr}")
            print_to_file_itr(f"iteration: {itr}")
            grad_vals = np.zeros(len(self.pool), dtype=float)
            for j, grad in enumerate(self.gradient_pool):
                grad_mat = generate_truncated_hamiltonian(1j * grad, self.comp_basis)
                grad_vals[j] = (vec_qsci.T @ grad_mat @ vec_qsci).real
            sorted_indices = np.argsort(np.abs(grad_vals))[::-1]

            # find largest index of generator
            precise_grad_vals = {}
            if self.num_precise_gradient is not None and self._is_grad_round:
                # calculate the precise value of gradient
                for i_ in list(sorted_indices):
                    if i_ not in self.ignored_gen_inx:
                        if i_ in self.precise_grad_vals_mem.keys():
                            grad = self.precise_grad_vals_mem[i_]
                        else:
                            grad = commutator(self.hamiltonian, self.pool[i_])
                            self.precise_grad_vals_mem[i_] = grad
                        grad_val = (
                            vec_qsci.T
                            @ generate_truncated_hamiltonian(1j * grad, self.comp_basis)
                            @ vec_qsci
                        )
                        precise_grad_vals[i_] = grad_val
                    else:
                        pass
                    if len(precise_grad_vals.keys()) >= self.num_precise_gradient:
                        break
                # print(precise_grad_vals)
                sorted_keys = sorted(precise_grad_vals.keys(), key=lambda x: abs(precise_grad_vals[x]), reverse=True)
                # print(len(sorted_keys),self.num_precise_gradient)
                # assert len(sorted_keys) == self.num_precise_gradient

                # select generator whose abs. gradient is second largest when same generator is selected twice in a row
                if self.check_duplicate:
                    if (len(self.operator_index_history) >= 1 and len(sorted_keys) >= 2) and \
                            (sorted_keys[0] == self.operator_index_history[-1]):
                        largest_index: int = sorted_keys[1]
                        print("selected second largest gradient")
                        self.ignored_gen_inx.append(sorted_keys[0])
                        print(f"index {sorted_keys[0]} added to ignored list")
                    else:
                        largest_index: int = sorted_keys[0]
                else:
                    largest_index = sorted_indices[0]
                grad_vals = precise_grad_vals.values()
                print(
                    f"new generator: {str(self.pool[largest_index]).split('*')}, index: {largest_index} "
                    f"out of {len(self.pool)}. # precise gradient: {self.num_precise_gradient}"
                )
                self.gradient_vector_history.append(key_sortedabsval(precise_grad_vals))
            else:
                largest_index = sorted_indices[0]
            self.operator_index_history.append(largest_index)
            self.gradient_history.append(np.abs(max(grad_vals)))
            operator_coeff_term = str(self.pool[largest_index]).split("*")
            new_coeff, new_pauli_str = float(operator_coeff_term[0]), operator_coeff_term[1]
            self.generator_history.append(new_pauli_str)

            # add new generator to ansatz
            new_param_name = f"theta_{itr}"
            circuit_qsci = self.pauli_rotation_circuit_qsci.add_new_gates(new_pauli_str, new_coeff, new_param_name)
            new_param_value = self._get_optimized_parameter(vec_qsci, self.comp_basis)
            if np.isclose(new_param_value, 0.):
                self.ignored_gen_inx.append(largest_index)
                print(f"index {largest_index} added to ignored list")
            self.opt_param_value_history.append(new_param_value)
            if self.pauli_rotation_circuit_qsci.fusion_mem:
                self.param_values[
                    self.pauli_rotation_circuit_qsci.fusion_mem[0]
                ] += new_param_value
            else:
                if np.isclose(0.0, new_param_value):
                    circuit_qsci = self.pauli_rotation_circuit_qsci.delete_newest_gate()
                else:
                    self.param_values.append(new_param_value)
            try:
                new_gen_indices = sorted(circuit_qsci.gates[-1].target_indices)
            except IndexError:
                print(f"ansatz seems to have no gates since optimized parameter was {new_param_value}")
                raise

            # increase sampling shots when same generator is selected twice in a row or parameter is close to 0.
            is_alert = new_gen_indices in self.generator_qubit_indices_history or np.isclose(0.0, new_param_value)
            self.generator_qubit_indices_history.append(new_gen_indices)
            sampling_shots = self.final_sampling_shots if is_alert else self.sampling_shots

            # prepare circuit for QSCI
            parametric_state_qsci = prepare_parametric_state(self.hf_state, circuit_qsci)
            target_circuit = parametric_state_qsci.parametric_circuit.bind_parameters(self.param_values)
            transpiled_circuit = RZSetTranspiler()(target_circuit)
            
            if self.is_classical:
                counts = self.sv_sampler(transpiled_circuit, shots=sampling_shots)
                pass

            else:
                # QSCI
                try:
                    "Using quantum resources"
                    counts = self.sampler(transpiled_circuit, sampling_shots)
                except ExceededError as e:
                    print(str(e))
                    return min(self.qsci_energy_history)
            self.comp_basis = pick_up_bits_from_counts(
                counts=counts,
                n_qubits=self.n_qubits,
                R_max=num_basis_symmetry_adapted_cisd(self.n_qubits),
                threshold=1e-10,
                post_select=self.post_selected,
                n_ele=self.n_ele_cas,
            )
            self.sampling_results_history.append(counts)
            self.comp_basis_history.append(self.comp_basis)
            vec_qsci, val_qsci = diagonalize_effective_ham(
                self.hamiltonian, self.comp_basis
            )
            self.qsci_energy_history.append(val_qsci)
            # print(f"basis selected: {[bin(b.bits)[2:].zfill(self.n_qubits) for b in self.comp_basis]}")
            print(f"QSCI energy: {val_qsci}, (new generator {new_pauli_str})")
            print_to_file_itr(f"QSCI energy: {val_qsci}, (new generator {new_pauli_str})") 
            
            # terminate condition
            if (
                abs(self.qsci_energy_history[-2] - self.qsci_energy_history[-1])
                < self.atol
            ):
                self.num_converged += 1
                if self.num_converged == self.max_num_converged:
                    break
                else:
                    continue

            # empty ignored index list periodically
            if itr % self.reset_ignored_inx_mode == 0:
                print(f"ignored list emptied: {self.ignored_gen_inx} -> []")
                self.ignored_gen_inx = []
        return min(self.qsci_energy_history)


class PauliRotationCircuit:
    def __init__(
        self, generators: list, coeffs: list, param_names: list, n_qubits: int
    ):
        self.generators: list = generators
        self.coeffs: list = coeffs
        self.param_names: list = param_names
        self.n_qubits: int = n_qubits
        self.fusion_mem: list = []
        self.generetors_history: list = []

    def __call__(self):
        return self.construct_circuit()

    def construct_circuit(
        self, generators=None
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        circuit = LinearMappedUnboundParametricQuantumCircuit(self.n_qubits)
        if generators is None:
            generators = self.generators
        for generator, coeff, name in zip(generators, self.coeffs, self.param_names):
            param_name = circuit.add_parameter(name)
            if isinstance(generator, str):
                generator = pauli_label(generator)
            else:
                raise
            pauli_index_list, pauli_id_list = zip(*generator)
            coeff = coeff.real
            circuit.add_ParametricPauliRotation_gate(
                pauli_index_list,
                pauli_id_list,
                {param_name: -2.0 * coeff},
            )
        return circuit

    def add_new_gates(
        self, generator: str, coeff: float, param_name: str
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        self._reset()
        self.generetors_history.append(generator)
        for i, (g, n) in enumerate(zip(self.generators[::-1], self.param_names[::-1])):
            if is_equivalent(generator, g):
                self.fusion_mem = [-i]
                print(f"FUSED: {g, generator}")
                break
            elif is_commute(generator, g):
                continue
            else:
                break
        if not self.fusion_mem:
            self.generators.append(generator)
            self.coeffs.append(coeff)
            self.param_names.append(param_name)
        return self.construct_circuit()

    def delete_newest_gate(self) -> LinearMappedUnboundParametricQuantumCircuit:
        self._reset()
        self.generators = self.generators[:-1]
        self.coeffs = self.coeffs[:-1]
        self.param_names = self.param_names[:-1]
        return self.construct_circuit()

    def _reset(self):
        self.fusion_mem = []


def diagonalize_effective_ham(
    ham_qp: Operator, comp_bases_qp: list[ComputationalBasisState]
) -> Tuple[np.ndarray, np.ndarray]:
    effective_ham_sparse = generate_truncated_hamiltonian(ham_qp, comp_bases_qp)
    assert np.allclose(effective_ham_sparse.todense().imag, 0)
    effective_ham_sparse = effective_ham_sparse.real
    if effective_ham_sparse.shape[0] > 10:
        eig_qsci, vec_qsci = scipy.sparse.linalg.eigsh(
            effective_ham_sparse, k=1, which="SA"
        )
        eig_qsci = eig_qsci.item()
        vec_qsci = vec_qsci.squeeze()
    else:
        eig_qsci, vec_qsci = np.linalg.eigh(effective_ham_sparse.todense())
        eig_qsci = eig_qsci[0]
        vec_qsci = np.array(vec_qsci[:, 0])

    return vec_qsci, eig_qsci

# Modify
def generate_truncated_hamiltonian(
    hamiltonian: Operator,
    states: Sequence[ComputationalBasisState],
) -> scipy.sparse.spmatrix:
    """Generate truncated Hamiltonian on the given basis states."""
    dim = len(states)
    values = []
    row_ids = []
    column_ids = []
    h_transition_amp_repr = transition_amp_representation(hamiltonian)
    for m in range(dim):
        for n in range(m, dim):
            mn_val = transition_amp_comp_basis(
                h_transition_amp_repr, states[m].bits, states[n].bits
            )
            if mn_val:
                values.append(mn_val)
                row_ids.append(m)
                column_ids.append(n)
                if m != n:
                    values.append(mn_val.conjugate())
                    row_ids.append(n)
                    column_ids.append(m)
    truncated_hamiltonian = coo_matrix(
        (values, (row_ids, column_ids)), shape=(dim, dim)
    ).tocsc(copy=False)
    truncated_hamiltonian.eliminate_zeros()

    return truncated_hamiltonian


def _add_term_from_bsv(
    bsvs: List[List[Tuple[int, int]]], ops: List[Operator]
) -> Operator:
    ret_op = Operator()
    op0_bsv, op1_bsv = bsvs[0], bsvs[1]
    op0, op1 = ops[0], ops[1]
    for i0, (pauli0, coeff0) in enumerate(op0.items()):
        for i1, (pauli1, coeff1) in enumerate(op1.items()):
            bitwise_string = str(
                bin(
                    (op0_bsv[i0][0] & op1_bsv[i1][1])
                    ^ (op0_bsv[i0][1] & op1_bsv[i1][0])
                )
            )
            if bitwise_string.count("1") % 2 == 1:
                pauli_prod_op, pauli_prod_phase = pauli_product(pauli0, pauli1)
                tot_coef = 2 * coeff0 * coeff1 * pauli_prod_phase
                ret_op.add_term(pauli_prod_op, tot_coef)
    return ret_op


def pauli_string_to_bsv(pauli_str: str) -> BinarySymplecticVector:
    return pauli_label_to_bsv(pauli_label(pauli_str))


def get_bsv(pauli: Union[PauliLabel, str]) -> BinarySymplecticVector:
    if isinstance(pauli, str):
        bsv = pauli_string_to_bsv(pauli)
    else:
        bsv = pauli_label_to_bsv(pauli)
    return bsv


def is_commute(pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    x1_z2 = bsv1.x & bsv2.z
    z1_x2 = bsv1.z & bsv2.x
    is_bitwise_commute_str = str(bin(x1_z2 ^ z1_x2)).split("b")[-1]
    return sum(int(b) for b in is_bitwise_commute_str) % 2 == 0


def is_equivalent(
    pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]
) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    return bsv1.x == bsv2.x and bsv1.z == bsv2.z


def operator_bsv(op: Operator) -> List[Tuple[int, int]]:
    ret = []
    for pauli in op.keys():
        bsv_pauli = get_bsv(pauli)
        ret.append((bsv_pauli.x, bsv_pauli.z))
    return ret


def round_hamiltonian(op: Operator, num_pickup: int = None, coeff_cutoff: float = None):
    ret_op = Operator()
    if coeff_cutoff in [None, 0.0] and num_pickup is None:
        return op
    sorted_pauli = sorted(op.keys(), key=lambda x: abs(op[x]), reverse=True)
    if num_pickup is not None:
        sorted_pauli = sorted_pauli[:num_pickup]
    if coeff_cutoff is None:
        coeff_cutoff = 0
    for pauli in sorted_pauli:
        coeff = op[pauli]
        if abs(coeff) < coeff_cutoff:
            pass
        else:
            ret_op += Operator({pauli: coeff})
    return ret_op


def commutator(
    op0: Union[Operator, float, int, complex], op1: Union[Operator, float, int, complex]
) -> Operator:
    if not isinstance(op0, Operator) or not isinstance(op1, Operator):
        return Operator({PAULI_IDENTITY: 0.0})
    else:
        assert isinstance(op0, Operator) and isinstance(op1, Operator)
        op0_bsv = operator_bsv(op0)
        op1_bsv = operator_bsv(op1)
        ret_op = _add_term_from_bsv([op0_bsv, op1_bsv], [op0, op1])
        return ret_op


def prepare_parametric_state(initial_state, ansatz):
    circuit = LinearMappedUnboundParametricQuantumCircuit(initial_state.qubit_count)
    circuit += initial_state.circuit
    circuit += ansatz
    return ParametricCircuitQuantumState(initial_state.qubit_count, circuit)


def key_sortedabsval(data: Union[list, dict, np.ndarray], round_: int = 5) -> dict:
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys(), key=lambda x: abs(data[x]), reverse=True)
    else:
        sorted_keys = np.argsort(np.abs(data))[::-1]
    ret_dict = {}
    for k in sorted_keys:
        val = float(data[int(k)].real)
        assert np.isclose(val.imag, 0.0)
        ret_dict[int(k)] = round(val, round_)
    return ret_dict

# CHANGE
def create_qubit_adapt_pool_XY_XXXY(
    n_qubits,
    use_singles: bool = False,
    single_excitation_dominant: bool = False,
    double_excitation_dominant: bool = False,
    mode: list[int] = None,
    n_electrons: int = None,
) -> list[Operator]:
    operator_pool_qubit = []
    if use_singles:
        for p, q in combinations(range(n_qubits), 2):
            if single_excitation_dominant and not (p < n_electrons <= q):
                continue
            operator_pool_qubit.append(QubitOperator(f"X{p} Y{q}", 1.0))
    if mode is None:
        mode = [0, 1, 2, 3]
    for m in mode:
        assert m in [0, 1, 2, 3, 4]
        if m == 4:
            mode = [4]
            break
    for p, q, r, s in combinations(range(n_qubits), 4):
        if double_excitation_dominant and not (q < n_electrons <= r):
            continue
        for m in mode:
            x_index = m if m in [0, 1, 2, 3] else randint(0, 3)
            p_list = ["Y" if _ == x_index else "X" for _ in range(4)]
            gen_string_list = " ".join(
                [f"{p}{i}" for p, i in zip(p_list, (p, q, r, s))]
            )
            operator_pool_qubit.append(QubitOperator(gen_string_list, 1.0))
    operator_pool_qubit = [
        operator_from_openfermion_op(op) for op in operator_pool_qubit
    ]
    return operator_pool_qubit


def num_basis_symmetry_adapted_cisd(n_qubits: int):
    return (n_qubits**4 - 4 * n_qubits**3 + 20 * n_qubits**2 + 64) // 64


def pick_up_bits_from_counts(
    counts: Mapping[int, Union[int, float]],
    n_qubits,
    R_max=None,
    threshold=None,
    post_select=False,
    n_ele=None,
):
    sorted_keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    if threshold is None:
        heavy_bits = sorted_keys
    else:
        heavy_bits = [bit for bit in sorted_keys if counts[bit] >= threshold]
    if post_select:
        assert n_ele is not None
        heavy_bits = [i for i in heavy_bits if bin(i).count("1") == n_ele]
    if R_max is not None:
        heavy_bits = heavy_bits[:R_max]
    comp_bases_qp = [
        ComputationalBasisState(n_qubits, bits=int(key)) for key in heavy_bits
    ]
    return comp_bases_qp


class Wrapper:
    def __init__(self, number_qubits, ham, is_classical, use_singles, num_pickup, coeff_cutoff, self_selection, iter_max, sampling_shots, atol, final_sampling_shots_coeff, num_precise_gradient, max_num_converged, reset_ignored_inx_mode) -> None:
        challenge_sampling.reset()

        self.number_qubits = number_qubits
        self.ham = ham
        self.is_classical = is_classical #use SV solver
        self.use_singles = use_singles #include single excitations in operator pool
        self.num_pickup = num_pickup #retain largest N terms in Hamiltonian
        self.coeff_cutoff = coeff_cutoff #cutoff all terms smaller than this from the num_pickup terms remaining
        self.post_selection = self_selection #force it to work in subspace with correctr number of 1s and 0s
        self.iter_max = iter_max #max total iterations
        self.sampling_shots = sampling_shots #how many shots to use per iteration
        self.atol = atol # the tolerance at which we say it is converged
        self.final_sampling_shots_coeff = final_sampling_shots_coeff #how many more shots to use in the calculatino if the same operator appears twice or the operator parameter is close to zero
        self.num_precise_gradient = num_precise_gradient #how many operators from pool to calculate gradient more precisely
        self.max_num_converged = max_num_converged #how many iterations does it need to stay within atol to be considered converged
        self.reset_ignored_inx_mode = reset_ignored_inx_mode #after how many iterations do we allow previously used operators to be used again


    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = self.number_qubits
        ham = self.ham
        n_electrons = n_qubits // 2
        use_singles = self.use_singles
        jw_hamiltonian = jordan_wigner(ham)
        qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        num_pickup, coeff_cutoff = self.num_pickup, self.coeff_cutoff
        post_selection = self.post_selection
        mps_sampler = challenge_sampling.create_sampler()
        pool = create_qubit_adapt_pool_XY_XXXY(
            n_qubits,
            use_singles=use_singles,
            single_excitation_dominant=True,
            double_excitation_dominant=True,
            mode=[4],
            n_electrons=n_electrons,
        )

        solver = Solver(
            self.is_classical,
            qp_hamiltonian,
            pool,
            n_qubits=n_qubits,
            n_ele_cas=n_electrons,
            sampler=mps_sampler,
            iter_max=self.iter_max,
            post_selected=post_selection,
            sampling_shots=self.sampling_shots,
            atol=self.atol,
            final_sampling_shots_coeff=self.final_sampling_shots_coeff,
            round_op_config=(num_pickup, coeff_cutoff),
            num_precise_gradient=self.num_precise_gradient,
            max_num_converged=self.max_num_converged,
            check_duplicate=True,
            reset_ignored_inx_mode=self.reset_ignored_inx_mode,
        )
        res = solver.run()
        return res

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self,n_qubits: int, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        # Get the current date and time
        now = datetime.now()
 
        # Format the time as a string
        current_time_s = now.strftime("%H:%M:%S")
 
        # Print the current time
        print("Current Time:", current_time_s)

        # n_qubits = 4
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)

        ###is_classical, binary
        ###use_singles, binary
        ###num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
        ###coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
        ###self_selection, binary
        ###iter_max, int definitely > 1 (want large)
        ###sampling_shots, int definitely >1 probably want fairly large, at least 100 
        ###atol, float definitely >0 and < 1, probably < 1e-3 
        ###final_sampling_shots_coeff, int definitely > 0 and probably < 10
        ###num_precise_gradient, int definitely >0 
        ###max_num_converged, int definitely > 1
        ###reset_ignored_inx_mode, int deffinitely >=0

        is_OPT = True
        if is_OPT:
            print ("Optimised params")
            # Get better hyper-params
            if n_qubits == 20:
                if (seed == 0):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           4.90000000000000000e+02, 1.34913612892553899e-03, 0.00000000000000000e+00,
                           9.10000000000000000e+01, 5.50451000000000000e+05, 1.29522457908607893e-05,
                           5.00000000000000000e+00, 2.20000000000000000e+02, 3.00000000000000000e+00,
                           2.70000000000000000e+01 ] # 20,0, Number of max iteration: 91
                        , dtype=float)

                elif (seed == 1):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           3.43000000000000000e+02, 9.84194749365019557e-03, 0.00000000000000000e+00,
                           1.80500000000000000e+02, 1.01383000000000000e+05, 8.03531130333837135e-05,
                           5.00000000000000000e+00, 2.94000000000000000e+02, 3.00000000000000000e+00,
                           4.80000000000000000e+01 ] # 20,1, Number of max iteration: 180
                        , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           1.49000000000000000e+02, 9.79684582097996998e-03, 0.00000000000000000e+00,
                           1.51000000000000000e+02, 1.00000000000000000e+05, 9.99999999999999955e-07,
                           5.00000000000000000e+00, 2.91000000000000000e+02, 2.00000000000000000e+00,
                           7.00000000000000000e+01 ] # 20,2, Number of max iteration: 151
                        , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                         [2.000000000000000e+01, 1.000000000000000e+00, 1.000000000000000e+00
                           , 5.240000000000000e+02, 7.093479841027719e-03, 0.000000000000000e+00
                           , 2.265910000000000e+05, 7.665400000000000e+04, 4.395152474144069e-05
                           , 4.000000000000000e+00, 6.900000000000000e+01, 2.000000000000000e+00
                           , 3.600000000000000e+01]  # 20,3
                        , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           9.10000000000000000e+01, 1.00000000000000002e-03, 0.00000000000000000e+00,
                           1.00000000000000000e+02, 9.49570000000000000e+04, 9.99999999999999955e-07,
                           5.00000000000000000e+00, 1.28000000000000000e+02, 2.00000000000000000e+00,
                           3.35000000000000000e+01 ] # 20,4, Number of max iteration: 100
                        , dtype=float)

                elif (seed == 5):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           5.30000000000000000e+01, 9.90279593077902399e-03, 0.00000000000000000e+00,
                           1.10000000000000000e+01, 2.21290100000000000e+06, 5.45371793030312744e-05,
                           7.00000000000000000e+00, 3.00000000000000000e+02, 4.00000000000000000e+00,
                           8.60000000000000000e+01 ] # 20,5, Number of max iteration: 11
                        , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 24:
                if (seed == 5):
                    res_opt = np.array(
                          [ 2.40000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
  5.60000000000000000e+01, 4.30962367155318139e-03, 0.00000000000000000e+00,
  4.80000000000000000e+01, 2.17535000000000000e+05, 7.89079483788077124e-07,
  4.00000000000000000e+00, 2.96000000000000000e+02, 3.00000000000000000e+00,
  5.00000000000000000e+01 ] # 24,5
                        , dtype=float)

                elif (seed == 6):
                    res_opt = np.array(
                          [ 2.40000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
  7.70000000000000000e+01, 8.72720940726242483e-03, 0.00000000000000000e+00,
  2.99000000000000000e+02, 2.25504000000000000e+05, 2.68932724972004344e-05,
  7.00000000000000000e+00, 2.84000000000000000e+02, 2.50000000000000000e+00,
  7.52500000000000000e+01 ] # 24,6
                        , dtype=float)

                elif (seed == 7):
                    res_opt = np.array(
                          [ 2.40000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
  6.80000000000000000e+01, 6.95752548589043812e-03, 0.00000000000000000e+00,
  4.23000000000000000e+02, 2.23761000000000000e+05, 3.81985639999710894e-06,
  5.00000000000000000e+00, 2.68000000000000000e+02, 4.00000000000000000e+00,
  3.20000000000000000e+01  ] # 24,7
                        , dtype=float)

                elif (seed == 8):
                    res_opt = np.array(
                          [
                           2.40000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           7.70000000000000000e+01, 9.73544552054709904e-03, 0.00000000000000000e+00,
                           2.30000000000000000e+01, 9.48419000000000000e+05, 3.53935303238853561e-05,
                           7.00000000000000000e+00, 2.78000000000000000e+02, 4.00000000000000000e+00,
                           3.30000000000000000e+01 ] # 24,8, Number of max iteration: 23
                        , dtype=float)

                elif (seed == 9):
                    res_opt = np.array(
                          [
                           2.40000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           6.60000000000000000e+01, 5.26840492957128884e-03, 0.00000000000000000e+00,
                           3.55625000000000000e+01, 1.37122750000000000e+05, 1.82683796039928472e-05,
                           3.00000000000000000e+00, 2.35000000000000000e+02, 4.00000000000000000e+00,
                           4.78750000000000000e+01 ] # 24,9, Number of max iteration: 36
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 28:
                if (seed == 0):
                    res_opt = np.array(
                          [
                           2.80000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           4.60000000000000000e+02, 9.77681428831346511e-03, 0.00000000000000000e+00,
                           8.00000000000000000e+01, 8.93780000000000000e+04, 7.75972442913625525e-05,
                           9.00000000000000000e+00, 2.90000000000000000e+02, 3.00000000000000000e+00,
                           3.00000000000000000e+01 ] # 28,0, Number of max iteration: 80
                       , dtype=float)

                elif (seed == 1):
                    res_opt = np.array(
                          [
                           2.80000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           2.38000000000000000e+02, 3.44905933405150974e-05, 0.00000000000000000e+00,
                           5.65000000000000000e+01, 1.29065562500000000e+05, 3.03710472299496162e-07,
                           4.00000000000000000e+00, 2.26875000000000000e+02, 3.00000000000000000e+00,
                           7.17500000000000000e+01 ] # 28,1, Number of max iteration: 56
                       , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [
                           2.80000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           9.60000000000000000e+01, 1.88585673979859824e-03, 0.00000000000000000e+00,
                           4.70000000000000000e+01, 8.59890000000000000e+04, 1.26974693345332292e-06,
                           5.00000000000000000e+00, 1.62000000000000000e+02, 3.00000000000000000e+00,
                           2.65000000000000000e+01 ] # 28,2, Number of max iteration: 47
                       , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [
                           2.80000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           1.79000000000000000e+02, 1.87003739117375112e-03, 0.00000000000000000e+00,
                           3.70000000000000000e+01, 1.31364000000000000e+05, 1.86638155644660054e-05,
                           2.50000000000000000e+00, 2.31000000000000000e+02, 4.00000000000000000e+00,
                           2.90000000000000000e+01 ] # 28,3, Number of max iteration: 37
                       , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [
                           2.80000000000000000e+01, 1.00000000000000000e+00, 1.00000000000000000e+00,
                           3.55000000000000000e+02, 9.92939167035086605e-03, 0.00000000000000000e+00,
                           7.40000000000000000e+01, 9.34060000000000000e+04, 9.40621635672334695e-05,
                           5.00000000000000000e+00, 2.59000000000000000e+02, 2.00000000000000000e+00,
                           1.90000000000000000e+01 ] # 28,4, Number of max iteration: 74
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            else:
                print ("EERROR n_qubits is : " + str(n_qubits))
                return 0

            param_bool_2 = True if round(res_opt[5]) != 0 else False
            wrapper=Wrapper(n_qubits, ham, False, True, round(res_opt[3]), res_opt[4], param_bool_2,
                      # n_qubits, ham, True, True, 100,               0.001,      False,
                            round(res_opt[6]), round(res_opt[7]), res_opt[8],
                      #     100,               10**5,             1e-6,
                            round(res_opt[9]), round(res_opt[10]), 4, round(res_opt[12]))
                      #     5,                 128,                2,                  0

        else:
            print ("Default params")
            res_opt = [False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0]
            # The call to the VQE
            wrapper = Wrapper(n_qubits, ham, res_opt[0], res_opt[1], res_opt[2], res_opt[3], res_opt[4], res_opt[5], res_opt[6], res_opt[7], res_opt[8], res_opt[9], res_opt[10], res_opt[11])


        # wrapper = Wrapper(n_qubits, ham, True, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0)
        res=wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")
        print("type: ",type(res))
        """
        ####################################
        add codes here
        ####################################
        """
        # Get the current date and time
        now = datetime.now()
 
        # Format the time as a string
        current_time = now.strftime("%H:%M:%S")
 
        # Print the current time
        print("Current Time:", current_time)
        print_to_file(">>> Start | " + str(current_time_s) + " | OPT? | " + str(is_OPT) + " | seed | " + str(seed) + " | n_qubits  | " + str(n_qubits) + " | res | " + str(res))
        return res


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(n_qubits=24,seed=6, hamiltonian_directory="../hamiltonian"))
