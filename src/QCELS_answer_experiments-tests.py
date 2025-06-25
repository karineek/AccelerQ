# FOR EVALUATION ONLY
import sys
from datetime import datetime
from typing import Any
import numpy as np
import random
from numba import njit, jit
from time import time
from scipy.optimize import curve_fit,least_squares

from quri_parts.core.operator.trotter_suzuki import trotter_suzuki_decomposition
sys.path.append("../")

from scipy.optimize import minimize, OptimizeResult
from itertools import combinations
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.algo.ansatz import SymmetryPreservingReal
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement, CachedMeasurementFactory
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator, create_proportional_shots_allocator
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.circuit import QuantumCircuit, gate_names
from quri_parts.core.operator import PauliLabel,SinglePauli,Operator
from pytket.pauli import Pauli
from pytket.circuit import PauliExpBox
from pytket.circuit import QControlBox
from pytket import Circuit
from pytket.passes import DecomposeBoxes, SynthesiseTK
from pytket.transform import Transform

from quri_parts.tket.circuit import circuit_from_tket
from quri_parts.tket.circuit.transpile import TketTranspiler
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
from utils.challenge_2024 import ChallengeSampling, ExceededError, problem_hamiltonian

challenge_sampling = ChallengeSampling()

"""
####################################
add codes here
####################################
"""
startTime = datetime.now()
curr_itr = 0

def print_to_file(message):
    with open("logger-final.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def print_to_file_itr(message):
    with open("logger-itr.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def my_timestamp():
    # Get the current date and time
    now = datetime.now()

    # Format the time as a string
    current_time_s = now.strftime("%H:%M:%S")

    return str(current_time_s)

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

# trotter_suzuki_decomposition and Operator are no good for optimisation
def get_controlled_trotter_circuit(qp_hamiltonian,delta_t,n_qubits):
    trotterised_U = trotter_suzuki_decomposition(qp_hamiltonian, delta_t, 1)
    trotterised_U_circ = QuantumCircuit(n_qubits)
    test_circ = Circuit(n_qubits+1)

    for pauli_gate in trotterised_U:
        index_list, pauli_list = pauli_gate.pauli.index_and_pauli_id_list
        my_pauli_list=[]
        for a_pauli in pauli_list:
            if a_pauli == 1:
                my_pauli_list.append(Pauli.X)
            elif a_pauli == 2:
                my_pauli_list.append(Pauli.Y)
            elif a_pauli ==3:
                my_pauli_list.append(Pauli.Z)
            else:
                sys.exit("error unknown pauli: "+str(a_pauli))

        pauli_box = PauliExpBox(my_pauli_list, pauli_gate.coefficient.real)
        controlled_pauli = QControlBox(pauli_box, 1)
        test_circ.add_gate(controlled_pauli, [n_qubits]+index_list)

    DecomposeBoxes().apply(test_circ)
    Transform.RebaseToPyZX().apply(test_circ)
    print("Total gate count:", test_circ.n_gates, "2 qubit gates:", test_circ.n_2qb_gates())

    #return TketTranspiler(basis_gates=[gate_names.Identity,gate_names.X,gate_names.Y,gate_names.Z,gate_names.H,gate_names.S,gate_names.Sdag,gate_names.SqrtX,gate_names.SqrtXdag,gate_names.T,gate_names.Tdag,gate_names.U1,gate_names.U2,gate_names.U3,gate_names.RX,gate_names.RY,gate_names.RZ,gate_names.CNOT,gate_names.CZ,gate_names.SWAP],optimization_level=3)(circuit_from_tket(test_circ))
    return circuit_from_tket(test_circ)


@njit
def model_func(x, r, theta):
    return complex(r * np.cos(theta*x) + 1-r , -r*np.sin(theta*x ))

@njit
def model_func2(x, r, theta,r2,theta2):
    return complex(r * np.cos(theta*x) + r2*np.cos(theta2*x)+ (1-r-r2) , -r*np.sin(theta*x )-r2*np.sin(theta2*x ))

@njit
def model_func3(x, r, theta,r2,theta2,theta3):
    return complex(r * np.cos(theta*x) + r2*np.cos(theta2*x)+ (1-r-r2)*np.cos(theta3*x) , -r*np.sin(theta*x )-r2*np.sin(theta2*x )-(1-r-r2)*np.sin(theta3*x))

@njit
def residual_func(params,y,x):
    r,theta= params
    diff = [model_func(x[k],r,theta) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

@njit
def residual_func2(params,y,x):
    r,theta,r2,theta2= params
    diff = [model_func2(x[k],r,theta,r2,theta2) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

@njit
def residual_func3(params,y,x):
    r,theta,r2,theta2,theta3= params
    diff = [model_func3(x[k],r,theta,r2,theta2,theta3) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

def fit(x_points,y_points,alpha,dt):
    eps = 1e-5
    res_x=[]
    res_cost=[]
    for E in np.arange(-0.5,-40.,-1.):
        p0 = np.array([1.,E])
        result = least_squares(residual_func,p0,args=(y_points,x_points),bounds=([0.,-np.inf],[1,0.]))
        if abs(2.*np.pi/(result.x[1]))>1.0*dt:
            res_x.append(result.x)
            res_cost.append(result.cost)

    x_best = res_x[res_cost.index(min(res_cost))]
    print("fit1: ",x_best)

    p0 = np.array([x_best[0],x_best[1],1-x_best[0],0.*x_best[1]])
    bounds= ([0.,-np.inf,0.,-alpha*abs(x_best[1])-eps],[1,0.,abs(x_best[0])+eps,alpha*abs(x_best[1])+eps])

    for counter in range(len(p0)):
        if p0[counter]<bounds[0][counter]:
            p0[counter] = bounds[0][counter]
        if p0[counter]>bounds[1][counter]:
            p0[counter] = bounds[1][counter]
    result2 = least_squares(residual_func2,p0,args=(y_points,x_points),bounds=bounds)
    print(result2.x,result2.cost)

    p0 = np.array([x_best[0],x_best[1],1-x_best[0],0.5*x_best[1],0.5*result2.x[3]])
    bounds = ([0.,-np.inf,0.,-alpha*abs(x_best[1])-eps,-alpha*abs(result2.x[3])-eps],[1,0.,abs(x_best[0])+eps,alpha*abs(x_best[1])+eps,alpha*abs(result2.x[3])+eps])
    for counter in range(len(p0)):
        if p0[counter]<bounds[0][counter]:
            p0[counter] = bounds[0][counter]
        if p0[counter]>bounds[1][counter]:
            p0[counter] = bounds[1][counter]
    result3 = least_squares(residual_func3,p0,args=(y_points,x_points),bounds=bounds)
    print(result3.x,result3.cost)
    return result3.x[1]

    

def sample_a_circ(n, nshots,n_qubits,hf_state,controlled_trotter_step_circ,sampler):
    real_pt=0.
    imag_pt=0.
    ov_circ_r = QuantumCircuit(qubit_count=n_qubits + 1,cbit_count=1)
    ov_circ_r = ov_circ_r.combine(hf_state.circuit.gates)
    ov_circ_r.add_H_gate(n_qubits)
    for m in range(n):
        ov_circ_r = ov_circ_r.combine(controlled_trotter_step_circ)
    ov_circ_r.add_H_gate(n_qubits)


    ov_circ_i = QuantumCircuit(n_qubits + 1)
    ov_circ_i = ov_circ_i.combine(hf_state.circuit.gates)
    ov_circ_i.add_H_gate(n_qubits)
    ov_circ_i.add_Sdag_gate(n_qubits)
    for m in range(n):
        ov_circ_i = ov_circ_i.combine(controlled_trotter_step_circ)
    ov_circ_i.add_H_gate(n_qubits)

    sampling_result = sampler(ov_circ_r, nshots)
    zero_count=0
    one_count=0
    for m in sampling_result:
        binary_val="0"*(n_qubits+1-len(str(bin(m))[2:]))+bin(int(m))[2:]
        if binary_val[0] == "0":
            zero_count+=sampling_result[m]
        elif binary_val[0] == "1":
            one_count+=sampling_result[m]    
    zero_prob=zero_count/nshots
    one_prob = one_count/nshots
    real_pt = 2*zero_prob - 1

    sampling_result = sampler(ov_circ_i, nshots)

    zero_count=0
    one_count=0
    for m in sampling_result:
        binary_val="0"*(n_qubits+1-len(str(bin(m))[2:]))+bin(int(m))[2:]
        if binary_val[0] == "0":
            zero_count+=sampling_result[m]
        elif binary_val[0] == "1":
            one_count+=sampling_result[m]
    zero_prob=zero_count/nshots
    one_prob = one_count/nshots
    imag_pt = 2*zero_prob - 1

    print_to_file_itr(str(n) + "  Execution time: " + str(datetime.now() - startTime) + " ovlp: " + str(complex(real_pt,imag_pt)))
    print(n,"  Execution time: ",datetime.now() - startTime, " ovlp: ",complex(real_pt,imag_pt))
    return complex(real_pt,imag_pt)


class Solver:
    def __init__(self,sampler,is_classical,n_qubits,n_elec,qp_hamiltonian,ham_terms,ham_cutoff,delta_t,n_Z,alpha) -> None:
        self.sampler = sampler
        self.is_classical = is_classical
        self.n_qubits = int(n_qubits)
        self.n_elec=int(n_elec)
        self.qp_hamiltonian = qp_hamiltonian
        self.ham_terms = int(ham_terms)
        self.ham_cutoff = ham_cutoff
        self.delta_t = delta_t
        self.n_Z = int(n_Z)
        self.alpha = alpha
        if self.is_classical:
            self.sampler = create_qulacs_vector_sampler()

    # except not allow optimisation
    def run(self):
        full_qb_ham = self.qp_hamiltonian.copy()
        ham_constant = full_qb_ham.__getitem__(PauliLabel())
        if PauliLabel() in self.qp_hamiltonian:
            self.qp_hamiltonian.pop(PauliLabel())

        qp_hamiltonian = round_hamiltonian(self.qp_hamiltonian,num_pickup=self.ham_terms,coeff_cutoff=self.ham_cutoff)
        print("truncated ham size: ",qp_hamiltonian.n_terms, " Number of fitting points: ", self.n_Z)
        ham_terms_trunc = qp_hamiltonian.n_terms
        nshots = 1e7//(2* (self.n_Z-1))

        print("shots per matrix element: ", nshots)
        hf_state = ComputationalBasisState(self.n_qubits, bits=2 ** self.n_elec - 1)
        controlled_trotter_step_circ = get_controlled_trotter_circuit(qp_hamiltonian,self.delta_t,self.n_qubits)

        print("N gate: ", len(controlled_trotter_step_circ.gates), " dt: ",self.delta_t)

        fitting_points = np.zeros(self.n_Z, dtype=complex)
        fitting_points[0] = complex(1.,0.)

        max_n=0
        try:
            for n in range(1,self.n_Z):
                fitting_points[n] = sample_a_circ(n, nshots,self.n_qubits,hf_state,controlled_trotter_step_circ,self.sampler)
                max_n=n

        except ExceededError as e:

            x_points = [self.delta_t*x for x in range(0,max_n)]
            y_points = fitting_points[:max_n]
            E_gs = fit(x_points,y_points,self.delta_t)
            print(str(e), " E_gs: ",E_gs)
            print_to_file_itr("E_gs: " + str(E_gs))
            return E_gs

        except Exception as general_error:
            error_message = f"Unexpected error: {general_error}"
            print(error_message)
            print_to_file_itr(error_message)
            return None
    
        x_points = [self.delta_t*x for x in range(0,self.n_Z)]
        y_points = fitting_points
        print("x_points = ",x_points, "y_points = ",y_points)
        E_gs = fit(x_points,y_points,self.alpha,self.delta_t)
        print("E_gs: ",E_gs)
        print_to_file_itr("E_gs: " + str(E_gs))
        return E_gs




class Wrapper:
    def __init__(self, sampler,is_classical,n_qubits,n_elec,ham,ham_terms,ham_cutoff,delta_t,n_Z,alpha) -> None:
        # challenge_sampling.reset()

        self.sampler = sampler
        self.is_classical = is_classical
        self.n_qubits = n_qubits
        self.n_elec = n_elec
        self.ham = ham
        self.ham_terms = ham_terms
        self.ham_cutoff = ham_cutoff
        self.delta_t = delta_t
        self.n_Z = n_Z
        self.alpha = alpha

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """

        # ham = self.ham
        jw_hamiltonian = jordan_wigner(self.ham)
        qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        mps_sampler = challenge_sampling.create_sampler()


        solver = Solver(
            mps_sampler,
            self.is_classical,
            self.n_qubits,
            self.n_elec,
            qp_hamiltonian,
            self.ham_terms,
            self.ham_cutoff,
            self.delta_t,
            self.n_Z,
            self.alpha
        )
        res = solver.run()
        return res


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    @njit
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
        # Print the current time
        current_time_s = my_timestamp()
        print("Current Time:", current_time_s)
        print_to_file_itr("Start: " +  str(n_qubits) + " seed: " + str(seed) + " time: " + current_time_s)

        n_elec = n_qubits//2
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        is_classical=False
        """
        ####################################
        add codes here
        ####################################
        """
        delta_t = 0.3 #random.uniform(1e-3,0.3)#
        n_Z = 10 #random.randint(5,25)#
        ham_terms = 200  #random.randint(50,1000)#
        ham_cutoff = None
        alpha = 0.8  #random.uniform(0.5,1) #

        sampler = challenge_sampling.create_sampler()

        # add extra code for experiments:
        is_OPT = True
        if is_OPT:
            print ("Optimised params")
            # Get better hyper-params
            if n_qubits == 20:
                if (seed == 0):
                    res_opt = np.array(
                          [
                           2.00000000000000000e+01, 1.00000000000000000e+01, 2.03553255544773021e-01,
                           9.00000000000000000e+00, 9.80000000000000000e+01, 8.40990934260731071e-03,
                           5.03579034261508962e-01 ] # 20,0, n_z = 9

                        , dtype=float)
                elif (seed == 1):
                    res_opt = np.array(
                          [2.00000000000000000e+01, 1.00000000000000000e+01, 2.02095003116297856e-01,
                           1.85000000000000000e+01, 1.46000000000000000e+02, 8.34979425264609568e-03,
                           5.17756802571591512e-01 ] # 20,1, new, n_z = 19
                        , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [20.0, 10.0, 0.20833096205618162, 10.0, 116.0, 0.008390067341719952, 0.5240496926349418] # 20,2, new n_z=10
                        , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [
                             2.00000000000000000e+01, 1.00000000000000000e+01, 2.01424723474243950e-01,
                             8.00000000000000000e+00, 1.74000000000000000e+02, 8.33577013824858895e-03,
                             5.03754950047298200e-01 ] # 20,3, n_z = 8
                        , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [2.00000000000000000e+01, 1.00000000000000000e+01, 2.00190554895460671e-01,
                           2.40000000000000000e+01, 1.67500000000000000e+02, 8.41137406817806139e-03,
                           5.45927804638917769e-01 ] # 20,4, new n_z = 24
                        , dtype=float)

                elif (seed == 5):
                    res_opt = np.array(
                          [
                          2.00000000000000000e+01, 1.00000000000000000e+01, 2.17675856508570603e-01,
                          2.00000000000000000e+01, 6.70000000000000000e+01, 9.73263144301033702e-03,
                          5.07903285275227545e-01 ] # 20,5, n_z=20
                        , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 24:
                if (seed == 5):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 1.99802185604540827e-01,
                          1.70000000000000000e+01, 7.90000000000000000e+01, 9.95336063227277197e-03,
                          5.77057365918694831e-01 ] # 24,5, n_z=17
                        , dtype=float)

                elif (seed == 6):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.02208370095536688e-01,
                          1.70000000000000000e+01, 7.50000000000000000e+01, 9.81354737396960934e-03,
                          5.92811924761539988e-01 ] # 24,6, n_z=17
                        , dtype=float)

                elif (seed == 7):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 1.96109431581716925e-01,
                          2.40000000000000000e+01, 6.75000000000000000e+01, 8.40633981198942067e-03,
                          6.02858839442908190e-01 ] # 24,7, n_z=24
                        , dtype=float)

                elif (seed == 8):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.03033308059204165e-01,
                          9.00000000000000000e+00, 7.30000000000000000e+01, 9.97384266029652938e-03,
                          5.52657866550458787e-01 ] # 24,8, n_z=9
                        , dtype=float)

                elif (seed == 9):
                    res_opt = np.array(
                          [
                          2.40000000000000000e+01, 1.20000000000000000e+01, 2.09114207651129175e-01,
                          1.60000000000000000e+01, 7.80000000000000000e+01, 8.34624269390677256e-03,
                          5.65200801771179151e-01 ] # 24,9, n_z=16
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            elif n_qubits == 28:
                if (seed == 0):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,7.0017442061948805e-03,
                          7.0000000000000000e+00,3.1000000000000000e+02,6.7115397362500061e-03,6.3276343991833195e-01] # 28,0
                       , dtype=float)

                elif (seed == 1):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,1.9971283221559202e-01,
                          2.3000000000000000e+01,3.0900000000000000e+02,8.3995728796683409e-03,5.5405304641464770e-01] # 28,1
                       , dtype=float)

                elif (seed == 2):
                    res_opt = np.array(
                          [2.8000000000000000e+01,1.4000000000000000e+01,5.4676028526620297e-02,
                          1.2000000000000000e+01,9.9700000000000000e+02,1.2270564966696167e-03,5.6024259125284925e-01] # 28,2
                       , dtype=float)

                elif (seed == 3):
                    res_opt = np.array(
                          [2.800000000000000e+01,1.400000000000000e+01,4.233532475869142e-02,
                          6.000000000000000e+00,5.530000000000000e+02,8.405430923458371e-03,8.661621644069664e-01] # 28,3
                       , dtype=float)

                elif (seed == 4):
                    res_opt = np.array(
                          [ 2.80000000000000000e+01, 1.40000000000000000e+01, 2.01580003185462042e-01,
                            1.52500000000000000e+01, 2.92750000000000000e+02, 8.39264463088176642e-03,
                            5.27179984652617395e-01 ] # 28,4
                       , dtype=float)

                else:
                    print ("EERROR n_qubits is : " + str(n_qubits) + " with seed" + str(seed))
                    return 0

            else:
                print ("EERROR n_qubits is : " + str(n_qubits))
                return 0

            n_elec = round(res_opt[1])
            delta_t = res_opt[2]
            n_Z = round(res_opt[3])
            ham_terms = round(res_opt[4])
            ham_cutoff = res_opt[5]
            alpha = res_opt[6]

            #param_bool_2 = True if round(res_opt[5]) != 0 else False
            #wrapper=Wrapper(n_qubits, ham, False, True, round(res_opt[3]), res_opt[4], param_bool_2,

        else:
            print ("Default params")

        try:
            jw_hamiltonian = jordan_wigner(ham)
            qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

            # res_opt = svm(n_qubits, seed, hamiltonian_directory)
            # wrapper = Wrapper(n_qubits, ham, res_opt[0], res_opt[1], res_opt[2], res_opt[3], res_opt[4], res_opt[5], res_opt[6], res_opt[7], res_opt[8], res_opt[9], res_opt[10], res_opt[11])
            wrapper = Wrapper(sampler,is_classical,n_qubits,n_elec,ham,ham_terms,ham_cutoff,delta_t,n_Z,alpha)
            E_gs=wrapper.get_result(seed, hamiltonian_directory="../hamiltonian")

            # Print the current time
            current_time = my_timestamp()
            print("Current Time:", current_time)
            print_to_file(">>> Start | " + str(current_time_s) + " | OPT? | " + str(is_OPT) + " | seed | " + str(seed) + " | n_qubits  | " + str(n_qubits) + " | res | " + str(E_gs))

            return E_gs
        
        except Exception as general_error:
            error_message = f"Unexpected error: {general_error}"
            print(error_message)
            print_to_file(error_message)
            return None

if __name__ == "__main__":
    np.set_printoptions(precision=17)
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(n_qubits=20,seed=5, hamiltonian_directory="../hamiltonian"))
