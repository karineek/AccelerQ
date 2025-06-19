import sys
from typing import Any

import numpy as np
import random
from quri_parts.core.operator.trotter_suzuki import trotter_suzuki_decomposition
sys.path.append("..")
from typing import Any
from scipy.optimize import minimize, OptimizeResult
from itertools import combinations
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.algo.ansatz import SymmetryPreservingReal
#from quri_parts.algo.optimizer import SPSA, OptimizerStatus, Adam, LBFGS, NFT, AdaBelief
#from quri_parts.algo.optimizer import (
#    Optimizer, OptimizerState, CostFunction, GradientFunction
#)
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
from datetime import datetime
# from multiprocessing import Pool
from time import time

from scipy.optimize import curve_fit,least_squares

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, problem_hamiltonian

challenge_sampling = ChallengeSampling()

"""
####################################
add codes here
####################################
"""
startTime = datetime.now()

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
    print("Total gate count:", test_circ.n_gates)
    print("2 qubit gates:", test_circ.n_2qb_gates())
    
    #return TketTranspiler(basis_gates=[gate_names.Identity,gate_names.X,gate_names.Y,gate_names.Z,gate_names.H,gate_names.S,gate_names.Sdag,gate_names.SqrtX,gate_names.SqrtXdag,gate_names.T,gate_names.Tdag,gate_names.U1,gate_names.U2,gate_names.U3,gate_names.RX,gate_names.RY,gate_names.RZ,gate_names.CNOT,gate_names.CZ,gate_names.SWAP],optimization_level=3)(circuit_from_tket(test_circ))
    return circuit_from_tket(test_circ)



def model_func(x, r, theta):
    return complex(r * np.cos(theta*x) + 1-r , -r*np.sin(theta*x ))

def model_func2(x, r, theta,r2,theta2):
    return complex(r * np.cos(theta*x) + r2*np.cos(theta2*x)+ (1-r-r2) , -r*np.sin(theta*x )-r2*np.sin(theta2*x ))

def model_func3(x, r, theta,r2,theta2,theta3):
    return complex(r * np.cos(theta*x) + r2*np.cos(theta2*x)+ (1-r-r2)*np.cos(theta3*x) , -r*np.sin(theta*x )-r2*np.sin(theta2*x )-(1-r-r2)*np.sin(theta3*x))


def residual_func(params,y,x):
    r,theta= params
    diff = [model_func(x[k],r,theta) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

def residual_func2(params,y,x):
    r,theta,r2,theta2= params
    diff = [model_func2(x[k],r,theta,r2,theta2) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

def residual_func3(params,y,x):
    r,theta,r2,theta2,theta3= params
    diff = [model_func3(x[k],r,theta,r2,theta2,theta3) - y[k] for k in range(len(x))]
    return [k.real**2 + k.imag**2 for k in diff]

# def fit(x_points,y_points,alpha):
#     res_x=[]
#     res_cost=[]
#     for E in np.arange(-0.5,-40.,-1.):
#         p0 = np.array([1.,E])
#         result = least_squares(residual_func,p0,args=(y_points,x_points),bounds=([0.5,-np.inf],[1,0.]))
#         res_x.append(result.x)
#         res_cost.append(result.cost)

#     x_best = res_x[res_cost.index(min(res_cost))]
#     print("fit1: ",x_best)

#     p0 = np.array([x_best[0],x_best[1],1-x_best[0],0.*x_best[1]])
#     result2 = least_squares(residual_func2,p0,args=(y_points,x_points),bounds=([0.,-np.inf,0.,-alpha*abs(x_best[1])],[1,0.,abs(x_best[0]),alpha*abs(x_best[1])]))
#     print(result2.x,result2.cost)

    
#     p0 = np.array([x_best[0],x_best[1],1-x_best[0],(alpha - 1e-4)*x_best[1],0.*result2.x[3]])
        
#     result3 = least_squares(residual_func3,p0,args=(y_points,x_points),bounds=([0.,-np.inf,0.,-alpha*abs(x_best[1]),-alpha*abs(result2.x[3])],[1,0.,abs(x_best[0]),alpha*abs(x_best[1]),alpha*abs(result2.x[3])]))
#     print(result3.x,result3.cost)
#     return result3.x[1]


eps = 1e-5

def fit(x_points,y_points,alpha,dt):
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

    # print(p0,bounds)
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
        # print(m,binary_val[n_qubits])
        if binary_val[0] == "0":
            zero_count+=sampling_result[m]
        elif binary_val[0] == "1":
            one_count+=sampling_result[m]    
    zero_prob=zero_count/nshots
    one_prob = one_count/nshots
    imag_pt = 2*zero_prob - 1

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


    def run(self):
        full_qb_ham = self.qp_hamiltonian.copy()    
        ham_constant = full_qb_ham.__getitem__(PauliLabel())
        if PauliLabel() in self.qp_hamiltonian:
            self.qp_hamiltonian.pop(PauliLabel())

        qp_hamiltonian = round_hamiltonian(self.qp_hamiltonian,num_pickup=self.ham_terms,coeff_cutoff=self.ham_cutoff)
        print("truncated ham size: ",qp_hamiltonian.n_terms)
        print("Number of fitting points: ", self.n_Z)
        ham_terms_trunc = qp_hamiltonian.n_terms
        nshots = 1e7//(2* (self.n_Z-1))

        print("shots per matrix element: ", nshots)
        hf_state = ComputationalBasisState(self.n_qubits, bits=2 ** self.n_elec - 1)
        controlled_trotter_step_circ = get_controlled_trotter_circuit(qp_hamiltonian,self.delta_t,self.n_qubits)

        print("N gate: ", len(controlled_trotter_step_circ.gates))
        print("dt: ",self.delta_t)

        fitting_points = np.zeros(self.n_Z, dtype=complex)
        fitting_points[0] = complex(1.,0.)

        # print("CPU count: ", os.cpu_count())
        # processes_pool = Pool(2)
        # fitting_points = processes_pool.map(sample_a_circ, range(1,n_Z))
        # fitting_points.insert(0,complex(1.,0.))
        max_n=0
        try:
            for n in range(1,self.n_Z):
                fitting_points[n] = sample_a_circ(n, nshots,self.n_qubits,hf_state,controlled_trotter_step_circ,self.sampler)
                max_n=n

        except ExceededError as e:

            x_points = [self.delta_t*x for x in range(0,max_n)]
            y_points = fitting_points[:max_n]
            E_gs = fit(x_points,y_points)
            print(str(e))
            print("E_gs: ",E_gs)

            return E_gs
        
        x_points = [self.delta_t*x for x in range(0,self.n_Z)]

        y_points = fitting_points
        print("x_points = ",x_points)
        print("y_points = ",y_points)
        E_gs = fit(x_points,y_points,self.alpha,self.delta_t)
        print("E_gs: ",E_gs)   
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
        n_qubits = 8
        n_elec = n_qubits//2
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        is_classical=True
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
        


        
        jw_hamiltonian = jordan_wigner(ham)
        qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        # res_opt = svm(n_qubits, seed, hamiltonian_directory)
        # wrapper = Wrapper(n_qubits, ham, res_opt[0], res_opt[1], res_opt[2], res_opt[3], res_opt[4], res_opt[5], res_opt[6], res_opt[7], res_opt[8], res_opt[9], res_opt[10], res_opt[11])
        wrapper = Wrapper(sampler,is_classical,n_qubits,n_elec,ham,ham_terms,ham_cutoff,delta_t,n_Z,alpha)
        E_gs=wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")
        return E_gs


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=5, hamiltonian_directory="../hamiltonian"))
