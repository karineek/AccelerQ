
"""
This script is 

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os

# Imports for Q part
sys.path.append("../")
from first_answer import Wrapper

# Imports for ML part
import numpy as np
import random
import copy


# Imports for Q part
from openfermion import FermionOperator
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from first_answer import PauliRotationCircuit, round_hamiltonian
from quri_parts.openfermion.operator import operator_from_openfermion_op
from openfermion import QubitOperator, jordan_wigner
from kcl_util import process_file, ham_to_vector



#############
def get_round_ham_size(x_vec_params, hamiltonian, flag_max=False):
    # Ensure Hamiltonian is an Operator
    if not isinstance(hamiltonian, FermionOperator):
        return 0

    # Ensure num_pickup is an integer and coeff_cutoff is a float
    num_pickup = int(x_vec_params[3])
    coeff_cutoff = float(x_vec_params[4])

    # Run round_hamiltonian with converted Hamiltonian
    jw_hamiltonian = jordan_wigner(hamiltonian)
    qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
    len_qp = len(qp_hamiltonian.items())
    if (flag_max is True):
        num_pickup = len_qp

    round_ham = round_hamiltonian(qp_hamiltonian, num_pickup=num_pickup, coeff_cutoff=coeff_cutoff)
    len_round = len(round_ham.items())

    #print ("Size before qp_hamiltonian:", len(qp_hamiltonian), " and " , len_qp, " and after", len_round)

    return (len_qp, len_round)


def print_1(x_vec_params, hamiltonian, num_qubits, seed):
    (len_qp, len_round) = get_round_ham_size(x_vec_params, hamiltonian)
    ham_vec = ham_to_vector(hamiltonian, 50)
    print ("Seed: " + str(seed) + " qubits: " + str(num_qubits) + "hamiltonian round size: " + str(len_round))
    print ("GREPME: (orig)" + str(len(hamiltonian.terms)) + " | (compressed-ML)" + str(len(ham_vec)) + " | (qp-before) " + str(len_qp))


#############
def test_1(x_vec_params, hamiltonian):
    """
    Test 1: check if iter_max * sampling_shots = 10^(7) - 10^7 fix for this platform.
    (D,E)
    """
    iter_max: int = int(round(x_vec_params[6]))
    if (iter_max < 7): # that's way too little, exit
        return False
    if (iter_max > 1000): # that's way too much, exit
        return False
    if (x_vec_params[7] >= 3_000_000):
       return False

    # Then check we are not over shooting
    total = (x_vec_params[6] * x_vec_params[7])
    ## print ("x_vec_params[6] * x_vec_params[7] = ", str(x_vec_params[6]), " * ",  str(x_vec_params[7]), " = ",  total)
    return ((total < 100_000_000) ## 10**8
           and (total > 1_000_000)) ## 10**6


def test_2(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 2: Check if the compressed ham size parameters affect the compressed Hamiltonian size.
    If we result in the same size, then the test fails.
    coeff_cutoff and num_pickup
    (A,B)
    """
    if (len_round < 1):
        return False
    num_pickup = int(x_vec_params[3])
    if (num_pickup > len_orig):
        return False
    if (len_orig < 100):
        return (len_round <= len_orig)
    return (len_round < len_orig) and (len_round < 800)


def test_3(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 3: Check if the cutoff is sensible or all the term's coeff are larger than this.
    coeff_cutoff
    (B)
    """
    if (len_round < 1):
        return False

    # Run round_hamiltonian with converted Hamiltonian
    (len_orig_max, len_round_max) = get_round_ham_size(x_vec_params, hamiltonian, True)
    if (len_round_max < 1):
        return False
    if (len_orig != len_orig_max):
        return False
    if (len_round_max < len_round):
        return False
    if (len_orig < 100):
        return len_round <= len_round_max
    # If not all these odd cases, check the cut-off actually does something
    return (len_round_max - len_round) > 25

###############
def is_particle_conserving(hamiltonian):
    for term, coeff in hamiltonian.terms.items():
        # Count the net particle change
        particle_count = sum(1 if '^' in op else -1 for op in term)
        if particle_count != 0:
            return False
    return True

def test_4(x_vec_params, hamiltonian):
    """
    Test 4: Test if the flag self_selection is sensibly set
    self_selection that is post_selection and then post_selected
    (C)
    """
    is_Fermion = isinstance(hamiltonian, FermionOperator)
    is_conserving = is_particle_conserving(hamiltonian)
    self_selection = True if round(x_vec_params[5]) != 0 else False

    # Define logical conditions to determine if `self_selection` is sensible
    sensible_selection = (
        (self_selection and is_Fermion and is_conserving) or       # Case 1: Fermionic and particle-conserving, needs post-selection
        (not self_selection and not is_Fermion) or                 # Case 2: Non-fermionic, no need for post-selection
        (not self_selection and is_Fermion and not is_conserving)  # Case 3: Fermionic but non-particle-conserving, avoids post-selection
    )
    return sensible_selection

def test_9(x_vec_params, hamiltonian):
    """
    Test 9: check that if C is true, then D is bigger
    than J (that is D and J are sensible)
    """
    self_selection = True if round(x_vec_params[5]) != 0 else False
    if (self_selection is False):
        return True

    max_iterations = round(x_vec_params[6])
    reset_itr = round(x_vec_params[12])
    return max_iterations > reset_itr

def test_10(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 10: check that A is sensible
    """
    num_pickup = int(x_vec_params[3])
    if (num_pickup < 40):
        return False
    # Run round_hamiltonian with converted Hamiltonian
    (len_orig, len_round) = get_round_ham_size(x_vec_params, hamiltonian)
    if (len_orig < num_pickup):
        return False

    return True

############
# Deeper tests to check we are doing something sensible:
def test_get_gate_count(circuit: PauliRotationCircuit):
    return len(circuit.generators) > 0

def test_get_parameter_count(circuit: PauliRotationCircuit):
    return len(circuit.param_names) > 0

def get_qubit_count(circuit: PauliRotationCircuit):
    return circuit.n_qubits >= 20 # As this is the definition of the systems we optimising

# Take tests from here: https://quri-parts.qunasys.com/docs/tutorials/basics/circuits/
# Test is unitary
#def test_5(x_vec_params, hamiltonian):
#    """
#    Test 3: Check if the cutoff is sensible or if all the term's coeff are larger than this.
#    coeff_cutoff
#    (B)
#    """
#    return True

# Test is can print a QC
#def test_6(x_vec_params, hamiltonian):
#    """
#    Test 3: Check if the cutoff is sensible or all the term's coeff are larger than this.
#    coeff_cutoff
#    (B)
#    """
#    return True

# No need to run the code or part of it, of the implementation we optimise
def test_static_adapt(x_vec_params, hamiltonian):
    # print(">>>> RET test-1-4:", str(test_1(x_vec_params) and test_4(x_vec_params, hamiltonian)))
    # return test_1(x_vec_params) and test_4(x_vec_params, hamiltonian) and test_9(x_vec_params, hamiltonian)
    if (not test_1(x_vec_params, hamiltonian)):
        return False
    if (not test_4(x_vec_params, hamiltonian)):
        return False
    if (not test_9(x_vec_params, hamiltonian)):
        return False

    return True

# Need to run part of the system/code we are optimising
def test_semi_dynamic_adapt(x_vec_params, hamiltonian):
    # print ("Test 2:", str(test_2(x_vec_params, hamiltonian)))
    # print ("Test 3:", str(test_3(x_vec_params, hamiltonian)))
    #return test_2(x_vec_params, hamiltonian) and test_3(x_vec_params, hamiltonian) and test_10(x_vec_params, hamiltonian)

    # Optimising: test 10
    num_pickup = int(x_vec_params[3])
    if (num_pickup < 40):
        return False

    # Only if we have to, we call this:
    # Run round_hamiltonian with converted Hamiltonian
    (len_orig, len_round) = get_round_ham_size(x_vec_params, hamiltonian)

    # Then run the tests:
    if (not test_10(x_vec_params, hamiltonian, len_orig, len_round)):
        return False
    if (not test_2(x_vec_params, hamiltonian, len_orig, len_round)):
        return False
    if (not test_3(x_vec_params, hamiltonian, len_orig, len_round)):
        return False
    return True

# We need to run the code in a few iterations to decide. Statistically decision.
def test_dynamic_adapt(circuit):
    return test_get_gate_count(circuit) and test_get_parameter_count(circuit) and get_qubit_count(circuit)
