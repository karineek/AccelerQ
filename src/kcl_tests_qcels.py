
"""
This module defines validation tests for filtering hyperparameter configurations used 
in the QCELS variant of AccelerQ. 

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
from QCELS_answer_experiments import Wrapper

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
    num_pickup = int(x_vec_params[4]) #  Now is ham_terms
    coeff_cutoff = float(x_vec_params[5]) #  Now is ham_cutoff

    # Run round_hamiltonian with converted Hamiltonian
    jw_hamiltonian = jordan_wigner(hamiltonian)
    qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
    len_qp = len(qp_hamiltonian.items())
    if (flag_max is True):
        num_pickup = len_qp

    round_ham = round_hamiltonian(qp_hamiltonian, num_pickup=num_pickup, coeff_cutoff=coeff_cutoff)
    len_round = len(round_ham.items())
    return (len_qp, len_round)

def print_1(x_vec_params, hamiltonian, num_qubits, seed):
     (len_qp, len_round) = get_round_ham_size(x_vec_params, hamiltonian)
     ham_vec = ham_to_vector(hamiltonian, 50)
     print ("Seed: " + str(seed) + " qubits: " + str(num_qubits) + "hamiltonian round size: " + str(len_round))
     print ("GREPME: (orig)" + str(len(hamiltonian.terms)) + " | (compressed-ML)" + str(len(ham_vec)) + " | (qp-before) " + str(len_qp))
 #############
#############

"""         n_elec = round(res_opt[1]) ?
            A: delta_t = res_opt[2] - A not too big not too small (that is?)
            B: n_Z = round(res_opt[3])
            C: ham_terms = round(res_opt[4]) ==> number of terms 600 == A in  adapt
            D: ham_cutoff = res_opt[5]  ==> B in adapt
            E: alpha = res_opt[6] should be between 0.5 to 1.0

            nshots = 1e7//(2* (self.n_Z-1)) --> not bigger than this
"""

#############
def test_1(x_vec_params):
    """
    Test 1: nshots = 1e7//(2* (self.n_Z-1)) --> not bigger than this, as 10^7 fix for this platform.
    (B)
    """
    n_z = round(x_vec_params[3])
    return ((n_z < 30) and (n_z > 5))


def test_2(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 2: Check if the compressed ham size parameters affect the compressed Hamiltonian size.
    If we result in the same size, then the test fails.
    coeff_cutoff and num_pickup (A,B) ==> ham_terms  and ham_cutoff (C,D)
    qubits - 20, we up to 200, | 24, up to 200, | and if 28 up to 500.
    """
    num_pickup = round(x_vec_params[4])
    qubits = round(x_vec_params[0])
    if (num_pickup > 500) and (qubits > 24):
        return False
    if (num_pickup > 200) and (qubits <= 24):
        return False
    if (len_round < 1):
        return False
    if (num_pickup > len_orig):
        return False
    if (len_orig < 100):
        return (len_round <= len_orig)

    return (len_round < len_orig) and (len_round < 500)


def test_3(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 3: Check if the cutoff is sensible or all the term's coeff are larger than this.
    ham_cutoff
    (D)
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
def test_4(x_vec_params):
    """
    Test 4: A: delta_t = res_opt[2]
    """
    delta_t = x_vec_params[2]
    return ((delta_t < 0.5) and (delta_t > 0.0001))

def test_9(x_vec_params):
    """
    Test 9: alpha = res_opt[6]
    """
    alpha = x_vec_params[6]
    return ((alpha < 0.9) and (alpha > 0.5))

def test_10(x_vec_params, hamiltonian, len_orig, len_round):
    """
    Test 10: check that A is sensible
    """
    num_pickup = int(x_vec_params[4])
    if (num_pickup < 40):
        return False
    if (len_orig < num_pickup):
        return False

    return True

############
# No need to run the code or part of it, of the implementation we optimise
def test_static_qcels(x_vec_params, hamiltonian):
    if not test_1(x_vec_params):
        return False
    if not test_4(x_vec_params):
        return False
    if not test_9(x_vec_params):
        return False
    return True

# Need to run part of the system/code we are optimising
def test_semi_dynamic_qcels(x_vec_params, hamiltonian):
    # Pull so of the tests to here for optimisation:
    num_pickup = int(x_vec_params[4])
    if (num_pickup < 40):
        return False
    num_pickup = round(x_vec_params[4])
    qubits = round(x_vec_params[0])
    if (num_pickup > 500) and (qubits > 24):
        return False
    if (num_pickup > 200) and (qubits <= 24):
        return False

    # Run round_hamiltonian with converted Hamiltonian
    (len_orig, len_round) = get_round_ham_size(x_vec_params, hamiltonian)
    if not test_10(x_vec_params, hamiltonian, len_orig, len_round):
        return False
    if not test_2(x_vec_params, hamiltonian, len_orig, len_round):
        return False
    if not test_3(x_vec_params, hamiltonian, len_orig, len_round):
        return False
    return True
