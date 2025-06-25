
"""
This script contains a set of utility functions for QCELS QE implementation for the 
AccelerQ pipeline. 

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
from utils.challenge_2024 import ChallengeSampling

# Imports for ML part
import numpy as np
import random

def generate_hyper_params_qcels(seed, n_qubits):
    """
    Generate hyperparameters for the QCLES algorithm based on the seed and number of qubits.

    Parameters:
    - seed (int): Index refers to the loop (which generated seed it is).
    - n_qubits (int): Number of qubits.

    Returns:
    - numpy.ndarray: Array of hyperparameters.

    Notes:
    - Default hyperparameters are initialized first. These are commonly used parameters in avqe.
    - If seed > 1, additional random integers and floats are included in the hyperparameter array.
    """
    # Set precision high enough for this alg.
    np.set_printoptions(precision=17)

    # Init to default values
    n_elec = n_qubits//2
    delta_t = 0.03
    n_Z = 10
    ham_terms = 200
    ham_cutoff = 1e-9 #None
    alpha = 0.8
    if seed > 1:
        n_elec = n_qubits//2
        delta_t = random.uniform(1e-3,0.3)  #0.03
        n_Z = random.randint(5,25)          #10
        ham_terms = random.randint(50,1000) #200
        ham_cutoff = random.uniform(1e-8, 1e-2) #None
        alpha = random.uniform(0.5,1)       #0.8
        
    # Construct hyperparameter vector with random values
    x_vec_params = np.array([float(n_qubits), n_elec, delta_t, n_Z, ham_terms, ham_cutoff, alpha], dtype=float)

    # Return the generated hyperparameter vector
    return x_vec_params

def wrapper_qcels(x_vec_params, n_qubits, ham):
    """
    Use the QCLES wrapper classically to obtain Y.

    Parameters:
    - x_vec_params (numpy.ndarray): Array of hyperparameters for configuring the adapt QCLES wrapper.

    Returns:
    - energy level, float: Y obtained from the adapt QCLES wrapper classically.

    """
    # Determine if certain parameters should be boolean based on x_vec_params
    challenge_sampling = ChallengeSampling()
    sampler = challenge_sampling.create_sampler()

    n_elec = x_vec_params[1]
    delta_t = x_vec_params[2]
    n_Z = x_vec_params[3]
    ham_terms = x_vec_params[4]
    ham_cutoff = x_vec_params[5]
    alpha = x_vec_params[6]
    is_classical = True
    
    wrapper = Wrapper(sampler, is_classical, n_qubits, n_elec,ham,ham_terms,ham_cutoff,delta_t,n_Z,alpha)

    # Get Y vector from the wrapper
    y_vec = wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")

    return y_vec


def compress_qcels(ham):
    """
    Use the compress function of this kind of hams to remove too small items
    """
    return ham.compress(0.05)
