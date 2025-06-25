
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

# Have a better distribution with all possible 10 to 10**5, then include also 10**6.
def generate_value_intersting_itr():
    choice = random.randint(1, 5)
    if choice == 1:
        return random.randint(10**1, 10**2)      # Range 10 to 100
    elif choice == 2:
        return random.randint(10**2, 10**3)      # Range 100 to 1,000
    elif choice == 3:
        return random.randint(10**3, 10**4)      # Range 1,000 to 10,000
    elif choice == 4:
        return random.randint(10**4, 10**5)      # Range 10,000 to 100,000
    else:
        return random.randint(10**5, 10**6)      # Range 100,000 to 1,000,000

def generate_value_intersting_shots():
    choice = random.randint(1, 5)
    if choice == 1:
        return random.randint(10**6, 10**7) 
    elif choice == 2:
        return random.randint(10**5, 10**6)
    elif choice == 3:
        return random.randint(10**4, 10**5)
    else:
        return random.randint(10**2, 10**5)

# Random hyper params
        ### 3 - is_classical, binary
        ### 4 - use_singles, binary, always true
        ### 5 - num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
        ### 6 - coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
        ### 7 - self_selection, binary
        ### 8 - iter_max, int definitely > 1 (want large)
        ### 9 - sampling_shots, int definitely >1 probably want fairly large, at least 100
        ### 10- atol, float definitely >0 and < 1, probably < 1e-3
        ### 11- final_sampling_shots_coeff, int definitely > 0 and probably < 10
        ### 12- num_precise_gradient, int definitely >0
        ### 13- max_num_converged, int definitely > 1
        ### 14- reset_ignored_inx_mode, int deffinitely >=0
        ### def __init__(self, number_qubits, ham, is_classical, use_singles, num_pickup, coeff_cutoff, self_selection, iter_max, sampling_shots, atol, final_sampling_shots_coeff, num_precise_gradient, max_num_converged, reset_ignored_inx_mode) -> None:
# Wrapper(n_qubits, ham, False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128,  2, 0)
#            1        2   3      [0]4  5   [0]**6   7     8     9    **10  11  12  13  14
# =========================================================================================================================


def generate_integers():
    """
    Generate random integer values for specific hyperparameters.

    This function generates random values for integer hyperparameters used in a
    specific algorithm.

    Returns:
    - list: A list containing randomly generated integer values.

    Notes:
    - The generated values are within predefined ranges suitable for certain
      hyperparameters (num_pickup, self_selection, iter_max, sampling_shots,
      final_sampling_shots_coeff, num_precise_gradient, max_num_converged,
      reset_ignored_inx_mode) used in the algorithm.
    """

    random_values_int = []

    # num_pickup: 10-100 - IND:5
    ###num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
    random_values_int.append(random.randint(50, 1000))

    ###self_selection, binary  IND:7
    random_values_int.append(random.randint(0, 1))

    # iter_max: 10-100 - IND:8
    ###iter_max, int definitely > 1 (want large)
    # random_values_int.append(int(np.clip(np.random.gamma(500, 2.0), 10, 10**5)))
    # random_values_int.append(random.randint(10, 10**5))
    random_values_int.append(generate_value_intersting_itr())
        
    # sampling_shots 10**1 to 10**5 - IND:9
    ###sampling_shots, int definitely >1 probably want fairly large, at least 100
    # random_values_int.append(random.randint(100, 10**6))
    random_values_int.append(generate_value_intersting_shots())

    # final_sampling_shots_coeff 1 to 10 - IND:11
    ###final_sampling_shots_coeff, int definitely > 0 and probably < 10
    random_values_int.append(random.randint(1, 9))

    # num_precise_gradient 10 to 300 - IND:12
    ###num_precise_gradient, int definitely >0
    random_values_int.append(random.randint(35, 300))

    # max_num_converged 0 to 10 - IND:13
    ###max_num_converged, int definitely > 1
    # assert max_num_converged >= 1
    # 5+ it gives error: IndexError: list index out of range
    random_values_int.append(random.randint(2, 4))

    # reset_ignored_inx_mode 0 to 100 - IND:14
    ##reset_ignored_inx_mode, int definitely >=0
    random_values_int.append(random.randint(0, 100))

    # 4, 5, 7, 8, 9, 11, 12, 13, 14
    return random_values_int

def generate_floats():

    """
    Generate random floating-point values for specific hyperparameters.

    This function generates random values for coefficients and precision/error
    hyperparameters used in a specific algorithm.

    Returns:
    - list: A list containing randomly generated floating-point values.

    Notes:
    - The generated values are within predefined ranges suitable for certain
      hyperparameters (coeff_cutoff and atol) used in the algorithm.
    """

    random_values_float = []

    # coeff_cutoff: 0.01 to 0.0001 - INd:6
    ###coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
    random_values_float.append(random.uniform(1e-8, 1e-2))

    # atol 1e-2 to 1e-8 ??? ==> the precision/error - IND:10
    ###atol, float definitely >0 and < 1, probably < 1e-3  --> need to move!
    random_values_float.append(random.uniform(1e-8, 1e-4))

    # IND:6 and IND:10
    return random_values_float

def generate_hyper_params_avqe(seed, n_qubits):
    """
    Generate hyperparameters for the Adapt VQE algorithm based on the seed and number of qubits.

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
    x_vec_params = np.array([float(n_qubits), 1, 1, 100, 0.001, 0, 100, 10**5, 1e-6, 5, 128, 2, 0], dtype=float)

    if seed > 1:
        rvi = generate_integers()
        rvf = generate_floats()
        # Construct hyperparameter vector with random values
        x_vec_params = np.array([float(n_qubits), 1, 1, rvi[0], rvf[0], rvi[1], rvi[2], rvi[3], rvf[1], rvi[4], rvi[5], rvi[6], rvi[7]], dtype=float)

    # Return the generated hyperparameter vector
    return x_vec_params

def wrapper_avqe(x_vec_params, n_qubits, ham):
    """
    Use the adapt VQE wrapper classically to obtain Y.

    Parameters:
    - x_vec_params (numpy.ndarray): Array of hyperparameters for configuring the adapt VQE wrapper.

    Returns:
    - energy level, float: Y obtained from the adapt VQE wrapper classically.

    """
    # Determine if certain parameters should be boolean based on x_vec_params
    param_bool = True if round(x_vec_params[5]) != 0 else False

    # Call the AVQE wrapper with specified parameters
    wrapper = Wrapper(n_qubits, ham, True, True, round(x_vec_params[3]), x_vec_params[4], param_bool,
                      round(x_vec_params[6]), round(x_vec_params[7]), x_vec_params[8],
                      round(x_vec_params[9]), round(x_vec_params[10]), round(x_vec_params[11]), round(x_vec_params[12]))

    # Get Y vector from the wrapper
    y_vec = wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")

    return y_vec


def compress_avqe(ham):
    """
    Use the compress function of this kind of hams to remove too small items
    """
    return ham.compress(0.05)
