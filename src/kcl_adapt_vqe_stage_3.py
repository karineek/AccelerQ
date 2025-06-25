
"""
This script is a wrapper for ADPT-QSCI QE implementation for phase 3 (model deployment)

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import numpy as np
from scipy.spatial import distance
from kcl_util import process_file, ham_to_vector
from kcl_util_adapt_vqe import generate_hyper_params_avqe
from kcl_opt_xgb import opt_hyperparams
from kcl_tests_adapt_vqe import test_static_adapt, test_semi_dynamic_adapt, test_dynamic_adapt
from openfermion import FermionOperator as FO

import xgboost as xgb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    np.set_printoptions(precision=17, floatmode='fixed', suppress=True, 
                    formatter={'float': lambda x: f"{x:.17e}"})

    print (">> Start Stage 3")

    # Set precision
    np.set_printoptions(precision=17)

    # inputs
    folder_path = "../hamiltonian/"
    prefix = "24qubits_05" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    model_file = "model_avqe_pre_xgb_28.json"
    max_size = 138306 # 12638

    # Get Data
    print (">> Read ham " + folder_path + ", " + file_name)
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))
    # Get the ham flatten once
    ham28=result[1] # Need to load from Elena's files
    if not isinstance(ham28, FO):
        print ("Error type. Exit")
    else:
        ham28_vec = ham_to_vector(ham28, 50)

        # Start mining
        print (">> Running Opt. Hyperparameters")
        res=opt_hyperparams(model_file, generate_hyper_params_avqe, xgb.XGBRegressor, n_qubits, ham28, ham28_vec, max_size, test_static_adapt, test_semi_dynamic_adapt, test_dynamic_adapt)
        x_vec_params = np.array([float(n_qubits), 1, 1, 100, 0.001, 0, 100, 10**5, 1e-6, 5, 128, 2, 0], dtype=float)
        if (res is not None):
            dst = distance.euclidean(x_vec_params, res)
            formatted = [f"{num:.17e}" for num in res]
            lines = [", ".join(formatted[i:i+3]) for i in range(0, len(formatted), 3)]
            print("Pre-opt succeeded! Result: [")
            for i, line in enumerate(lines):
                if i == len(lines) - 1:  # Last line should not have a trailing comma
                    print("  " + line + " ]")
                else:
                    print("  " + line + ",")         
            print ("Euclidean dist:", dst)
            print ("Number of max iteration: " + str(round(res[6])))

        else:
            print ("FAILED.")
