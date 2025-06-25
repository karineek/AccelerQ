
"""
This script processes Hamiltonian systems and extracts size before and after truncation 
under different parameters. It iterates over a set of Hamiltonian datasets. The script 
outputs detailed numerical results for the size of systems (Hamiltonians) graph/table 
in the evaluation.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import numpy as np
from kcl_util import process_file, ham_to_vector
from kcl_train_xgb import train

import xgboost as xgb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":

    print (">> Start Stage 2")

    # Set precision
    np.set_printoptions(precision=17)

    # inputs
    folder_path = "../hamiltonian/"
    prefix = "28qubits_01" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    # Get Data
    print (">> Read ham " + folder_path + ", " + file_name)
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))
    # Get the ham flatten once
    ham28=result[1] # Need to load from Elena's files

    # According to the qubits adjust the repeats
    if n_qubits < 20:
        print (">> Train only on 20+ qubits!")
        exit

    # Model + traininng data
    model_file="model_qcels_pre_xgb_" + str(n_qubits) + ".json"
    folder="../data/"
    ham28_vec = ham_to_vector(ham28, 50)
    # Parameters
    params_ml = {
        'objective': 'reg:pseudohubererror',
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 100,     # Number of trees to fit
        'alpha': 0.1,           # L1 regularization term on weights
        'lambda': 0.1           # L2 regularization term on weights
    }

    # Start mining
    print (">> Start Training")
    cpu=1 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Change when using a GPU
    max_size = train(folder, model_file, ham28_vec, xgb.XGBRegressor, cpu, params_ml, 13)
    print (">> End Training. Max Size is ")
    print (max_size)
