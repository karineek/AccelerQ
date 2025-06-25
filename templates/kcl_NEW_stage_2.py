
"""
This script is a **template wrapper** for integrating a **new Quantum Eigensolver (QE) implementation**
into **Phase 2 (model construction)** of the AccelerQ pipeline.

It performs the following:
- Trains an XGBoost regressor on Phase 1-generated data
- Saves the trained model for use in Phase 3 (hyperparameter optimisation)

❗ **To use this template**, you must:
1. Adjust the `prefix` and `model_file` to reflect your QE implementation.
2. Ensure that the required Phase 1 `.X.data.npy` and `.Y.data.npy` files are present in `../data/`.

Note:
- You can switch to GPU training by modifying the `cpu` flag.
____

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

    ### KEEP THIS PART AS IS, UNLESS YOU WISH TO USE LARGER SYSTEMS (above 28 qubits, and then the numbers should be adjusted)
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
    # Get the ham flattened once
    ham28=result[1] # Need to load from Elena's files

    # According to the qubits, adjust the repeats
    if n_qubits < 20:
        print (">> Train only on 20+ qubits!")
        exit
    ### END OF KEEP THIS PART AS IS PART
  
    # Model + traininng data
## TODO: rename the model file name to be of the QE implementation (just for the good order)
    model_file="model_template_pre_xgb_" + str(n_qubits) + ".json"
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
