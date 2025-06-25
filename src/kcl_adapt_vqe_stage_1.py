"""
This script is a wrapper for ADPT-QSCI QE implementation for phase 1 (data augmentation)

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import numpy as np
from kcl_util import process_file
from kcl_util_adapt_vqe import generate_hyper_params_avqe, wrapper_avqe, compress_avqe
from kcl_prepare_data import miner

if __name__ == "__main__":

    print (">> Start Stage 1")

    # Set precision
    np.set_printoptions(precision=17)

    # inputs
    folder_path = "../hamiltonian/"
    prefix = "16qubits_05" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    X_file = prefix + ".X.data"
    Y_file = prefix + ".Y.data"

    # Get Data
    print (">> Read ham " + folder_path + ", " + file_name)
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))
    # Get the ham flatten once
    ham=result[1] # Need to load from Elena's files

    # According to the qubits adjust the repeats - for performance
    if n_qubits < 5:
        repeats=10
    elif n_qubits < 10:
        repeats=250
    elif n_qubits < 12:
        repeats=100
    elif n_qubits < 14:
        repeats=50
    elif n_qubits < 16:
        repeats=250
    else:
        repeats=100 # Performance - we will need to split it to many

    # Start mining
    print (">> Running Miner")
    res=miner(n_qubits, ham, repeats, 660, X_file, Y_file, generate_hyper_params_avqe, wrapper_avqe, compress_avqe)
    print (">> Miner succeeded! Mined ", res, " records")
