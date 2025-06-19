import sys
import os
import numpy as np
from kcl_util import process_file
from kcl_util_qcels import generate_hyper_params_qcels, wrapper_qcels, compress_qcels
from kcl_prepare_data import miner

if __name__ == "__main__":

    print (">> Start Stage 1")

    # Set precision
    np.set_printoptions(precision=17)

    # inputs
    folder_path = "../hamiltonian/"
    prefix = "16qubits_06" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    X_file = prefix + "A.X.data"
    Y_file = prefix + "A.Y.data"

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
        repeats=500
    else:
        repeats=1000 # Performance - we will need to split it to many

    # Start mining
    print (">> Running Miner")
    res=miner(n_qubits, ham, repeats, 660, X_file, Y_file, generate_hyper_params_qcels, wrapper_qcels, compress_qcels)
    print (">> Miner succeeded! Mined ", res, " records")
