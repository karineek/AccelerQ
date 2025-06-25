
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
import numpy as np
from kcl_util import process_file, ham_to_vector

if __name__ == "__main__":
    # Set precision
    np.set_printoptions(precision=17)

    # Ensure the user provides the prefix as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <prefix>")
        sys.exit(1)

    # Get the prefix from the command-line arguments
    prefix = sys.argv[1]

    # inputs
    folder_path = "../hamiltonian/"
    # prefix = "20qubits_01" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
    # Get Data
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    # Get the ham flatten once
    ham28=result[1] # Need to load from Elena's files
    ham28_vec = ham_to_vector(ham28, 50)

    print ("GREPME:" + prefix + " | (orig)" + str(len(ham28.terms)) + " | (compressed)" + str(len(ham28_vec)))
