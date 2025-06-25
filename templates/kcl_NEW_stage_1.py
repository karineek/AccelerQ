
"""
This script is a **template wrapper** for integrating a **new Quantum Eigensolver (QE) implementation** 
into **Phase 1 (data augmentation)** of the AccelerQ pipeline.

It defines the logic for generating synthetic training data using:
- A new QE-specific parameter generator (`generate_hyper_params_*`)
- A QE wrapper to compute energy levels (`wrapper_*`)
- An optional Hamiltonian compression function (`compress_*`)

❗ **To use this template**, you must:
1. Create a new module (e.g., `kcl_util_myeq.py`) containing:
   - `generate_hyper_params_myeq`
   - `wrapper_myeq`
   - `compress_myeq`
2. Replace the TODO lines below with proper import statements from your new module.
3. Adjust the `prefix` and output filenames as needed.

---

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
from kcl_prepare_data import miner

## TODO: You need to create a new kcl_util_new.py file and include the implementation of these three functions:
#from kcl_util_template import generate_hyper_params_template, wrapper_template, compress_template
## TODO: After these are defined, you can rename from *_template to the name of the QE implementation.


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
## TODO: You can pick a different prefix instead of 'A.' for the Y and X files.

    # Get Data
    print (">> Read ham " + folder_path + ", " + file_name)
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))
    # Get the ham flattened once
    ham=result[1] # Need to load from Elena's files

## TODO: You can explore these parameters based on the size of the data sampled (if too large, you can push down these numbers, but you do not have to).
    # According to the qubits, adjust the repeats - for performance
    if n_qubits < 5:
        repeats=10
    elif n_qubits < 10:
        repeats=250
    elif n_qubits < 12:
        repeats=500
    else:
        repeats=1000 # Performance - we will need to split it into many

    # Start mining
    print (">> Running Miner")
  
## TODO: rename generate_hyper_params_template, wrapper_template, compress_template to the names you imported above in line 36 of this template
    res=miner(n_qubits, ham, repeats, 660, X_file, Y_file, generate_hyper_params_template, wrapper_template, compress_template)
    print (">> Miner succeeded! Mined ", res, " records")
