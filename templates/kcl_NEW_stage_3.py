
"""
This script is a template for Phase 3 of the AccelerQ pipeline: **model deployment** for a
new unseen Quantum Eigensolver (QE) implementation. It optimises hyperparameters using a 
previously trained ML model and optionally applies domain-specific tests to filter candidates.
Instructions for Adaptation:
----------------------------
✅ Replace template stubs with actual QE implementation details:
  - `generate_hyper_params_template(...)` → generates candidate hyperparameter vectors.
  - `test_static_template(...)` (optional) → fast, rule-based filtering (e.g., ranges, types).
  - `test_semi_dynamic_template(...)` (optional) → deeper Hamiltonian-dependent checks.

✅ Place these in a new file named e.g., `kcl_util_<your_qe>.py` and import them here:
    from kcl_util_<your_qe> import generate_hyper_params_<your_qe>
    from kcl_tests_<your_qe> import test_static_<your_qe>, test_semi_dynamic_<your_qe>

✅ Update:
  - The `prefix` to match the system you’re targeting.
  - The `model_file` to the correct pre-trained XGBoost model name.
  - `max_size` to match the maximum padded input size from Phase 2 training, otherwise 
     there is no need to change it.
__________

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
from kcl_opt_xgb import opt_hyperparams

## TODO:
# Create a new module `kcl_util_template.py` and implement:
#   - `generate_hyper_params_template(...)`: function that produces candidate hyperparameter vectors for your QE method.
#
# Once implemented, import it here:
# from kcl_util_template import generate_hyper_params_template
#
# After customising for your QE implementation, rename both the file and function to reflect the actual method name.

## TODO:
# Implement and add test functions to `kcl_tests_template.py`.
# At minimum, define:
#   - `test_static_template(...)`: for fast, rule-based checks (e.g., parameter range, symmetry)
#   - `test_semi_dynamic_template(...)`: for deeper structural checks (e.g., Hamiltonian reduction, circuit depth)
# These tests will be used during hyperparameter search (Phase 3) to filter out invalid or suboptimal configurations.
#
# Then import them here:
# from kcl_tests_template import test_static_template, test_semi_dynamic_template

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
    prefix = "28qubits_03" #  >>>>>>>>>>>>>>>> Change only this!
    file_name = prefix + ".data"
## TODO: rename the model file name to be of the QE implementation (just for the good order)
    model_file = "model_template_pre_xgb_28.json"
    max_size = 138300

    # Get Data
    print (">> Read ham " + folder_path + ", " + file_name)
    result = process_file(folder_path, file_name)

    # Get Ham Qubit Size
    n_qubits=int(result[0])
    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))
    # Get the ham flattened once
    ham28=result[1] # Need to load from Elena's files
    ham28_vec = ham_to_vector(ham28, 50)

    # Start mining
    print (">> Running Opt. Hyperparameters")
## TODO:
# Replace `generate_hyper_params_template`, `test_static_template`, and `test_semi_dynamic_template`
# with your actual QE-specific implementations.
# If no tests, you can pass `None` for `test_static_template` and/or `test_semi_dynamic_template`.
    res=opt_hyperparams(model_file, generate_hyper_params_template, xgb.XGBRegressor, n_qubits, ham28, ham28_vec, max_size, test_static_template, test_semi_dynamic_template, None)
    formatted = [f"{num:.17e}" for num in res]
    lines = [", ".join(formatted[i:i+3]) for i in range(0, len(formatted), 3)]
    print("Pre-opt succeeded! Result: [")
    for i, line in enumerate(lines):
        if i == len(lines) - 1:  # Last line should not have a trailing comma
            print("  " + line + " ]")
        else:
            print("  " + line + ",")  
