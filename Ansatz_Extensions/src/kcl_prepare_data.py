
"""
This script is responsible for generating training data by evaluating
random parameterisations of a quantum eigensolver implementation against a given Hamiltonian.

This is the code of Phase 1 - data augmentation logic.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import time
from kcl_util import (
    save_to_binary,
    ham_to_vector,
    print_to_file,
    csv_length,
    init_or_append_csv,
    RECORD_FIELD_NAMES,
    profile_mining_call,
    mining_profile_csv_fields,
)
from kcl_util_hamiltonian import compute_hamiltonian_features
import numpy as np
import stopit

HAM_FIELD_NAMES = [
    'ham_id', 'n_qubits', 'dataset', 'sparsity', 'term_count', 'coefficient_variance',
]


def miner(n_qubits, ham, dataset, repeats, timeout, file_X, file_Y, ham_csv_file, records_csv_file, generator_caller, wrapper_caller, compress_caller):
    np.set_printoptions(precision=17)

    X = []
    Y = []

    ham_vec = ham_to_vector(ham)

    ham_features = compute_hamiltonian_features(ham)
    ham_id = csv_length(ham_csv_file) + 1 if csv_length(ham_csv_file) > 0 else 1

    init_or_append_csv(ham_csv_file, HAM_FIELD_NAMES, {
        'ham_id': ham_id,
        'n_qubits': n_qubits,
        'dataset': dataset,
        'sparsity': ham_features[0],
        'term_count': ham_features[1],
        'coefficient_variance': ham_features[2],
    })

    repeats = max(repeats, 5)
    timeout = max(timeout, 20)

    orig = str(len(ham.terms))
    ham = compress_caller(ham)
    print_to_file(">>> compress to " + str(len(ham.terms)) + " from " + orig)

    x_file_stem = file_X
    npy_index_counter = 0

    for i in range(0, repeats):
        start = time.perf_counter()

        print_to_file(">>>> Collecting data with qubits "+str(n_qubits) + " iteration " + str(i))
        x_vec_params = generator_caller(i, n_qubits)
        y_vec = 0.0
        status = 'failed'
        npy_index = ''
        prof = {}

        try:
            with stopit.ThreadingTimeout(timeout) as context_manager:
                y_vec, prof, mining_exc = profile_mining_call(
                    wrapper_caller, x_vec_params, n_qubits, ham,
                )
                if mining_exc is not None:
                    raise mining_exc

            if context_manager.state == context_manager.TIMED_OUT:
                y_vec = 3.0
                status = 'timeout'
                print_to_file("Y is 3.0 due to timeout")
            else:
                status = 'success'

            x_vec = np.append(x_vec_params, ham_vec)
            X.append(x_vec)
            Y.append(y_vec)
            npy_index = npy_index_counter
            npy_index_counter += 1
            print_to_file(">>>> Register data, X size of " + str(len(x_vec)) + " with energy level " + str(y_vec))

        except Exception as e:
            print(e)
            print_to_file("Y is 0 due to exception")

        print_to_file(">> End Collecting Data Samples, y:= " + str(y_vec))
        print_to_file(x_vec_params)

        mining_time = time.perf_counter() - start
        record_id = csv_length(records_csv_file) + 1 if csv_length(records_csv_file) > 0 else 1

        row = {
            'record_id': record_id,
            'ham_id': ham_id,
            'mining_time': mining_time,
            'true_energy': y_vec if status != 'failed' else '',
            'status': status,
            'npy_index': npy_index if status != 'failed' else '',
            'x_file': x_file_stem if status != 'failed' else '',
            **mining_profile_csv_fields(prof),
        }
        init_or_append_csv(records_csv_file, RECORD_FIELD_NAMES, row)

    print_to_file(">> End Mining Phase. With " + str(len(X)) + " training data")
    if npy_index_counter != len(X):
        raise ValueError("npy_index_counter != len(X)")
    if len(X) != len(Y):
        print("size of X: " + str(len(X)))
        print("size of Y: " + str(len(Y)))
        raise ValueError("Size of X and Y must be equal")

    save_to_binary(file_X, X)
    save_to_binary(file_Y, Y)

    return len(X)
