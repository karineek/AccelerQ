
"""
This script is a wrapper for HEA-VQE implementation for phase 1 (data augmentation)

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes
  were made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.

"""

import csv
import os
from functools import partial

import numpy as np

from kcl_prepare_data import miner
from kcl_util import csv_length, determine_dataset, process_file, read_records_csv
from kcl_util_HEA import (
    OPTIMAL_HEA_FIELD_NAMES,
    compress_hea,
    generate_hyper_params_hea,
    wrapper_hea_miner,
)
from kcl_util_hamiltonian import REPO_HAMILTONIAN_DIR


def _sync_optimal_npy_fields(records_csv_file, optimal_csv_file):
    """Backfill npy_index and x_file in optimal sidecar from records CSV."""
    records = {int(r["record_id"]): r for r in read_records_csv(records_csv_file)}
    if not os.path.exists(optimal_csv_file):
        return
    updated = []
    with open(optimal_csv_file, newline="") as f:
        for row in csv.DictReader(f):
            record_id = int(row["record_id"])
            record = records.get(record_id)
            if record and record.get("status") == "success":
                row["npy_index"] = record.get("npy_index", "")
                row["x_file"] = record.get("x_file", "")
            updated.append(row)
    with open(optimal_csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPTIMAL_HEA_FIELD_NAMES, restval="")
        writer.writeheader()
        for row in updated:
            writer.writerow(row)


if __name__ == "__main__":

    print (">> Start Stage 1 (HEA-VQE)")

    np.set_printoptions(precision=17)

    DATA_DIR = '../data/HEA'
    os.makedirs(DATA_DIR, exist_ok=True)

    ham_dir = REPO_HAMILTONIAN_DIR
    prefix = "10qubits_05"  #  >>>>>>>>>>>>>>>> Change only this!
    ham_seed = int(prefix.split("_")[1])
    dataset = determine_dataset(prefix)
    file_name = prefix + ".data"
    file_X = prefix + ".X.data"
    file_Y = prefix + ".Y.data"
    ham_csv_file = 'hamiltonian_HEA.csv'
    records_csv_file = 'records_HEA.csv'
    optimal_csv_file = 'optimal_HEA.csv'

    print (">> Read ham " + ham_dir + ", " + file_name)
    n_qubits, ham = process_file(ham_dir, file_name)
    n_qubits = int(n_qubits)

    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))

    # Lower repeats than ADAPT-VQE: each sample runs full Qiskit VQE
    if n_qubits < 5:
        repeats = 20
    elif n_qubits < 8:
        repeats = 40
    elif n_qubits < 10:
        repeats = 50
    elif n_qubits < 12:
        repeats = 50
    elif n_qubits < 14:
        repeats = 25
    elif n_qubits < 16:
        repeats = 100
    else:
        repeats = 50

    if n_qubits < 6:
        timeout = 30
    elif n_qubits < 8:
        timeout = 100
    elif n_qubits < 10:
        timeout = 200
    elif n_qubits < 14:
        timeout = 300
    elif n_qubits < 18:
        timeout = 600

    print (">> Running Miner")
    prev_cwd = os.getcwd()
    os.chdir(DATA_DIR)
    try:
        ham_id = csv_length(ham_csv_file) + 1 if csv_length(ham_csv_file) > 0 else 1
        miner_wrapper = partial(
            wrapper_hea_miner,
            ham_id=ham_id,
            records_csv_file=records_csv_file,
            optimal_csv_file=optimal_csv_file,
            ham_seed=ham_seed,
            ham_dir=ham_dir,
        )
        n_mined = miner(
            n_qubits, ham, dataset, repeats, timeout,
            file_X, file_Y, ham_csv_file, records_csv_file,
            generate_hyper_params_hea, miner_wrapper, compress_hea,
        )
        _sync_optimal_npy_fields(records_csv_file, optimal_csv_file)
    finally:
        os.chdir(prev_cwd)
    print (">> Miner succeeded! Mined ", n_mined, " records")
