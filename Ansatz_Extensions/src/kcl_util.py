
"""
This script contains a set of general utilities and Hamiltonian preprocessing functions
for the AccelerQ pipeline.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import glob
import tracemalloc
import numpy as np
import csv
from datetime import datetime
#import joblib
#import pickle

from openfermion import FermionOperator as FO
from openfermion import load_operator
sys.path.append("../")

from utils.challenge_2024 import problem_hamiltonian # Needed for process_file - can remove if discard it.




#===========================================
# FUNCTIONS for Reading and Writing to Files
#===========================================


def save_to_binary(file_name, data):
    """Save array/list to ``.npy`` via NumPy."""
    if isinstance(data, list):
        data = np.array(data)
    np.save(file_name, data)
    print(f"Data written successfully to {file_name}")

def load_from_binary(file_name):
    """Load array from ``.npy`` via NumPy."""
    data = np.load(file_name)
    print(f"Data loaded successfully from {file_name}")
    return data

def simple_load_ham(folder, file):
    """Load a FermionOperator from OpenFermion binary on disk."""
    print(">> Load: " + folder + "," + file)
    ham = load_operator(
        file_name=file,
        data_directory=folder,
        plain_text=False
    )
    return ham

def process_file(folder_path, file_name):
    """Load challenge Hamiltonian; returns (file_prefix, FermionOperator)."""
    file_path = os.path.join(folder_path, file_name)
    postfix = file_name[-7:]
    ham = problem_hamiltonian(file_name[:2], postfix[:2], folder_path)
    return file_name[:2], ham

def print_to_file(message):
    """Append ``message`` to logger.txt with a timestamp."""
    with open("logger.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def load_and_split_data(files):
    """Load each ``.npy`` and split rows 90/10 into (set90, set10)."""
    set90 = []
    set10 = []

    for file in files:
        data = load_from_binary(file)

        split_index = int(len(data) * 0.9)
        data_90 = data[:split_index]
        data_10 = data[split_index:]

        for x in data_90:
            set90.append(x)
        for x in data_10:
            set10.append(x)

        print (">>> Stat. >>> Size of set loaded from: " + file + " is: " + str(len(data_90)))

    return set90, set10

def list_x_npy_files(folder):
    """Return sorted paths for folder/*.X.data.npy (matches UCCSD *A.X.data.npy)."""
    return sorted(glob.glob(os.path.join(folder, "*.X.data.npy")))


def list_y_npy_files(folder):
    """Return Y paths paired with sorted X paths."""
    files_y = []
    for x_path in list_x_npy_files(folder):
        y_path = x_path.replace(".X.data.npy", ".Y.data.npy")
        if not os.path.exists(y_path):
            raise ValueError("Missing paired Y file for: " + x_path)
        files_y.append(y_path)
    return files_y


def x_npy_stem(path):
    """Basename without .npy suffix, e.g. 04qubits_05A.X.data"""
    basename = os.path.basename(path)
    if basename.endswith(".npy"):
        return basename[:-4]
    return basename


def load_data_set(folder):
    """Load all X/Y ``.npy`` pairs and split each file 90/10 into big/small arrays."""
    filesX = list_x_npy_files(folder)
    filesY = list_y_npy_files(folder)

    Xbig, Xsmall = load_and_split_data(filesX)
    Ybig, Ysmall = load_and_split_data(filesY)

    if len(Xbig) != len(Ybig):
        print("size of X: " + str(len(Xbig)))
        print("size of Y: " + str(len(Ybig)))
        raise ValueError("Size of X and Y must be equal")

    if len(Xsmall) != len(Ysmall):
        print("size of X (small): " + str(len(Xsmall)))
        print("size of Y (small): " + str(len(Ysmall)))
        raise ValueError("Size of X and Y (small) must be equal")

    print ("Size of Xbig and Ybig: " + str(len(Xbig)) + ", " + str(len(Ybig)))
    return Xbig, Ybig, Xsmall, Ysmall

def save_model(model_file, model):
    model.save_model(model_file)

def load_model(model_file, type):
    model = type()
    model.load_model(model_file)
    return model

def csv_length(csv_file):
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, newline='') as f:
        rows = list(csv.DictReader(f))
        return len(rows)


MINING_PROFILE_FIELD_NAMES = [
    'peak_traced_bytes', 'rss_before_bytes', 'rss_after_bytes', 'rss_peak_bytes',
]

RECORD_FIELD_NAMES = [
    'record_id', 'ham_id', 'mining_time', 'true_energy',
    'status', 'npy_index', 'x_file',
    *MINING_PROFILE_FIELD_NAMES,
    'predicted_energy', 'split', 'squared_error',
]


def _rss_bytes():
    """Process RSS in bytes; None if psutil unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return None


def _mining_prof_dict(peak_traced, rss_before, rss_after):
    rss_peak = None
    if rss_before is not None and rss_after is not None:
        rss_peak = max(rss_before, rss_after)
    elif rss_after is not None:
        rss_peak = rss_after
    elif rss_before is not None:
        rss_peak = rss_before
    return {
        'peak_traced_bytes': peak_traced,
        'rss_before_bytes': rss_before,
        'rss_after_bytes': rss_after,
        'rss_peak_bytes': rss_peak,
    }


def mining_profile_csv_fields(prof):
    """Map profiling dict to CSV row fields (empty string when missing)."""
    if not prof:
        return {name: '' for name in MINING_PROFILE_FIELD_NAMES}
    return {
        name: prof.get(name) if prof.get(name) is not None else ''
        for name in MINING_PROFILE_FIELD_NAMES
    }


def profile_mining_call(callable_fn, *args, **kwargs):
    """Run callable under tracemalloc+RSS; returns (result, prof_dict, error|None)."""
    rss_before = _rss_bytes()
    tracemalloc.start()
    result = None
    error = None
    try:
        result = callable_fn(*args, **kwargs)
    except Exception as exc:
        error = exc
    finally:
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss_after = _rss_bytes()
    prof = _mining_prof_dict(peak_traced, rss_before, rss_after)
    return result, prof, error


def init_or_append_csv(path, fieldnames, row_dict):
    """Create CSV with header on first write; append row on subsequent writes."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = 'a' if file_exists else 'w'
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def read_records_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_records_csv(path, rows, fieldnames=None):
    fieldnames = fieldnames or RECORD_FIELD_NAMES
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_npy_row_count(path):
    return len(np.load(path))


def build_npy_index_maps(folder):
    """Position→(stem, npy_index) maps for Xbig/X_extra using load_data_set file order."""
    xbig_map = {}
    xextra_map = {}
    file_meta = {}
    xbig_pos = 0
    xextra_pos = 0

    for path in list_x_npy_files(folder):
        stem = x_npy_stem(path)
        n_rows = _load_npy_row_count(path)
        split_index = int(n_rows * 0.9)
        file_meta[stem] = {'n_rows': n_rows, 'split_index': split_index}

        for npy_index in range(split_index):
            xbig_map[xbig_pos] = (stem, npy_index)
            xbig_pos += 1

        for npy_index in range(split_index, n_rows):
            xextra_map[xextra_pos] = (stem, npy_index)
            xextra_pos += 1

    return xbig_map, xextra_map, file_meta


def _records_lookup(rows):
    """Map (x_file, npy_index) -> row dict reference."""
    lookup = {}
    for row in rows:
        if row.get('status') == 'failed':
            continue
        x_file = row.get('x_file', '')
        npy_index = row.get('npy_index', '')
        if x_file and npy_index != '':
            lookup[(x_file, str(npy_index))] = row
    return lookup


def label_test_extra_split(records_csv_file, folder):
    """Label per-file last 10% npy rows as split=test_extra (matches load_and_split_data)."""
    if not os.path.exists(records_csv_file) or csv_length(records_csv_file) == 0:
        print(">> Skipping test_extra CSV labeling: empty or missing " + records_csv_file)
        return

    _, _, file_meta = build_npy_index_maps(folder)
    rows = read_records_csv(records_csv_file)

    for row in rows:
        if row.get('status') == 'failed':
            continue
        stem = row.get('x_file', '')
        npy_index_str = row.get('npy_index', '')
        if not stem or npy_index_str == '':
            continue
        meta = file_meta.get(stem)
        if meta is None:
            continue
        if int(npy_index_str) >= meta['split_index']:
            row['split'] = 'test_extra'

    write_records_csv(records_csv_file, rows)


def update_train_test_predictions(
    records_csv_file, folder, idx_train, idx_test,
    y_pred_test, y_true_test, y_pred_extra, y_true_extra,
):
    """Update records CSV with train/test splits and prediction errors."""
    if not os.path.exists(records_csv_file) or csv_length(records_csv_file) == 0:
        print(">> Skipping train/test CSV updates: empty or missing " + records_csv_file)
        return

    xbig_map, xextra_map, _ = build_npy_index_maps(folder)
    rows = read_records_csv(records_csv_file)
    lookup = _records_lookup(rows)

    for idx in idx_train:
        key = xbig_map.get(idx)
        if key is None:
            continue
        row = lookup.get((key[0], str(key[1])))
        if row is not None and row.get('split') != 'test_extra':
            row['split'] = 'train'

    for i, idx in enumerate(idx_test):
        key = xbig_map.get(idx)
        if key is None:
            continue
        row = lookup.get((key[0], str(key[1])))
        if row is not None and row.get('split') != 'test_extra':
            row['split'] = 'test'
            pred = float(y_pred_test[i])
            true = float(y_true_test[i])
            row['predicted_energy'] = pred
            row['squared_error'] = (pred - true) ** 2

    for i in range(len(xextra_map)):
        key = xextra_map[i]
        row = lookup.get((key[0], str(key[1])))
        if row is None:
            continue
        row['split'] = 'test_extra'
        pred = float(y_pred_extra[i])
        true = float(y_true_extra[i])
        row['predicted_energy'] = pred
        row['squared_error'] = (pred - true) ** 2

    write_records_csv(records_csv_file, rows)


#==========================================
# FUNCTIONS for Quantum to ML represntation
#==========================================


def flatten(nested_list):
    """Recursively flatten nested list/tuple to floats."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(float(item))
    return flat_list

def ham_to_vector(ham, reduce=1.0):
    """Legacy flat coeff vector from FermionOperator (prefer get_hamiltonian_features)."""
    vector = []

    if isinstance(ham, FO):
        terms = ham.terms
        vector = flatten(flatten(terms))
        vector = [element for element in vector if not (-(reduce) <= element <= reduce)]
        # pick top 1000 - check on ABS
        sorted_vector = np.sort(vector)

        for term, coefficient in terms.items():
            vector.append(float(coefficient.real))

        vector = [element for element in vector if not (-0.2 <= element <= 0.2)]
        vector = np.array(flatten(vector), order='C', dtype=float)

        print(">>>> adding ham of size " + str(len(terms)))

    else:
        typo = type(ham).__name__
        raise TypeError("Expected ham to be a FermionOperator, but got {}".format(typo))

    return vector

def vec_to_fixed_size_vec(size, vec_in):
    """Zero-pad vec to ``size``; raise if longer than ``size``."""
    vec = np.array(vec_in)

    if len(vec) < size:
        return np.pad(vec, (0, size - len(vec)), 'constant')

    elif len(vec) > size:
        raise ValueError("Size of X - padding failed. Vec larger than max size: " + str(size))

    else:
        return vec

def determine_dataset(prefix):
    """Map ham prefix suffix to QunaSys (<5) or Molecular_Opensource (>=5)."""

    suffix = int(prefix[-2:])

    if suffix < 0:
        raise ValueError('Hamiltonian instance label is invalid!')

    if suffix < 5:
        return 'QunaSys'
    else:
        return 'Molecular_Opensource'
