import os
import csv
import time

import numpy as np
from kcl_util import process_file, ham_to_vector
from kcl_train_xgb import train
import xgboost as xgb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# XGBoost compute device: set ACCELERQ_XGB_DEVICE=cuda on GPU Slurm jobs; default cpu.
XGB_DEVICE = os.environ.get('ACCELERQ_XGB_DEVICE', 'cpu')

STAGE2_FIELD_NAMES = ['ham_prefix', 'n_qubits', 'train_time_s', 'max_size']


def append_stage2_profiling(csv_path, row_dict):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STAGE2_FIELD_NAMES,
                                restval='', extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


if __name__ == '__main__':
    print (">> Start Stage 2 (UCCSD-VQE)")

    np.set_printoptions(precision=17)

    ham_dir = '../hamiltonian'
    prefix = '24qubits_05'
    file_name = prefix + '.data'

    print (">> Read ham " + ham_dir + ", " + file_name)

    n_qubits, ham = process_file(ham_dir, file_name)
    n_qubits = int(n_qubits)

    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))

    if n_qubits < 20:
        print (">> Train only on 20+ qubits!")
        exit()

    ham_vec = ham_to_vector(ham, 50)
    DATA_DIR = '../data/UCCSD'
    os.makedirs(DATA_DIR, exist_ok=True)
    model_file = os.path.join(DATA_DIR, 'model_uccsd_pre_xgb_' + str(n_qubits) + '.json')
    params_ml = {
        'objective': 'reg:pseudohubererror',
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'alpha': 0.1,
        'lambda': 0.1,
        'tree_method': 'hist',
        'device': XGB_DEVICE,
    }
    records_csv_file = os.path.join(DATA_DIR, 'records_UCCSD.csv')
    
    print(">> Start Training")
    print(f">> XGBoost device: {XGB_DEVICE}")
    chunked_fit = 0  # NOTE: unrelated to XGBoost device (see XGB_DEVICE). Toggles chunked incremental fit in kcl_train_xgb.py.
    train_start = time.perf_counter()
    max_size = train(DATA_DIR, model_file, ham_vec, xgb.XGBRegressor, chunked_fit, params_ml, records_csv_file, params_q_size=14)
    train_elapsed = time.perf_counter() - train_start
    print(f">> End Training. Max Size is {max_size}. Time: {train_elapsed:.2f}s")
    append_stage2_profiling(
        os.path.join(DATA_DIR, 'stage2_profiling_UCCSD.csv'),
        {
            'ham_prefix': prefix,
            'n_qubits': n_qubits,
            'train_time_s': train_elapsed,
            'max_size': max_size,
        },
    )
