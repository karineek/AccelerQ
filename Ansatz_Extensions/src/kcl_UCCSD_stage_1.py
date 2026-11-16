import os
from functools import partial

import numpy as np

from kcl_util import process_file, determine_dataset
from kcl_prepare_data import miner
from kcl_util_UCCSD import wrapper_uccsd, generate_hyper_params_uccsd, compress_uccsd
from kcl_util_hamiltonian import REPO_HAMILTONIAN_DIR

if __name__ == '__main__':

    np.set_printoptions(precision=17)

    DATA_DIR = '../data/UCCSD'
    os.makedirs(DATA_DIR, exist_ok=True)

    ham_dir = REPO_HAMILTONIAN_DIR
    prefix = "10qubits_05" #  >>>>>>>>>>>>>>>> Change only this!
    ham_seed = int(prefix.split('_')[1])
    dataset = determine_dataset(prefix)
    file_name = prefix + '.data'

    print (">> Read ham " + ham_dir + ", " + file_name)
    
    n_qubits, ham = process_file(ham_dir, file_name)
    n_qubits = int(n_qubits)

    print (">> Start processing: "+file_name+" with qubits "+str(n_qubits))

    file_X = prefix + 'A.X.data'
    file_Y = prefix + 'A.Y.data'
    ham_csv_file = 'hamiltonian_UCCSD.csv'
    records_csv_file = 'records_UCCSD.csv'

    if n_qubits > 2 and n_qubits < 6:
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
        timeout = 60
    if n_qubits < 8:
        timeout = 200
    elif n_qubits < 10:
        timeout = 400
    elif n_qubits < 14:
        timeout = 600
    elif n_qubits < 18:
        timeout = 600
    
    print (">> Running Miner")
    miner_wrapper = partial(
        wrapper_uccsd,
        ham_seed=ham_seed,
        ham_dir=ham_dir,
    )
    prev_cwd = os.getcwd()
    os.chdir(DATA_DIR)
    try:
        n_mined = miner(n_qubits, ham, dataset, repeats, timeout, file_X, file_Y, ham_csv_file,
                    records_csv_file, generate_hyper_params_uccsd, miner_wrapper, compress_uccsd)
    finally:
        os.chdir(prev_cwd)
    print (">> Miner succeeded! Mined ", n_mined, " records")
