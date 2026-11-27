
"""
Stage 2b: train the NN warm-start model on HEA Stage 1 records.

Reads records_HEA.csv, optimal_HEA.csv, and HEA .npy files produced by
kcl_HEA_stage_1.py. NN-HEA shares the HEA ansatz and mining distribution.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import csv
import os
import time

from kcl_NN_model import train_nn_model

STAGE2B_FIELD_NAMES = ['ham_prefix', 'n_qubits', 'train_time_s']


def append_stage2b_profiling(csv_path, row_dict):
  file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
  mode = 'a' if file_exists else 'w'
  with open(csv_path, mode, newline='') as f:
    writer = csv.DictWriter(
      f,
      fieldnames=STAGE2B_FIELD_NAMES,
      restval='',
      extrasaction='ignore',
    )
    if not file_exists:
      writer.writeheader()
    writer.writerow(row_dict)


if __name__ == "__main__":

  print(">> Start Stage 2b (NN-HEA): Neural network training")

  DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "NN_HEA")
  HEA_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "HEA")
  os.makedirs(DATA_DIR, exist_ok=True)

  records_csv_path = os.path.join(HEA_DATA_DIR, "records_HEA.csv")
  optimal_csv_path = os.path.join(HEA_DATA_DIR, "optimal_HEA.csv")
  ham_csv_path = os.path.join(HEA_DATA_DIR, "hamiltonian_HEA.csv")
  nn_model_path = os.path.join(DATA_DIR, "nn_hea_model.pt")

  train_start = time.perf_counter()
  ham_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hamiltonian"))
  train_nn_model(
    records_csv_path,
    HEA_DATA_DIR,
    optimal_csv_path,
    ham_csv_path,
    nn_model_path,
    n_epochs=200,
    ham_dir=ham_dir,
  )
  train_elapsed = time.perf_counter() - train_start
  print(">> NN model saved to", nn_model_path)
  append_stage2b_profiling(
    os.path.join(DATA_DIR, 'stage2b_profiling_NN_HEA.csv'),
    {
      'ham_prefix': 'ALL',
      'n_qubits': '',
      'train_time_s': train_elapsed,
    },
  )
