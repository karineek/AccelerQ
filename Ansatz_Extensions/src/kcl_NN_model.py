
"""
Neural network model for NN-HEA warm-start (hyperparameters + rotation angles).

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
"""

import csv
import os
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from kcl_util import build_npy_index_maps, read_records_csv
from kcl_util_hamiltonian import (
  HAMILTONIAN_FEATURE_LENGTH,
  REPO_HAMILTONIAN_DIR,
  get_hamiltonian_features,
)

NN_SEED = -1
NN_INPUT_DIM = HAMILTONIAN_FEATURE_LENGTH  # fixed-length coeff vector
NN_HYPERPARAM_DIM = 12
NN_ANGLE_DIM = 336
NN_OUTPUT_DIM = NN_HYPERPARAM_DIM + NN_ANGLE_DIM
NN_HIDDEN_DIM = 8

_X_FILE_PREFIX_RE = re.compile(r"^(.+)\.X\.data$")


class NNHEAModel(nn.Module):
  """Maps Hamiltonian coefficient features to hyperparams (Part A) and angles (Part B)."""

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(NN_INPUT_DIM, NN_HIDDEN_DIM),
      nn.ReLU(),
      nn.Linear(NN_HIDDEN_DIM, NN_OUTPUT_DIM),
    )

  def forward(self, x):
    return self.net(x)


def _parse_optimal_angles(raw: str) -> np.ndarray:
  parts = [float(v) for v in raw.split(",") if v.strip() != ""]
  angles = np.zeros(NN_ANGLE_DIM, dtype=np.float64)
  n = min(len(parts), NN_ANGLE_DIM)
  angles[:n] = parts[:n]
  return angles


def _prefix_from_x_file(x_file: str) -> str:
  """Map records x_file stem (e.g. 04qubits_05.X.data) to ham prefix 04qubits_05."""
  stem = (x_file or "").strip()
  match = _X_FILE_PREFIX_RE.match(stem)
  if match:
    return match.group(1)
  # Fallback: strip trailing .npy / .X.data fragments
  if stem.endswith(".npy"):
    stem = stem[:-4]
  if stem.endswith(".X.data"):
    return stem[: -len(".X.data")]
  return stem


def _load_ham_features_by_id(records, ham_dir: str) -> dict:
  """ham_id → fixed-length coeffs; lazy-imports process_file (juliacall)."""
  from kcl_util import process_file

  prefix_by_id = {}
  for row in records:
    if row.get("status") != "success":
      continue
    try:
      ham_id = int(row["ham_id"])
    except (TypeError, ValueError, KeyError):
      continue
    if ham_id in prefix_by_id:
      continue
    x_file = row.get("x_file", "")
    prefix = _prefix_from_x_file(x_file)
    if prefix:
      prefix_by_id[ham_id] = prefix

  features = {}
  for ham_id, prefix in sorted(prefix_by_id.items()):
    data_name = f"{prefix}.data"
    data_path = os.path.join(ham_dir, data_name)
    if not os.path.isfile(data_path):
      print(f"  Warning: missing {data_path} — skip ham_id={ham_id}")
      continue
    try:
      _, ham = process_file(ham_dir, data_name)
      features[ham_id] = np.asarray(
        get_hamiltonian_features(ham), dtype=np.float64
      )
    except Exception as exc:
      print(f"  Warning: feature extract failed for {prefix}: {exc}")
  return features


def _load_optimal_by_record_id(optimal_csv_path: str) -> dict:
  optimal = {}
  if not os.path.exists(optimal_csv_path):
    return optimal
  with open(optimal_csv_path, newline="") as f:
    for row in csv.DictReader(f):
      record_id = int(row["record_id"])
      optimal[record_id] = _parse_optimal_angles(row["optimal_angles"])
  return optimal


def _load_x_vec_from_npy(data_dir: str, x_file: str, npy_index: int) -> np.ndarray:
  path = os.path.join(data_dir, x_file + ".npy")
  rows = np.load(path)
  return np.array(rows[int(npy_index)][:14], dtype=np.float64)


def train_nn_model(
  records_csv_path,
  data_dir,
  optimal_csv_path,
  ham_csv_path,
  model_save_path,
  n_epochs=200,
  top_k=5,
  ham_dir=None,
  weight_decay=1e-2
):
  """Train NNHEAModel from Stage 1 top_k-by-energy records; ham_csv_path unused."""
  del ham_csv_path  # unused; features recomputed from .data files
  if ham_dir is None:
    ham_dir = REPO_HAMILTONIAN_DIR

  records = read_records_csv(records_csv_path)
  ham_features_by_id = _load_ham_features_by_id(records, ham_dir)
  optimal_by_record = _load_optimal_by_record_id(optimal_csv_path)
  _, _, file_meta = build_npy_index_maps(data_dir)

  by_ham = {}
  for row in records:
    if row.get("status") != "success":
      continue
    record_id = int(row["record_id"])
    ham_id = int(row["ham_id"])
    if record_id not in optimal_by_record:
      continue
    try:
      energy = float(row["true_energy"])
    except (TypeError, ValueError):
      continue
    by_ham.setdefault(ham_id, []).append((energy, record_id, row))

  samples_x = []
  samples_y = []
  for ham_id, entries in by_ham.items():
    if ham_id not in ham_features_by_id:
      continue
    entries.sort(key=lambda t: t[0])
    for _, record_id, row in entries[:top_k]:
      x_file = row.get("x_file", "")
      npy_index = row.get("npy_index", "")
      if not x_file or npy_index == "":
        continue
      if x_file not in file_meta:
        continue
      x_vec = _load_x_vec_from_npy(data_dir, x_file, int(npy_index))
      y_hyper = x_vec[2:14].copy()
      y_hyper[10] = 0.0  # IND:12 placeholder in training target
      y_angles = optimal_by_record[record_id]
      y = np.concatenate([y_hyper, y_angles])
      samples_x.append(ham_features_by_id[ham_id])
      samples_y.append(y)

  if len(samples_x) == 0:
    raise ValueError("No training samples found for NN-HEA model")

  X = np.array(samples_x, dtype=np.float64)
  Y = np.array(samples_y, dtype=np.float64)
  if X.shape[1] != NN_INPUT_DIM:
    raise ValueError(
      f"Expected feature dim {NN_INPUT_DIM}, got {X.shape[1]}"
    )

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  model = NNHEAModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
  criterion = nn.MSELoss()

  X_t = torch.tensor(X_scaled, dtype=torch.float32)
  Y_t = torch.tensor(Y, dtype=torch.float32)

  for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_t)
    loss = criterion(pred, Y_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0 or epoch == 0:
      print(f"Epoch {epoch + 1}/{n_epochs}, train loss: {loss.item():.6f}")

  os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
  torch.save(
    {
      "model": model.state_dict(),
      "scaler_mean": scaler.mean_.astype(np.float64),
      "scaler_std": scaler.scale_.astype(np.float64),
      "input_dim": NN_INPUT_DIM,
    },
    model_save_path,
  )
  print(f">> NN model saved to {model_save_path} (input_dim={NN_INPUT_DIM})")


def load_nn_model(model_save_path):
  checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)
  model = NNHEAModel()
  model.load_state_dict(checkpoint["model"])
  model.eval()
  return model, checkpoint["scaler_mean"], checkpoint["scaler_std"]


def _scale_features(ham_features, scaler_mean, scaler_std):
  x = np.asarray(ham_features, dtype=np.float64).reshape(1, -1)
  std = np.where(scaler_std == 0, 1.0, scaler_std)
  return (x - scaler_mean) / std


def _forward_raw(model, scaler_mean, scaler_std, ham_features):
  x_scaled = _scale_features(ham_features, scaler_mean, scaler_std)
  with torch.no_grad():
    out = model(torch.tensor(x_scaled, dtype=torch.float32))
  return out.numpy().reshape(-1)


def predict_x_vec(model, scaler_mean, scaler_std, ham_features, n_qubits):
  from kcl_util_NN_HEA import sanitize_x_vec_params_nn_hea

  raw_out = _forward_raw(model, scaler_mean, scaler_std, ham_features)
  part_a = raw_out[:NN_HYPERPARAM_DIM]
  x_vec = np.array(
    [
      float(n_qubits),
      1.0,
      part_a[0],
      part_a[1],
      part_a[2],
      part_a[3],
      part_a[4],
      part_a[5],
      part_a[6],
      part_a[7],
      part_a[8],
      part_a[9],
      float(NN_SEED),
      part_a[11],
    ],
    dtype=float,
  )
  return sanitize_x_vec_params_nn_hea(x_vec)


def predict_angles(model, scaler_mean, scaler_std, ham_features, n_params):
  raw_out = _forward_raw(model, scaler_mean, scaler_std, ham_features)
  n_params = int(n_params)
  return raw_out[NN_HYPERPARAM_DIM : NN_HYPERPARAM_DIM + n_params]
