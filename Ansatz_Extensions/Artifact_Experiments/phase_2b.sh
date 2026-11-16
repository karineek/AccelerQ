#!/bin/bash
# NN-HEA only: train nn_hea_model.pt after phase_2_short.sh (HEA XGBoost).
# Requires records_HEA.csv, hamiltonian_HEA.csv, optimal_HEA.csv,
# and {prefix}.X/Y.data.npy from a prior phase_1_short.sh HEA run. Needs torch.
set -e
echo ">> Phase 2b (NN-HEA) starting..."

cd ../src
DATA_NN_HEA=../data/NN_HEA
mkdir -p "$DATA_NN_HEA"

# Remove stale NN model so stage 2b always retrains from current HEA Stage 1 data
rm -f "$DATA_NN_HEA/nn_hea_model.pt"

echo ">> Running kcl_NN_HEA_stage_2b.py"
python3 kcl_NN_HEA_stage_2b.py

echo ">> Phase 2b (NN-HEA) complete."
