#!/bin/bash
set -e
echo ">> Phase 2 starting..."

# Step 1: move to src directory
cd ../src
# Step 2: create ../data if it doesn't exist
rm -rf ../data
mkdir ../data



# Step 3: Copy QCELS A.X/Y files to data
cp *A.X.data.npy *A.Y.data.npy ../data
ls -l ../data
# Step 4: Run QCELS Stage 2
echo ">> Running kcl_QCELS_stage_2.py"
python3 kcl_QCELS_stage_2.py



# Step 5: Copy all .X/.Y output files to ../data
cp *.X.data.npy *.Y.data.npy ../data
rm ../data/*A.X.data.npy ../data/*A.Y.data.npy
ls -l ../data
# Step 6: Run Adapt-VQE Stage 2
echo ">> Running kcl_adapt_vqe_stage_2.py"
python3 kcl_adapt_vqe_stage_2.py

echo ">> Phase 2 complete."
