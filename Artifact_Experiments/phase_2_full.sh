#!/bin/bash
set -e

echo ">> Phase 2 starting..."

# Step 1: move to src directory
cd ../src
# Step 2: create ../data if it doesn't exist
mkdir -p ../data

# Step 3: Download QCELS and ADAPT-QSCI datasets
echo ">> Downloading QCELS-data.tar.xz"
wget -O QCELS-data.tar.xz "https://zenodo.org/records/13328383/files/QCELS-data.tar.xz?download=1"
echo ">> Downloading ADAPT-QSCI-data.tar.xz"
wget -O ADAPT-QSCI-data.tar.xz "https://zenodo.org/records/13328383/files/ADAPT-QSCI-data.tar.xz?download=1"
echo ">> Extracting QCELS-data.tar.xz"
tar -xf QCELS-data.tar.xz
echo ">> Extracting ADAPT-QSCI-data.tar.xz"
tar -xf ADAPT-QSCI-data.tar.xz

# Step 4: Copy Copy all .X/.Y output files to ../data and run training
echo ">> Copying QCELS *.npy files to ../data"
rm ../data/*
cp QCELS-data/data/*.npy ../data
echo ">> Running kcl_QCELS_stage_2.py"
python3 kcl_QCELS_stage_2.py

# Step 5: Copy all .X/.Y output files to ../data and run training
echo ">> Copying  ADAPT-QSCI *.npy files to ../data"
rm ../data/*
cp ADAPT-QSCI-data/data/*.npy ../data
echo ">> Running kcl_adapt_vqe_stage_2.py"
python3 kcl_adapt_vqe_stage_2.py

echo ">> Phase 2 Full complete."
