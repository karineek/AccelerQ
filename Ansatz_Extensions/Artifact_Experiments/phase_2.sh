#!/bin/bash
set -e
echo ">> Phase 2 starting..."

cd ../src
mkdir -p ../data/HEA ../data/UCCSD

echo ">> Running kcl_HEA_stage_2.py"
python3 kcl_HEA_stage_2.py

echo ">> Running kcl_UCCSD_stage_2.py"
python3 kcl_UCCSD_stage_2.py

echo ">> Phase 2 complete."
