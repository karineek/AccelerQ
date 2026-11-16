#!/bin/bash
set -e

prefixes=(
    20qubits_05 20qubits_00
)

cd ../src/

scripts=(
    kcl_HEA_stage_3.py
    kcl_NN_HEA_stage_3.py
    kcl_UCCSD_stage_3.py
)

for prefix in "${prefixes[@]}"; do
  for script in "${scripts[@]}"; do
    echo ">>>> Running $script with prefix $prefix"
    token=$(grep "Change only this!" "$script" | cut -d'"' -f2)
    sed -i "/Change only this!/s|\"$token\"|\"$prefix\"|" "$script"
    pwd
    python3 "$script"
  done
done

echo ">> Stage 3 done."

cd ../Artifact_Experiments/
for prefix in "${prefixes[@]}"; do
  echo ">>>> Final VQE for $prefix"
  ./final_VQE_run.sh --mode all --ham-prefix "$prefix"
done

echo ">> Final VQE done."
