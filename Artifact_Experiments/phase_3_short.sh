#!/bin/bash
set -e
# Define new prefixes
prefixes=(
    20qubits_05
    24qubits_06 24qubits_07
)

cd ../src/
scripts=(
    kcl_QCELS_stage_3.py
    kcl_adapt_vqe_stage_3.py
)

for prefix in "${prefixes[@]}"; do
  for script in "${scripts[@]}"; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running $script with prefix $prefix"
    token=$(grep "Change only this!" "$script" | cut -d'"' -f2)
    sed -i "s|$token|$prefix|g" "$script"
    pwd
    python3 $script
  done
done

echo ">> Done"
