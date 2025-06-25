#!/bin/bash
set -e
# Define new prefixes
prefixes=(
    20qubits_00 20qubits_01 20qubits_02 20qubits_03 20qubits_04 20qubits_05
    24qubits_05 24qubits_06 24qubits_07 24qubits_08 24qubits_09
    28qubits_00 28qubits_01 28qubits_02 28qubits_03 28qubits_04
)

cd ../src/
scripts=(
    kcl_QCELS_stage_3.py
    kcl_adapt_vqe_stage_3.py
)

for prefix in "${prefixes[@]}"; do
  for script in "${scripts[@]}"; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running $script with prefix $prefix"
    grep "res=miner(n_qubits, ham, repeats" $script
    token=$(grep "Change only this!" "$script" | cut -d'"' -f2)
    sed -i "s|$token|$prefix|g" "$script"
    python3 $script
  done
done

echo ">> Done"
