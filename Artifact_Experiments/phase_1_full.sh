#!/bin/bash
set -e
prefixes=(
    02qubits_05
    04qubits_00 04qubits_01 04qubits_02 04qubits_03 04qubits_04 04qubits_05
    06qubits_05 06qubits_06 06qubits_07
    07qubits_05 07qubits_06
    08qubits_05 08qubits_06
    10qubits_05 10qubits_06
    12qubits_00 12qubits_01 12qubits_02 12qubits_03 12qubits_04 12qubits_05
    14qubits_05 14qubits_06
    16qubits_05 16qubits_06
)

cd ../src/
scripts=(
    kcl_QCELS_stage_1.py
    kcl_adapt_vqe_stage_1.py
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
