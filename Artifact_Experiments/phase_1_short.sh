#!/bin/bash
set -e
prefixes=(
    02qubits_05
    04qubits_05
    14qubits_05 14qubits_06
)

cd ../src/
scripts=(
    kcl_QCELS_stage_1.py
    kcl_adapt_vqe_stage_1.py
)

for script in "${scripts[@]}"; do
  cp "$script" "$script.original"
done

for prefix in "${prefixes[@]}"; do
  for script in "${scripts[@]}"; do
    echo ">> Running $script with prefix $prefix"
    token=$(grep "Change only this!" "$script" | cut -d'"' -f2)
    sed -i "s|$token|$prefix|g" "$script"
    python3 $script
  done
done

echo ">> Done"
