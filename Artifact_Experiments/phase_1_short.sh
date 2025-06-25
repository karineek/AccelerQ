#!/bin/bash
set -e
prefixes=(
    04qubits_05
    06qubits_05
    08qubits_05
)

cd ../src/
scripts=(
    kcl_QCELS_stage_1.py
    kcl_adapt_vqe_stage_1.py
)

for script in "${scripts[@]}"; do
  sed -i "s:res=miner(n_qubits, ham, repeats, 660:res=miner(n_qubits, ham, repeats, 5:g" $script
done

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
