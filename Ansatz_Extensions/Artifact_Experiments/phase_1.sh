#!/bin/bash
set -e
prefixes=(
    #04qubits_00 #04qubits_01 #04qubits_05
    #06qubits_05 #06qubits_06 #06qubits_07
    #08qubits_05 #08qubits_06
    10qubits_05 #10qubits_06
)

cd ../src/
scripts=(
  #kcl_HEA_stage_1.py
  kcl_UCCSD_stage_1.py
)



DATA_HEA=../data/HEA
DATA_UCCSD=../data/UCCSD
mkdir -p "$DATA_HEA" "$DATA_UCCSD"

# Clean npy for prefixes in this run only (comment other impl block for HEA/UCCSD-only runs)
#for prefix in "${prefixes[@]}"; do
  #rm -f "$DATA_HEA/${prefix}.X.data.npy" "$DATA_HEA/${prefix}.Y.data.npy"
  #rm -f "$DATA_UCCSD/${prefix}A.X.data.npy" "$DATA_UCCSD/${prefix}A.Y.data.npy"
#done
#rm -f "$DATA_HEA/records_HEA.csv" "$DATA_HEA/hamiltonian_HEA.csv" "$DATA_HEA/optimal_HEA.csv"
#rm -f "$DATA_UCCSD/records_UCCSD.csv" "$DATA_UCCSD/hamiltonian_UCCSD.csv"

# for script in "${scripts[@]}"; do
#   sed -i 's/, 660,/, 5,/g' "$script"
# done

for prefix in "${prefixes[@]}"; do
  for script in "${scripts[@]}"; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running $script with prefix $prefix"
    # grep ", 5," "$script" | grep repeats || true
    token=$(grep "Change only this!" "$script" | cut -d'"' -f2)
    # Only the marker line — avoid rewriting other identical string literals.
    sed -i "/Change only this!/s|\"$token\"|\"$prefix\"|" "$script"
    python3 $script
  done
done

echo ">> Done"
