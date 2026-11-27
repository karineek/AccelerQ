#!/bin/bash
set +e

prefixes=(
    20qubits_05
)

for prefix in "${prefixes[@]}"; do
  echo ">>>> Final VQE baseline for $prefix"
  ./final_VQE_run.sh --mode baseline --ham-prefix "$prefix"

  echo ">>>> Final VQE optimised HEA for $prefix"
  ./final_VQE_run.sh --mode optimised --impl HEA --ham-prefix "$prefix"

  echo ">>>> Final VQE optimised NN_HEA for $prefix"
  ./final_VQE_run.sh --mode optimised --impl NN_HEA --ham-prefix "$prefix"
done

echo ">> Final VQE (retry) done."
