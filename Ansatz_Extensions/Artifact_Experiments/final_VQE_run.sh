#!/bin/bash
# Final VQE orchestrator (lives under Artifact_Experiments/).
# Resolves repo root as the parent of this directory, then runs src/final_VQE_run.py.
#
# From repo root:
#   ./Artifact_Experiments/final_VQE_run.sh
#   ./Artifact_Experiments/final_VQE_run.sh --mode baseline --ham-prefix 24qubits_05
#   ./Artifact_Experiments/final_VQE_run.sh --mode optimised --impl HEA
# From this directory:
#   ./final_VQE_run.sh [--mode all|baseline|optimised] [--ham-prefix ...] [--impl HEA|NN_HEA]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
OUTPUT_DIR="$REPO_ROOT/data/Final_VQE_run"

mkdir -p "$OUTPUT_DIR"

if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/venv/bin/activate"
fi

cd "$SRC_DIR"
python3 final_VQE_run.py "$@"
