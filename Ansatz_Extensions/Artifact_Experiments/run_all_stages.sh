#!/bin/bash
# Full AccelerQ pipeline: Stage 1 → 2 → 2b → 3 → Final VQE.
#
# Runs the phase wrappers in order. Phase 3 already invokes final_VQE_run.sh
# for each configured Hamiltonian prefix.
#
# From the repository root:
#   ./Artifact_Experiments/run_all_stages.sh
# From this directory:
#   ./run_all_stages.sh
#
# Edit prefix lists / script selections inside phase_1.sh and phase_3.sh
# before launching a full run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/venv/bin/activate"
fi

cd "$SCRIPT_DIR"
chmod +x phase_1.sh phase_2.sh phase_2b.sh phase_3.sh final_VQE_run.sh

echo "========================================"
echo ">> Stage 1 (data mining)"
echo "========================================"
./phase_1.sh

echo "========================================"
echo ">> Stage 2 (XGBoost surrogates)"
echo "========================================"
./phase_2.sh

echo "========================================"
echo ">> Stage 2b (NN-HEA training)"
echo "========================================"
./phase_2b.sh

echo "========================================"
echo ">> Stage 3 + Final VQE"
echo "========================================"
./phase_3.sh

echo "========================================"
echo ">> Full pipeline complete."
echo "========================================"
