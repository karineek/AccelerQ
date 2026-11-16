#!/bin/bash
# Run all post-hoc AccelerQ analysis scripts.
#
# From the repository root:
#   ./Artifact_Experiments/run_all_analysis.sh
# From this directory:
#   ./run_all_analysis.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/venv/bin/activate"
fi

cd "$SRC_DIR"

echo "Running Objective 1 analysis..."
python3 kcl_obj1_analyse.py

echo "Running Objective 2 structural analysis..."
python3 kcl_obj2_analyse.py

echo "Running Objective 3 resource analysis..."
python3 kcl_obj3_analyse.py

echo "Running Objective 4 hyperparameter + SHAP analysis..."
python3 kcl_obj4_analyse.py

echo "All analyses completed."
