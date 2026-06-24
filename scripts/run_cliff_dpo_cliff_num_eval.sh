#!/usr/bin/env bash
# Backward-compatible alias for scripts/run_dpo_cliff_num_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_dpo_cliff_num_eval.sh" "$@"
