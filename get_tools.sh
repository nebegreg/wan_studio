#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="${1:-./tools}"
python3 "$(dirname "$0")/get_tools.py" "$TOOLS_DIR"
