#!/usr/bin/env bash
set -euo pipefail

# INSTALL.SH
# Creates a Python virtual environment named .venv at the project root,
# installs project dependencies (if `requirements.txt` is present),
# and prints currently installed packages. Use --snapshot to write
# a requirements.txt from the current venv.

# Check if python3 is installed
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python 3 is not installed or not in PATH."
  echo "Please install Python 3 to continue."
  exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python installation script
echo "Delegating to install.py..."
python3 "$SCRIPT_DIR/install.py" "$@"