#!/usr/bin/env bash
set -euo pipefail

# INSTALL.SH
# Creates a Python virtual environment named .venv at the project root,
# installs project dependencies (if `requirements.txt` is present),
# and prints currently installed packages. Use --snapshot to write
# a requirements.txt from the current venv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

usage() {
  cat <<EOF
Usage: $0 [--snapshot]

  --snapshot   After creating/activating the venv, write current installed
               packages to ./requirements.txt (overwrites if exists).

This script will:
  1) ensure a Python venv exists at .venv
  2) activate it
  3) upgrade pip/setuptools/wheel
  4) install from requirements.txt (if present)
  5) print installed packages (useful to verify)
EOF
}

SNAPSHOT=0
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if [[ ${1:-} == "--snapshot" ]]; then
  SNAPSHOT=1
fi

PY=python3

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY not found. Please install Python 3 and retry." >&2
  exit 2
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtualenv at ./.venv ..."
  "$PY" -m venv .venv
else
  echo "Virtualenv .venv already exists — skipping creation." 
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel >/dev/null

if [[ -f "requirements.txt" ]]; then
  echo "Installing requirements from requirements.txt..."
  pip install -r requirements.txt
else
  echo "No requirements.txt found in project root."
  echo "If you want to capture the packages currently installed in this venv, rerun this script with --snapshot after installing them manually."
fi

echo
echo "Installed packages (pip list):"
pip list --format=columns

if [[ "$SNAPSHOT" -eq 1 ]]; then
  echo
  echo "Saving current packages to requirements.txt (overwrite)..."
  pip freeze > requirements.txt
  echo "Saved: requirements.txt"
fi

echo
echo "Done. Activate the venv with: source .venv/bin/activate"
