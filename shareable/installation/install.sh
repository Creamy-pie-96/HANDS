#!/usr/bin/env bash
set -euo pipefail

# INSTALL.SH
# Creates a Python virtual environment named .venv at the project root,
# installs project dependencies (if `requirements.txt` is present),
# and prints currently installed packages. Use --snapshot to write
# a requirements.txt from the current venv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Requirements file lives next to this script (SCRIPT_DIR), but the venv/project
# root remains `PROJECT_ROOT`. Use REQ_FILE when installing or snapshotting.
REQ_FILE="$SCRIPT_DIR/requirements.txt"

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

if [[ -f "$REQ_FILE" ]]; then
  echo "Installing requirements from $REQ_FILE..."
  # Use the activated venv's python executable to install packages to the venv
  python -m pip install -r "$REQ_FILE" || {
    echo "ERROR: Failed to install requirements from $REQ_FILE" >&2
    exit 1
  }
else
  echo "No requirements.txt found at $REQ_FILE."
  echo "If you want to capture the packages currently installed in this venv, rerun this script with --snapshot after installing them manually."
fi

echo
echo "Installed packages (pip list):"
pip list --format=columns

if [[ "$SNAPSHOT" -eq 1 ]]; then
  echo
  echo "Saving current packages to $REQ_FILE (overwrite)..."
  pip freeze > "$REQ_FILE"
  echo "Saved: $REQ_FILE"
fi

echo
echo "Done. Activate the venv with: source .venv/bin/activate"

# --- Automated verification step for encrypted clone ---
# If workspace exists, attempt to run its clone script
# and verify the cloned output using the precomputed hashes. This
# helps ensure the installation/obfuscation artifacts are intact.
Working_dir="$PROJECT_ROOT"
if [[ -d "$Working_dir" ]]; then
  echo
  echo "workspace found at: $Working_dir"

  # Prefer a clone script bundled inside the encrypted output, fall back to project scripts
  if [[ -x "$Working_dir/scripts/clone.sh" ]]; then
    CLONE_SCRIPT="$Working_dir/scripts/clone.sh"
  elif [[ -x "$PROJECT_ROOT/scripts/clone.sh" ]]; then
    CLONE_SCRIPT="$PROJECT_ROOT/scripts/clone.sh"
  else
    CLONE_SCRIPT=""
  fi

  if [[ -z "$CLONE_SCRIPT" ]]; then
    echo "No clone script found (checked $Working_dir/scripts/clone.sh and $PROJECT_ROOT/scripts/clone.sh). Skipping encrypted verification."
  else
    CLONE_TARGET="$Working_dir"
    echo "Running clone script: $CLONE_SCRIPT -d $CLONE_TARGET"
    
    "$CLONE_SCRIPT" -d "$CLONE_TARGET"

    # Determine which verifier and hash-file to use (prefer the one bundled in encrypted)
    if [[ -f "$Working_dir/scripts/verify_clone.py" ]] && [[ -f "$Working_dir/scripts/clone_hashes.txt" ]]; then
      VERIFY_PY="$Working_dir/scripts/verify_clone.py"
      HASH_FILE="$Working_dir/scripts/clone_hashes.txt"
      PYTHONPATH_ROOT="$Working_dir"
    else
      VERIFY_PY="$PROJECT_ROOT/scripts/verify_clone.py"
      HASH_FILE="$PROJECT_ROOT/scripts/clone_hashes.txt"
      PYTHONPATH_ROOT="$PROJECT_ROOT"
    fi

    echo "Running verification using: $VERIFY_PY"
    echo "Hash file: $HASH_FILE"

    # Ensure PYTHONPATH includes the obfuscated root so pyarmor runtime package is importable
    PYTHONPATH="$PYTHONPATH_ROOT" python3 "$VERIFY_PY" --use-hashes "$CLONE_TARGET" --hash-file "$HASH_FILE"
    VERIFY_RC=$?

    if [[ $VERIFY_RC -eq 0 ]]; then
      echo
      echo "Installation completed: encrypted clone verified successfully."
    else
      echo
      echo "Installation failed: encrypted clone verification failed (exit code: $VERIFY_RC)."
      exit $VERIFY_RC
    fi
  fi
fi
