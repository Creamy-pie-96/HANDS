#!/bin/bash
#
# HANDS Config Editor Launcher
#


# Resolve the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is the parent of the app directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found in $(pwd)!"
    exit 1
fi

# Locate venv python executable
if [ -x ".venv/bin/python3" ]; then
    PYEXEC=".venv/bin/python3"
elif [ -x ".venv/bin/python" ]; then
    PYEXEC=".venv/bin/python"
else
    echo "❌ Python executable not found in .venv/bin/"
    exit 1
fi

echo "Starting HANDS Config Editor..."
"$PYEXEC" -m source_code.config.config_gui
