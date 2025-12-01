#!/bin/bash
#
# HANDS Quick Start Script
# Launches the HANDS application with proper environment
#

echo "=================================="
echo "HANDS Quick Start"
echo "=================================="
echo ""


# Resolve the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is the parent of the app directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

echo "Working directory: $(pwd)"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found in $(pwd)!"
    echo "   Please create one with: python3 -m venv .venv"
    exit 1
fi

# Parse arguments and support cleaning flags anywhere: c, -c, --clean
# We preserve the order of other args and forward them to hands_app.py
CLEAN=false
NEWARGS=()
for a in "$@"; do
    case "$a" in
        c|-c|--clean)
            CLEAN=true
            ;;
        *)
            NEWARGS+=("$a")
            ;;
    esac
done

if [ "$CLEAN" = true ]; then
    echo "Cleaning Python caches (__pycache__ and .pyc files)..."
    find . -type d -name "__pycache__" -print -exec rm -rf {} +
    find . -type f -name "*.pyc" -print -delete
fi

# Locate venv python executable in a cross-platform way
if [ -x ".venv/bin/python3" ]; then
    PYEXEC=".venv/bin/python3"
elif [ -x ".venv/bin/python" ]; then
    PYEXEC=".venv/bin/python"
else
    echo "❌ Python executable not found in .venv/bin/"
    echo "   Ensure the virtualenv is created and contains a python binary"
    exit 1
fi

echo "Starting HANDS application..."
echo ""

# Run as a module from the root directory
"$PYEXEC" -m source_code.app.hands_app "${NEWARGS[@]}"
