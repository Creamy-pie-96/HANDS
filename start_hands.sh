#!/bin/bash
#
# HANDS Quick Start Script
# Launches the HANDS application with proper environment
#

echo "=================================="
echo "HANDS Quick Start"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please create one with: python3 -m venv .venv"
    exit 1
fi

# Activate venv and run
echo "Starting HANDS application..."
echo ""

.venv/bin/python hands_app.py "$@"
