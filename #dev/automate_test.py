#!/usr/bin/env python3
"""
automate_test.py

Automates the process of:
- Obfuscating the workspace using obfuscate_workspace.py
- Copying the obfuscated directory to a test location
- Running installation and startup scripts with timeouts
- Cleaning up if desired

Usage:
  python3 automate_test.py [shareable_dir] [test_dir]
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# Default directory names
shareable_dir = "shareable"
test_dir = "../test"

if len(sys.argv) > 1:
    # If user passed only one argument and it looks like a path (../ or / or contains /),
    # assume they meant the test directory. Otherwise treat first arg as shareable_dir.
    if len(sys.argv) == 2 and (sys.argv[1].startswith(".") or sys.argv[1].startswith("/") or '/' in sys.argv[1]):
        test_dir = sys.argv[1]
    else:
        shareable_dir = sys.argv[1]
if len(sys.argv) > 2:
    test_dir = sys.argv[2]

workspace_root = Path(__file__).resolve().parent.parent
shareable_path = workspace_root / shareable_dir
test_path = (workspace_root / test_dir).resolve()


# 1. Obfuscate workspace
print(f"Obfuscating workspace to: {shareable_path}")
print(f"Resolved test path: {test_path}")
print(f"sys.executable: {sys.executable}")
print(f"PATH: {os.environ.get('PATH')}")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
subprocess.run([
    sys.executable,
    str(workspace_root / "scripts" / "obfuscate_workspace.py"),
    str(workspace_root),
    "--output",
    str(shareable_path),
], check=True, cwd=str(workspace_root))

# 2. Copy to test directory
print(f"Copying {shareable_path} to {test_path}")
test_path.mkdir(parents=True, exist_ok=True)
shutil.rmtree(test_path / shareable_dir, ignore_errors=True)
shutil.copytree(shareable_path, test_path / shareable_dir)


# 3. Run Python install script
os.chdir(test_path / shareable_dir)
print(f"Running install.py in {os.getcwd()}")
# Run the installation script using an absolute path and ensure cwd is the shareable dir
install_script = test_path / shareable_dir / "installation" / "install.py"
if not install_script.exists():
    print(f"Error: install script not found at {install_script}")
    sys.exit(1)
# Run the installer from the copied test shareable directory (so it acts on the test copy)
subprocess.run([sys.executable, str(install_script)], check=True, cwd=str(test_path / shareable_dir))

# 4. Run config and start scripts with timeouts
os.chdir("app")
print(f"Changed to app directory: {os.getcwd()}")

for script in ["run_config.sh", "start_hands.sh"]:
    subprocess.run(["chmod", "+x", script], check=True)

print("Running run_config.sh with 30s timeout...")
ret_config = subprocess.run(["timeout", "30", "./run_config.sh"]).returncode
print("Running start_hands.sh with 40s timeout...")
ret_start = subprocess.run(["timeout", "40", "./start_hands.sh"]).returncode

print(f"RET_CONFIG: {ret_config}")
print(f"RET_START: {ret_start}")

if ret_start == 124:
    ret_start = 0
if ret_config == 124:
    ret_config = 0

if ret_config != 0 or ret_start != 0:
    print("One or more tasks failed.")
    sys.exit(1)

# 5. Optionally delete test directory
ans = input("Do you want to delete the dir? Y/N ").strip().lower()
if ans == "y":
    os.chdir(workspace_root)
    shutil.rmtree(test_path)
    print(f"Deleted {test_path}")
else:
    print(f"Test directory retained at {test_path}")
