#!/usr/bin/env python3
"""
HANDS Config Editor Launcher (Python)

Equivalent to the original `run_config.sh` shell script.
Starts from the project root, ensures `.venv` exists, finds the venv
python and runs the config GUI module: `source_code.config.config_gui`.
"""

import os
import sys
import subprocess


def find_venv_python(venv_dir: str) -> str | None:
    """Return path to python executable inside venv `venv_dir/bin`, or None."""
    candidates = ["python3", "python"]
    for name in candidates:
        p = os.path.join(venv_dir, "bin", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def main() -> int:
    # Directory where this script resides (app/)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Project root is the parent of the app directory
    project_root = os.path.dirname(script_dir)

    try:
        os.chdir(project_root)
    except OSError as exc:
        print(f"❌ Failed to change directory to project root: {project_root}: {exc}", file=sys.stderr)
        return 1

    venv_dir = os.path.join(project_root, ".venv")
    if not os.path.isdir(venv_dir):
        print(f"❌ Virtual environment not found in {project_root}!", file=sys.stderr)
        return 1

    pyexec = find_venv_python(venv_dir)
    if not pyexec:
        print("❌ Python executable not found in .venv/bin/", file=sys.stderr)
        return 1

    print("Starting HANDS Config Editor...")
    # Run module using the venv python from the project root
    try:
        subprocess.check_call([pyexec, "-m", "source_code.config.config_gui"], cwd=project_root)
    except subprocess.CalledProcessError as e:
        return e.returncode
    except FileNotFoundError as e:
        print(f"❌ Failed to run python: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
