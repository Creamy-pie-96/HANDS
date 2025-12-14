#!/usr/bin/env python3
"""
HANDS Cross-Platform Launcher
-----------------------------
Launches the HANDS application, handling environment setup, argument forwarding,
and control commands (pause/exit) across Linux, Windows, and macOS.

Usage:
    python app/start_hands.py [--clean] [control-flags] [-- <forwarded>]

Control Flags (runs app_control.py first):
    --pause [true|false]
    --exit [true|false]
    --config <path>
    --status
"""

import sys
import os
import subprocess
import shutil
import time
from pathlib import Path

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def clean_pycache():
    """Remove __pycache__ directories and .pyc files."""
    print("🧹 Cleaning Python caches...")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
        for f in files:
            if f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
    
    # Clear screen if possible (cross-platform)
    os.system('cls' if os.name == 'nt' else 'clear')

def get_venv_python():
    """Find virtual environment python executable."""
    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.exists():
        return None
    
    # Windows
    if sys.platform == 'win32':
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python3"
        if not python_exe.exists():
            python_exe = venv_dir / "bin" / "python"
            
    if python_exe.exists():
        return str(python_exe)
    return None

def main():
    # Check if we are running from the venv
    # If not, and venv exists, switch to it
    venv_python = get_venv_python()
    if venv_python:
        # Check if we are physically running the venv binary OR if we are in the venv context
        # Best check: is sys.prefix the venv dir?
        # venv_python is .../.venv/bin/python3
        # venv dir is .../.venv
        venv_root = Path(venv_python).parent.parent
        
        # Normalize paths for comparison
        try:
            current_prefix = Path(sys.prefix).resolve()
            target_prefix = venv_root.resolve()
            
            if current_prefix != target_prefix:
                print(f"🔄 Switching to venv: {venv_root}")
                # Re-exec using the venv executable
                os.execv(venv_python, [venv_python] + sys.argv)
        except Exception as e:
            print(f"⚠ Failed to switch to venv: {e}")

    # Parse arguments manually to separate launcher args from app args

    args = sys.argv[1:]
    
    clean_requested = False
    control_args = []
    forward_args = []
    
    # Simple manual parsing to preserving order for forwarding
    # and identifying control flags
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in ('--clean', '-c'):
            clean_requested = True
            i += 1
            continue
            
        # Check for control flags: --pause, --exit, --status, --config
        # We need to capture these for app_control.py if present
        if arg in ('--pause', '-p', '--exit', '-e', '--config', '-c'):
            # These take a value (usually), depending on how they are passed
            # If passed as --flag value, we take next. 
            # If passed as --flag=value, we take current.
            # CAUTION: --status is a flag, no value.
            
            control_args.append(arg)
            
            # If it's a flag that takes an argument, grab the next one if it's not a flag
            # (app_control logic is slightly loose, but usually --pause true or --pause=true)
            if '=' not in arg:
                if i + 1 < len(args) and not args[i+1].startswith('-'):
                    control_args.append(args[i+1])
                    # Also forward the value? Yes, usually.
                    forward_args.append(arg)
                    forward_args.append(args[i+1])
                    i += 2
                    continue
            
            forward_args.append(arg)
            i += 1
            continue
            
        elif arg in ('--status', '-s'):
            control_args.append(arg)
            forward_args.append(arg) # Forwarding status? Maybe not needed for app but harmless
            i += 1
            continue
            
        else:
            forward_args.append(arg)
            i += 1

    # Cleanup if requested
    if clean_requested:
        clean_pycache()

    # Run app_control if needed
    # We detect if specific control flags are present in what we extracted
    should_run_control = False
    exit_after_control = False
    
    for arg in control_args:
        if any(flag in arg for flag in ['--pause', '--exit', '--status']):
            should_run_control = True
        if 'exit' in arg:
            # Check if exit is true
            # This is a bit rough parsing, but matches shell script intent
            # If --exit is present, we assume intent to exit unless it says false explicitly later
            # But let's verify if 'true' or similar follows
            pass

    if should_run_control:
        print("⚙ Running system control...")
        cmd = [sys.executable, str(PROJECT_ROOT / "source_code" / "scripts" / "app_control.py")] + control_args
        
        # If config not specified in control args, add default
        has_config = any(c.startswith('--config') or c.startswith('-c') for c in control_args)
        if not has_config:
            default_config = PROJECT_ROOT / "source_code" / "config" / "config.json"
            cmd.extend(["--config", str(default_config)])
            
        subprocess.run(cmd)
        
        # Check if we should exit launcher (if --exit was passed with true/1/yes)
        # We search control_args for exit intent
        for idx, item in enumerate(control_args):
            if 'exit' in item:
                val = "true" # Default if flag only
                if '=' in item:
                    val = item.split('=')[1].lower()
                elif idx + 1 < len(control_args):
                    val = control_args[idx+1].lower()
                
                if val in ('true', '1', 'yes'):
                    print("🛑 Exit requested, stopping launch.")
                    time.sleep(1) # Give a moment to read output
                    sys.exit(0)

        # If --status was the only thing, we might want to exit? 
        # Shell script behavior: if control-only args AND no start flag, exit.
        # We'll mimic this simplicity: if no forward_args other than keys, exit?
        # Actually, let's just proceed to launch if not exiting.
        # The shell script logic for "Control-only invocation" is tricky to replicate perfectly 
        # without complex parsing, so we default to Launching unless Exit was explicit.

    # Launch main app
    print("🚀 Starting HANDS...")
    
    # We must run from PROJECT_ROOT for imports to work correctly (source_code.app...)
    # We explicitly set cwd
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "-m", "source_code.app.hands_app"] + forward_args
    
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped.")

if __name__ == "__main__":
    main()
