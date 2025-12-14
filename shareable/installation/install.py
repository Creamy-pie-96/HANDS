#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# Ensure working directory is the parent of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)

# ANSI colors for better output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_blue(msg):
        print(f"{Colors.BLUE}{msg}{Colors.ENDC}")

    @staticmethod
    def print_green(msg):
        print(f"{Colors.GREEN}{msg}{Colors.ENDC}")

    @staticmethod
    def print_fail(msg):
        print(f"{Colors.FAIL}{msg}{Colors.ENDC}")

    @staticmethod
    def print_warning(msg):
        print(f"{Colors.WARNING}{msg}{Colors.ENDC}")

# Disable colors on Windows if not supported or use colorama (not requiring it reduces dep)
if platform.system() == "Windows":
    # Simple workaround: disable codes
    class Colors:
        HEADER = ''
        BLUE = ''
        CYAN = ''
        GREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''
        
        @staticmethod
        def print_blue(msg): print(msg)
        @staticmethod
        def print_green(msg): print(msg)
        @staticmethod
        def print_fail(msg): print(msg)
        @staticmethod
        def print_warning(msg): print(msg)

def get_script_dir():
    return Path(__file__).resolve().parent

def get_project_root():
    return get_script_dir().parent

def run_command(command, cwd=None, shell=False, check=True):
    try:
        subprocess.run(command, cwd=cwd, shell=shell, check=check)
        return True
    except subprocess.CalledProcessError as e:
        Colors.print_fail(f"Error executing command: {e}")
        return False

def check_os_requirements():
    os_name = platform.system().lower()
    script_dir = get_script_dir()
    
    print(f"Detected OS: {os_name}")
    
    req_file = None
    if os_name == 'linux':
        req_file = script_dir / 'requirements_linux.txt'
    elif os_name == 'darwin':
        req_file = script_dir / 'requirements_macos.txt'
    elif os_name == 'windows':
        req_file = script_dir / 'requirements_windows.txt'
    
    if req_file and req_file.exists():
        Colors.print_blue(f"--- {os_name.upper()} SPECIFIC INSTRUCTIONS ---")
        content = req_file.read_text()
        print(content)
        Colors.print_blue("-------------------------------------")
        
        # Attempt to auto-install dependencies
        Colors.print_blue(f"Attempting to auto-install {os_name} system dependencies...")
        
        install_cmds = []
        if os_name == 'linux':
            # Extract apt or apt-get commands
            for line in content.splitlines():
                # If the line contains '&&', split it into separate commands
                if '&&' in line:
                    parts = [part.strip() for part in line.split('&&')]
                    for part in parts:
                        if part:
                            install_cmds.append(part.split())
                # Match lines with 'sudo apt install' or 'sudo apt-get install' or similar
                elif ("sudo apt install" in line or "sudo apt-get install" in line or 
                      "apt install" in line or "apt-get install" in line):
                    install_cmds.append(line.strip().split())
        elif os_name == 'darwin':
            # Extract brew commands, handle chained commands with '&&'
            for line in content.splitlines():
                if '&&' in line:
                    parts = [part.strip() for part in line.split('&&')]
                    for part in parts:
                        if part:
                            install_cmds.append(part.split())
                elif "brew install" in line:
                    install_cmds.append(line.strip().split())

        elif os_name == 'windows':
            # Extract choco/winget/powershell install commands, handle chained commands with '&&'
            for line in content.splitlines():
                if '&&' in line:
                    parts = [part.strip() for part in line.split('&&')]
                    for part in parts:
                        if part:
                            install_cmds.append(part.split())
                elif ("choco install" in line or "winget install" in line or "powershell" in line):
                    install_cmds.append(line.strip().split())
                    
        if install_cmds:
            for cmd in install_cmds:
                Colors.print_green(f"Running: {' '.join(cmd)}")
                # On linux, sudo might require interaction.
                # We let it run; user can interact.
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError:
                    Colors.print_fail("Automatic installation failed. Please install manually.")
        else:
             print("No automatic installation commands found.")
             
        Colors.print_blue("-------------------------------------")
        if os_name != 'windows':
             print("Press Enter to continue (or Ctrl+C to abort)...")
             input()
    
def setup_venv():
    project_root = get_project_root()
    venv_dir = project_root / '.venv'
    
    if not venv_dir.exists():
        Colors.print_blue(f"Creating virtual environment at {venv_dir}...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        except subprocess.CalledProcessError:
            Colors.print_fail("Failed to create virtual environment. Please install python3-venv (Linux) or check your Python installation.")
            sys.exit(1)
    else:
        Colors.print_green("Virtual environment already exists.")

    # Determine pip path
    if platform.system() == "Windows":
        pip_cmd = venv_dir / 'Scripts' / 'pip'
        python_cmd = venv_dir / 'Scripts' / 'python'
    else:
        pip_cmd = venv_dir / 'bin' / 'pip'
        python_cmd = venv_dir / 'bin' / 'python'
        
    return str(pip_cmd), str(python_cmd)

def install_dependencies(pip_cmd):
    project_root = get_project_root()
    script_dir = get_script_dir()
    
    # Try to find requirements.txt in installation directory first (where this script is)
    # Then fall back to project root
    req_file = script_dir / 'requirements.txt'
    if not req_file.exists():
        req_file = project_root / 'requirements.txt'
    
    Colors.print_blue("Upgrading pip, setuptools, wheel...")
    run_command([pip_cmd, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    
    if req_file.exists():
        Colors.print_blue(f"Installing requirements from {req_file}...")
        if run_command([pip_cmd, 'install', '-r', str(req_file)]):
            Colors.print_green("Dependencies installed successfully.")
        else:
            Colors.print_fail("Failed to install dependencies.")
            sys.exit(1)
    else:
        Colors.print_warning("requirements.txt not found.")

def verify_encrypted_clone(python_cmd):
    """
    Automated verification step for encrypted clone.
    """
    project_root = get_project_root()
    
    # Logic from install.sh: check if we are in a workspace that needs verification
    # Note: This logic assumes we might be running INSIDE a deployed/cloned workspace
    # or verifying a workspace that exists at PROJECT_ROOT.
    
    working_dir = project_root
    
    # Check for clone script (Python preferred, then Bash)
    clone_script = None
    is_python_clone = False
    
    # Check for clone.py
    possible_py_scripts = [
        working_dir / 'scripts' / 'clone.py',
        project_root / 'scripts' / 'clone.py'
    ]
    
    for s in possible_py_scripts:
        if s.exists():
            clone_script = s
            is_python_clone = True
            break
            
    # If not found, check for clone.sh (legacy/bash)
    if not clone_script:
        possible_sh_scripts = [
            working_dir / 'scripts' / 'clone.sh',
            project_root / 'scripts' / 'clone.sh'
        ]
        
        for s in possible_sh_scripts:
            if s.exists() and (platform.system() != "Windows" or shutil.which("bash")):
                clone_script = s
                break
            
    # Execution logic
    if clone_script:
        clone_target = working_dir
        Colors.print_blue(f"Running clone script: {clone_script}")
        
        try:
            if is_python_clone:
                # Run python clone script
                subprocess.run([sys.executable, str(clone_script), '-d', str(clone_target)], check=True)
            else:
                # Run bash clone script
                # On Windows this might fail if no bash, but we checked logic
                cmd = [str(clone_script), '-d', str(clone_target)]
                if platform.system() == "Windows":
                    cmd = ["bash"] + cmd
                subprocess.run(cmd, check=True)
                
            # Verification logic follows...
            pass 
        except subprocess.CalledProcessError as e:
            Colors.print_fail(f"Clone script failed: {e}")
            sys.exit(1)
    else:
        Colors.print_blue("No clone script found. Skipping encrypted verification/extraction.")

    # Determine which verifier and hash-file to use (prefer the one bundled in encrypted)
    verify_py = None
    hash_file = None
    
    if (working_dir / 'scripts' / 'verify_clone.py').exists() and (working_dir / 'scripts' / 'clone_hashes.txt').exists():
        verify_py = working_dir / 'scripts' / 'verify_clone.py'
        hash_file = working_dir / 'scripts' / 'clone_hashes.txt'
        pythonpath_root = working_dir
    elif (project_root / 'scripts' / 'verify_clone.py').exists() and (project_root / 'scripts' / 'clone_hashes.txt').exists():
        verify_py = project_root / 'scripts' / 'verify_clone.py'
        hash_file = project_root / 'scripts' / 'clone_hashes.txt'
        pythonpath_root = project_root
        
    if verify_py and hash_file:
        Colors.print_blue(f"Running verification using: {verify_py}")
        
        # Determine clone target - in install.sh it clones to "$Working_dir" (which is self?)
        # Actually line 118: "$CLONE_SCRIPT" -d "$CLONE_TARGET"
        # Wait, if we are in the directory, why clone to itself?
        # Maybe it restores files?
        # Or maybe this is for testing?
        # "If workspace exists... run its clone script... verify cloned output"
        # It seems it runs clone script to populate/restore files then verifies?
        
        # On Windows/Python we might skip the bash execution part if it's .sh
        # But we can run the verification script which is Python.
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(pythonpath_root)
        
        try:
            # We need to run verify_clone.py with the same python interpreter
            subprocess.run(
                [python_cmd, str(verify_py), '--use-hashes', str(working_dir), '--hash-file', str(hash_file)],
                env=env,
                check=True
            )
            Colors.print_green("Installation completed: encrypted clone verified successfully.")
        except subprocess.CalledProcessError:
            Colors.print_fail("Installation failed: encrypted clone verification failed.")
            sys.exit(1)
    else:
        # Not a failure, just skipping
        pass

def main():
    Colors.print_blue("====================================================================")
    Colors.print_blue("HANDS Installation Script")
    Colors.print_blue("====================================================================")
    
    check_os_requirements()
    
    pip_cmd, python_cmd = setup_venv()
    
    install_dependencies(pip_cmd)
    
    verify_encrypted_clone(python_cmd)

    # Set permissions for scripts (Linux/Mac)
    if platform.system() != "Windows":
        project_root = get_project_root()
        scripts = [
            project_root / 'app' / 'start_hands.sh',
            project_root / 'app' / 'run_config.sh'
        ]
        for s in scripts:
            if s.exists():
                try:
                    os.chmod(s, 0o755)
                except Exception:
                    pass

    Colors.print_blue("====================================================================")
    Colors.print_blue("Installation Complete!")
    Colors.print_blue("To start the application:")
    Colors.print_blue("  source .venv/bin/activate  (Linux/Mac)")
    Colors.print_blue("  .venv\\Scripts\\activate     (Windows)")
    Colors.print_blue("Go to the app dir by running cd app")
    Colors.print_blue("Then run: python3 hands_app.py")
    Colors.print_blue("====================================================================")

if __name__ == "__main__":
    main()
