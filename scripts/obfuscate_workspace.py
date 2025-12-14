#!/usr/bin/env python3
"""
obfuscate_workspace.py

Recursively clones a workspace directory:
- All .py files are obfuscated using PyArmor
- All other files are copied as-is
- Directory structure is preserved exactly

Usage:
  python3 obfuscate_workspace.py [SOURCE_DIR] [--output OUTPUT_DIR]
"""

import os
import sys
import shutil
import argparse
import subprocess
import glob
import tempfile
from pathlib import Path

# Patterns to ignore
IGNORE_PATTERNS = [
    ".venv", "venv", "env", ".env", ".git", "__pycache__",
    "*.pyc", "*.pyo", "*.pyd", ".DS_Store", "test_clone",
    "encrypted", "build", "dist"
]

def should_ignore(path: Path, source_dir: Path) -> bool:
    """Check if path should be ignored based on patterns."""
    name = path.name
    rel_path = path.relative_to(source_dir)
    
    # Check simple name match
    for pattern in IGNORE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
            
    # Check path components
    for part in rel_path.parts:
        for pattern in IGNORE_PATTERNS:
            if not pattern.startswith("*") and part == pattern:
                return True
                
    # Check if under virtualenv (pyvenv.cfg or bin/activate)
    # We walk up relative to source_dir
    # Actually, simpler to just check if any parent has venv markers
    # But for a recursive walk, if we pruned properly, we wouldn't be here.
    # We will implement pruning in the walker.
    return False

def is_virtualenv_root(path: Path) -> bool:
    return (path / "pyvenv.cfg").exists() or \
           (path / "bin" / "activate").exists() or \
           (path / "Scripts" / "activate").exists()

def main():
    parser = argparse.ArgumentParser(description="Obfuscate workspace using PyArmor")
    parser.add_argument("source_dir", nargs="?", default=None, help="Source directory to clone")
    parser.add_argument("-o", "--output", default="Hands_shareable", help="Output directory name")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    source_dir = Path(args.source_dir).resolve() if args.source_dir else script_dir.parent
    output_dir = Path(args.output).resolve()
    
    print("==========================================================================")
    print("WORKSPACE OBFUSCATION & CLONING (PYTHON)")
    print("==========================================================================")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}\n")
    
    # Check PyArmor
    try:
        subprocess.run([sys.executable, "-c", "import pyarmor"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("ERROR: PyArmor not installed. Install with: pip install pyarmor")
        sys.exit(1)
        
    if output_dir.exists():
        print(f"WARNING: Output directory {output_dir} already exists!")
        choice = input("Delete and recreate? [y/N] ").lower()
        if choice == 'y':
            shutil.rmtree(output_dir)
        else:
            print("Aborted.")
            sys.exit(1)
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_copy = []
    py_files = []
    
    print("Scanning source directory...")
    
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)

        # Prune ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(root_path / d, source_dir) and not is_virtualenv_root(root_path / d)]

        for file in files:
            file_path = root_path / file
            if should_ignore(file_path, source_dir):
                continue

            # Special case: do NOT obfuscate install.py, always copy as-is
            if file == "install.py":
                files_to_copy.append(file_path)
                continue

            # Do NOT obfuscate files under the top-level `app/` directory; copy them as-is
            try:
                rel = file_path.relative_to(source_dir)
            except Exception:
                rel = None
            if rel and len(rel.parts) > 0 and rel.parts[0] == "app":
                files_to_copy.append(file_path)
                continue

            if file.endswith(".py"):
                py_files.append(file_path)
            else:
                files_to_copy.append(file_path)

    print(f"Found {len(py_files)} Python files to obfuscate")
    print(f"Found {len(files_to_copy)} other files to copy")
    
    # Obfuscation
    if py_files:
        print("\nObfuscating Python files...")
        with tempfile.TemporaryDirectory() as stage_dir_str:
            stage_dir = Path(stage_dir_str)
            temp_obf = Path(tempfile.mkdtemp())
            
            try:
                # Stage files
                print(f"Staging to {stage_dir}...")
                for py_file in py_files:
                    rel_path = py_file.relative_to(source_dir)
                    dest = stage_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(py_file, dest)
                    
                    # Ensure init py in parents
                    curr = dest.parent
                    while curr != stage_dir and curr != curr.parent:
                        if not (curr / "__init__.py").exists():
                            (curr / "__init__.py").touch()
                        curr = curr.parent

                # Run PyArmor
                print("Running PyArmor...")
                subprocess.run(["pyarmor", "gen", "-r", "-O", str(temp_obf), str(stage_dir)], check=True)
                
                # Copy obfuscated files
                src_basename = stage_dir.name
                pyarmor_output = temp_obf / src_basename
                
                for py_file in py_files:
                    rel_path = py_file.relative_to(source_dir)
                    dest = output_dir / rel_path
                    obf_src = pyarmor_output / rel_path
                    
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    if obf_src.exists():
                        shutil.copy2(obf_src, dest)
                    else:
                        print(f"WARNING: Obfuscated version not found for {rel_path}, copying original")
                        shutil.copy2(py_file, dest)
                
                # Copy runtime
                runtime_dirs = list(temp_obf.glob("pyarmor_runtime_*"))
                if runtime_dirs:
                    rt = runtime_dirs[0]
                    print(f"Copying runtime: {rt.name}")
                    shutil.copytree(rt, output_dir / rt.name)
                    
                    # Add __init__.py to all subdirs
                    for root, dirs, _ in os.walk(output_dir):
                        for d in dirs:
                            if not d.startswith("pyarmor_runtime"):
                                init_file = Path(root) / d / "__init__.py"
                                if not init_file.exists():
                                    init_file.touch()

            finally:
                shutil.rmtree(temp_obf, ignore_errors=True)
    
    # Copy other files
    print("\nCopying non-Python files...")
    for f in files_to_copy:
        rel_path = f.relative_to(source_dir)
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        
    print("\nProcessing completed.")
    
    # Post processing (shareable logic)
    print("-- Preparing shareable package structure...")
    
    # Structure we want:
    # ROOT/
    #   installation/ (install.py, requirements*)
    #   scripts/ (clone.py, verify.py, hashes)
    #   source_code/ ...
    
    # Create installation directory
    install_dir = output_dir / "installation"
    install_dir.mkdir(exist_ok=True)
    
    # Move install files
    for f in ["install.sh", "install.py", "requirements.txt", 
              "requirements_linux.txt", "requirements_macos.txt", "requirements_windows.txt"]:
        src = output_dir / f
        if src.exists():
            shutil.move(src, install_dir / f)

    # Now run creator_of_clone on this restructured directory
    # This ensures clone.py and hashes cache the files in their FINAL location (installation/...)
    
    print(f"Running creator_of_clone in {output_dir}...")
    
    # Run creator_of_clone - detecting which one to use
    creator_py = output_dir / "creator_of_clone.py"
    creator_sh = output_dir / "creator_of_clone.sh"
    
    # The creator script needs to be present to run
    if creator_py.exists():
        subprocess.run([sys.executable, str(creator_py), "."], cwd=output_dir, check=True)
    elif creator_sh.exists() and shutil.which("bash"):
        subprocess.run(["bash", str(creator_sh), "."], cwd=output_dir, check=True)
    else:
        print("Warning: creator_of_clone script not found or runnable.")

    # Now that clone.py and hashes are generated (in scripts/), 
    # we need to CLEAN UP to leave only the shareable "launcher" files.
    # The user wants "only the installation/install.py and scripts dir clone.py" (and related files).
    
    print("Cleaning up: Retaining only installation/ and scripts/ directories...")
    
    # List of directories to keep
    KEEP_DIRS = ["installation", "scripts"]
    
    for item in output_dir.iterdir():
        if item.name not in KEEP_DIRS:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    # Clean scripts directory - keep only clone.py, clone_hashes.txt, and verify_clone.py
    print("Cleaning scripts directory...")
    scripts_dir = output_dir / "scripts"
    if scripts_dir.exists():
        KEEP_SCRIPTS = ["clone.py", "clone_hashes.txt", "verify_clone.py", "__init__.py"]
        for item in scripts_dir.iterdir():
            if item.name not in KEEP_SCRIPTS:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                    print(f"  Removed: scripts/{item.name}")

    print(f"Shareable package ready at: {output_dir}")
    print(f"Installer is at: {output_dir}/installation/install.py")
    return

if __name__ == "__main__":
    main()
