#!/usr/bin/env python3
"""
creator_of_clone.py

Recursively walks through the workspace and creates a clone.py script that
can recreate the entire workspace structure and files.

Usage:
  python3 creator_of_clone.py [SCAN_DIR]
"""

import os
import sys
import argparse
import base64
from pathlib import Path
import hashlib
import datetime

# Files and directories to ignore
IGNORE_PATTERNS = [
    ".venv", ".git", ".gitignore",
    "__pycache__",
    "*.pyc", "*.pyo", "*.pyd", ".DS_Store", "creator_of_clone.sh", 
    "creator_of_clone.py",
    "test_clone", "stdout.log", "stderr.log"
]

def should_ignore(path: Path, scan_dir: Path) -> bool:
    name = path.name
    rel_path = path.relative_to(scan_dir)
    rel_str = str(rel_path).replace("\\", "/") # Normalize for pattern matching
    
    # Ignore install.py, install.sh, and requirements*.txt everywhere
    if name == "install.py" or name == "install.sh":
        return True
    if name.startswith("requirements") and name.endswith(".txt"):
        return True

    if "#dev" in rel_path.parts:
        return True

    for pattern in IGNORE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern == name:
            return True
        elif pattern == rel_str:
            return True
            
    return False

def generate_clone_script(scan_dir: Path, output_script: Path):
    
    with open(output_script, 'w', encoding='utf-8') as script:
        script.write('''#!/usr/bin/env python3
import os
import sys
import base64
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Recreate workspace")
    parser.add_argument("-d", "--directory", default=".", help="Target directory")
    args = parser.parse_args()
    
    target_dir = Path(args.directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cloning to: {target_dir}")
    
    def create_file(rel_path_str, encoded_content):
        dest = target_dir / rel_path_str
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating: {rel_path_str}")
        try:
            content = base64.b64decode(encoded_content)
            with open(dest, 'wb') as f:
                f.write(content)
        except Exception as e:
            print(f"Error creating {rel_path_str}: {e}")

''')
        
        print("Scanning workspace...")
        file_count = 0
        
        # We need two passes or store in memory? Bash script did one pass append.
        # We can append calls to create_file.
        
        # Logic to skip scripts/* except verify_clone.py and __init__.py matches bash script
        
        files_to_process = []
        for root, dirs, files in os.walk(scan_dir):
            for file in files:
                fpath = Path(root) / file
                if should_ignore(fpath, scan_dir):
                    continue
                
                rel_path = fpath.relative_to(scan_dir)
                rel_str = str(rel_path).replace("\\", "/")
                
                if rel_str.startswith("scripts/"):
                    # For clone generation, we still exclude the clone script itself to verify recursion issues?
                    # But the user WANTS it cloned? 
                    # Actually, if we are generating clone.py, we can't include it in itself.
                    # So we MUST exclude clone.py from the generated script content.
                    if rel_str in ["scripts/clone.py", "scripts/clone.sh", "scripts/clone_hashes.txt"]:
                        continue
                
                files_to_process.append(fpath)
        
        # Sort for deterministic output
        files_to_process.sort()
        
        for fpath in files_to_process:
            rel_path = fpath.relative_to(scan_dir)
            rel_str = str(rel_path).replace("\\", "/")
            
            try:
                with open(fpath, 'rb') as f:
                    content = f.read()
                    encoded = base64.b64encode(content).decode('ascii')
                    
                script.write(f'    create_file("{rel_str}", "{encoded}")\n')
                file_count += 1
            except Exception as e:
                print(f"Skipping {rel_str}: {e}")

        script.write('''
    print(f"Cloning completed. {''' + str(file_count) + '''} files created.")
    
    # Make scripts executable on Unix
    if os.name != 'nt':
        for root, dirs, files in os.walk(target_dir / "scripts"):
            for f in files:
                if f.endswith(".sh") or f.endswith(".py"):
                    (Path(root)/f).chmod(0o755)

if __name__ == "__main__":
    main()
''')

    # Make output executable
    if os.name != 'nt':
        output_script.chmod(0o755)
        
    print(f"Clone script generated: {output_script} ({file_count} files)")


def generate_hashes(scan_dir: Path, hash_file: Path):
    print("Generating hashes...")
    
    hashes = []
    
    for root, dirs, files in os.walk(scan_dir):
        for file in files:
            fpath = Path(root) / file
            if should_ignore(fpath, scan_dir):
                continue
            
            rel_path = fpath.relative_to(scan_dir)
            rel_str = str(rel_path).replace("\\", "/")
            
            if rel_str.startswith("scripts/"):
                # We want to hash clone.py if it exists, but not hash file itself
                if rel_str == "scripts/clone_hashes.txt":
                    continue
                # All other scripts (verify, clone.py, etc) should be hashed

                    
            try:
                sha = hashlib.sha256()
                with open(fpath, 'rb') as f:
                    while chunk := f.read(8192):
                        sha.update(chunk)
                hashes.append((sha.hexdigest(), rel_str))
            except Exception:
                pass
                
    hashes.sort(key=lambda x: x[1])
    
    with open(hash_file, 'w', encoding='utf-8') as f:
        f.write("# SHA256 hashes of original files\n")
        f.write(f"# Generated on {datetime.datetime.now()}\n")
        f.write("# Format: <hash> <relative_path>\n\n")
        for h, p in hashes:
            f.write(f"{h}  {p}\n")
            
    print(f"Hashes generated: {hash_file}")


def main():
    parser = argparse.ArgumentParser(description="Create clone script")
    parser.add_argument("scan_dir", nargs="?", default=None, help="Directory to scan")
    parser.add_argument("--scan-dir", dest="scan_dir_opt", help="Directory to scan (flag version)")
    
    args = parser.parse_args()
    scan_dir_path = args.scan_dir or args.scan_dir_opt
    
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    
    if scan_dir_path:
        scan_dir = Path(scan_dir_path).resolve()
    else:
        scan_dir = script_dir
        
    print(f"Scanning root: {scan_dir}")
    
    output_scripts_dir = script_dir / "scripts"
    output_scripts_dir.mkdir(exist_ok=True)
    
    output_script = output_scripts_dir / "clone.py"
    hash_file = output_scripts_dir / "clone_hashes.txt"
    
    generate_clone_script(scan_dir, output_script)
    generate_hashes(scan_dir, hash_file)
    
if __name__ == "__main__":
    main()
