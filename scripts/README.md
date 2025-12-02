# Workspace Cloning System

This directory contains tools to clone the entire HANDS workspace structure and verify the clone.

## Files

- **`creator_of_clone.sh`** (root directory) - Scans the workspace and generates `clone.sh`
- **`clone.sh`** - Generated script that recreates the workspace structure
- **`verify_clone.py`** - Python script to verify cloned workspace integrity using SHA256 hashing
- **`obfuscate_workspace.sh`** - Obfuscates all Python files and clones workspace with obfuscated code

## Usage

### 1. Generate the clone script

```bash
./creator_of_clone.sh
```

This scans all files in the workspace (excluding `.venv`, `.git`, etc.) and generates `scripts/clone.sh`.

### 2. Clone the workspace

```bash
# Clone to current directory
./scripts/clone.sh

# Clone to specific directory
./scripts/clone.sh -d /path/to/target/directory
```

The clone script will:

- Create all necessary directories
- Recreate all files with exact content (uses base64 encoding)
- Display progress as it creates files

### 3. Verify the clone

The `verify_clone.py` script offers two modes of verification:

**Mode 1: Compare two directories directly**

```bash
python3 scripts/verify_clone.py <original_dir> <cloned_dir>
```

Example:

```bash
python3 scripts/verify_clone.py . test_clone
```

**Mode 2: Verify against pre-computed hashes (recommended)**

When you run `creator_of_clone.sh`, it automatically generates `scripts/clone_hashes.txt` with SHA256 hashes of all files. You can use these hashes to quickly verify a clone:

```bash
python3 scripts/verify_clone.py --use-hashes <cloned_dir> [--hash-file <hash_file>]
```

Example:

```bash
# Use default hash file location (scripts/clone_hashes.txt)
python3 scripts/verify_clone.py --use-hashes test_clone

# Or specify a custom hash file
python3 scripts/verify_clone.py --use-hashes test_clone --hash-file scripts/clone_hashes.txt
```

The verification script will:

- Calculate SHA256 hashes for all files in the cloned directory
- Compare against original hashes or pre-computed hash file
- Report any missing, extra, or different files
- Exit with code 0 if perfect match, 1 if any differences

**Note:** `scripts/clone_hashes.txt` and `scripts/verify_clone.py` are not included in the hash verification list to avoid self-referential issues.

### 4. Obfuscate workspace (create encrypted clone)

Create a complete clone of the workspace with all Python files obfuscated using PyArmor:

```bash
# Obfuscate current workspace to ./encrypted directory
./scripts/obfuscate_workspace.sh . --output encrypted

# Obfuscate specific directory with custom output
./scripts/obfuscate_workspace.sh /path/to/source --output /path/to/encrypted_output
```

**What it does:**

- Scans the source directory for all Python files (respecting `.venv`, `.git`, and other ignore patterns)
- Uses staging to avoid scanning virtualenv internals
- Obfuscates all `.py` files using PyArmor (preserves functionality)
- Copies all non-Python files as-is (configs, docs, shell scripts, etc.)
- Maintains exact directory structure
- Includes PyArmor runtime package (`pyarmor_runtime_000000`)
- Generates `run_obfuscated.sh` wrapper script and `README_OBFUSCATED.txt`

**Running obfuscated code:**

From within the obfuscated directory:

```bash
cd encrypted  # or your output directory

# Use the provided wrapper script (sets PYTHONPATH automatically)
./run_obfuscated.sh source_code/app/hands_app.py [args...]

# Or set PYTHONPATH manually
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 source_code/app/hands_app.py [args...]
```

**Cloning and verifying an obfuscated workspace:**

```bash
cd encrypted

# Generate clone script for the obfuscated workspace
../creator_of_clone.sh .

# Clone it
./scripts/clone.sh -d test

# Verify the clone with pre-computed hashes
export PYTHONPATH=. && python3 scripts/verify_clone.py --use-hashes test scripts/clone_hashes.txt
```

**Requirements:**

- `pyarmor` must be installed: `pip install pyarmor`
- Python 3.x

**Notes:**

- The obfuscated workspace is fully functional
- All imports and dependencies are preserved
- PyArmor trial version may have limitations with very large batch operations
- The output includes a `README_OBFUSCATED.txt` with detailed usage instructions

## What Gets Cloned

The system clones all workspace files **except**:

- `.venv` (virtual environment)
- `.git` (git repository)
- `.gitignore`
- `install.sh`
- `requirements.txt`
- `__pycache__` directories
- `*.pyc`, `*.pyo`, `*.pyd` files
- Test/temporary files
- `creator_of_clone.sh`
- Generated `scripts/clone.sh`

**Special handling in hash verification:**

- `scripts/clone_hashes.txt` - created during clone generation, not hashed (would be self-referential)
- `scripts/verify_clone.py` - the verifier script itself, not hashed (obfuscated, may change)

## Technical Details

- **Encoding**: Files are base64-encoded in the clone script to handle any special characters or binary content safely
- **Hashing**: Verification uses SHA256 for cryptographic-level integrity checking
- **Structure**: The clone script is self-contained - no external dependencies needed (except `base64` command which is standard on Unix/Linux)
- **Virtualenv handling**: The `obfuscate_workspace.sh` script safely ignores `.venv`, `venv`, `env` directories by using a staging approach before running PyArmor
- **Import paths**: When running obfuscated code that uses PyArmor runtime, ensure the obfuscated root directory is on `PYTHONPATH` (use `run_obfuscated.sh` or `export PYTHONPATH="$(pwd):$PYTHONPATH"`)

## Example Output

```
==================================================
WORKSPACE CLONING IN PROGRESS
==================================================

Target directory: /home/user/test_clone

[1] Creating: app/run_config.sh
[2] Creating: app/start_hands.sh
...
[21] Creating: zoom_detector_improvements.md

==================================================
CLONING COMPLETED
==================================================

Directories created: 6
Files created: 21
```

## Verification Output

```
======================================================================
WORKSPACE CLONE VERIFICATION
======================================================================

✅ SUCCESS: All files match perfectly!

  Total files verified: 20

Statistics:
  Identical files:     20
  Different content:   0
  Missing in clone:    0
  Extra in clone:      0
```
