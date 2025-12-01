# Workspace Cloning System

This directory contains tools to clone the entire HANDS workspace structure and verify the clone.

## Files

- **`creator_of_clone.sh`** (root directory) - Scans the workspace and generates `clone.sh`
- **`clone.sh`** - Generated script that recreates the workspace structure
- **`verify_clone.py`** - Python script to verify cloned workspace integrity using SHA256 hashing

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

```bash
python3 scripts/verify_clone.py <original_dir> <cloned_dir>
```

Example:

```bash
python3 scripts/verify_clone.py . test_clone
```

The verification script will:

- Calculate SHA256 hashes for all files
- Compare file lists and content
- Report any missing, extra, or different files
- Exit with code 0 if perfect match, 1 if any differences

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

## Technical Details

- **Encoding**: Files are base64-encoded in the clone script to handle any special characters or binary content safely
- **Hashing**: Verification uses SHA256 for cryptographic-level integrity checking
- **Structure**: The clone script is self-contained - no external dependencies needed (except `base64` command)

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
