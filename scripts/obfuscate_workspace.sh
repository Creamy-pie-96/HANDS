#!/usr/bin/env bash
set -uo pipefail

################################################################################
# obfuscate_workspace.sh
#
# Recursively clones a workspace directory:
# - All .py files are obfuscated using PyArmor
# - All other files are copied as-is
# - Directory structure is preserved exactly
#
# Usage:
#   ./obfuscate_workspace.sh [SOURCE_DIR] [--output OUTPUT_DIR]
#   ./obfuscate_workspace.sh --help
################################################################################

# Default values
SOURCE_DIR=""
OUTPUT_DIR="Hands_shareable"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Patterns to ignore (similar to creator_of_clone.sh)
IGNORE_PATTERNS=(
    ".venv"
    "venv"
    "env"
    ".env"
    ".git"
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".DS_Store"
    "test_clone"
    "encrypted"
    "build"
    "dist"
)

show_help() {
    cat <<EOF
Usage: $(basename "$0") [SOURCE_DIR] [OPTIONS]

Recursively clone a workspace with all Python files obfuscated.

Arguments:
  SOURCE_DIR              Source directory to clone (default: parent of script)

Options:
  -o, --output DIR        Output directory name (default: "encrypted")
  -h, --help             Show this help message

Examples:
  # Obfuscate current workspace to ./encrypted
  ./obfuscate_workspace.sh

  # Obfuscate specific directory to custom output
  ./obfuscate_workspace.sh /path/to/source --output /path/to/output

  # Obfuscate with custom output name in current dir
  ./obfuscate_workspace.sh . --output my_encrypted_workspace

EOF
}

should_ignore() {
    local path="$1"
    local basename=$(basename "$path")
    
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        # basename match (supports glob patterns)
        if [[ "$basename" == $pattern ]]; then
            return 0
        fi

        # path component match (directory anywhere in path)
        if [[ "$path" == *"/$pattern/"* ]] || [[ "$path" == *"/$pattern" ]]; then
            return 0
        fi
    done
    
    # Also detect virtualenv folders by looking for pyvenv.cfg or activation scripts
    # in any ancestor directory of the file. If found, treat as ignored.
    if is_under_virtualenv "$path"; then
        return 0
    fi

    return 1
}


is_under_virtualenv() {
    # Walk up from the file's directory to the SOURCE_DIR (or filesystem root)
    local filepath="$1"
    local dir
    if [[ -d "$filepath" ]]; then
        dir="$filepath"
    else
        dir=$(dirname "$filepath")
    fi

    # Stop at SOURCE_DIR if set and ancestor, otherwise stop at filesystem root
    while [[ -n "$dir" && "$dir" != "/" ]]; do
        if [[ -f "$dir/pyvenv.cfg" ]]; then
            return 0
        fi
        if [[ -f "$dir/bin/activate" ]] || [[ -f "$dir/Scripts/activate" ]]; then
            return 0
        fi
        # If we reached the configured SOURCE_DIR, stop searching
        if [[ -n "$SOURCE_DIR" && "$dir" == "$SOURCE_DIR" ]]; then
            break
        fi
        dir=$(dirname "$dir")
    done

    return 1
}

check_pyarmor() {
    if ! python3 -c "import pyarmor" >/dev/null 2>&1; then
        echo "ERROR: PyArmor not installed" >&2
        echo "Install it with: pip install pyarmor" >&2
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            show_help
            exit 2
            ;;
        *)
            if [[ -z "$SOURCE_DIR" ]]; then
                SOURCE_DIR="$1"
            else
                echo "Error: Multiple source directories specified" >&2
                show_help
                exit 2
            fi
            shift
            ;;
    esac
done

# Default source to parent of script directory if not specified
if [[ -z "$SOURCE_DIR" ]]; then
    SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Convert to absolute paths
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

echo "=========================================================================="
echo "WORKSPACE OBFUSCATION & CLONING"
echo "=========================================================================="
echo ""
echo "Source directory:  $SOURCE_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo ""

# Check prerequisites
check_pyarmor

# Create output directory
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "WARNING: Output directory already exists!"
    read -p "Delete and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OUTPUT_DIR"
    else
        echo "Aborted."
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

# Counters
PY_COUNT=0
OTHER_COUNT=0
DIR_COUNT=0

# Temp directory for PyArmor work
TEMP_OBF=$(mktemp -d)
trap "rm -rf $TEMP_OBF" EXIT

echo "Scanning source directory..."
echo ""

# Find all files
TEMP_LIST=$(mktemp)
find "$SOURCE_DIR" -type f | sort > "$TEMP_LIST"

# First pass: collect all Python files for obfuscation
PYTHON_FILES=()
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    
    # Skip ignored patterns
    if should_ignore "$file"; then
        continue
    fi
    
    # Check if it's a Python file
    if [[ "$file" == *.py ]]; then
        PYTHON_FILES+=("$file")
    fi
done < "$TEMP_LIST"

echo "Found ${#PYTHON_FILES[@]} Python files to obfuscate"
echo ""

# Obfuscate all Python files in one batch using PyArmor's recursive mode
if [[ ${#PYTHON_FILES[@]} -gt 0 ]]; then
    echo "Obfuscating Python files..."
    echo ""

    # Create a staging directory that contains only the files we want PyArmor
    # to process. This prevents PyArmor from scanning the full SOURCE_DIR
    # (for example, skipping virtualenv site-packages like .venv).
    STAGE_DIR=$(mktemp -d)
    trap "rm -rf $STAGE_DIR" EXIT

    echo "Creating staging directory for PyArmor: $STAGE_DIR"
    for py_file in "${PYTHON_FILES[@]}"; do
        rel_path="${py_file#$SOURCE_DIR/}"
        dest="$STAGE_DIR/$rel_path"
        dest_dir=$(dirname "$dest")
        if [[ ! -d "$dest_dir" ]]; then
            mkdir -p "$dest_dir"
        fi
        cp "$py_file" "$dest"
        # ensure package dirs contain __init__.py
        curdir="$dest_dir"
        while [[ "$curdir" != "$STAGE_DIR" && "$curdir" != "/" ]]; do
            if [[ ! -f "$curdir/__init__.py" ]]; then
                touch "$curdir/__init__.py"
            fi
            curdir=$(dirname "$curdir")
        done
    done

    # Use PyArmor to obfuscate the staging directory recursively
    echo "Running PyArmor on staging directory..."
    if pyarmor gen -r -O "$TEMP_OBF" "$STAGE_DIR" >/dev/null 2>&1; then
        echo "✓ PyArmor obfuscation completed"
        echo ""
        
        # PyArmor creates output in: TEMP_OBF/basename(STAGE_DIR)/...
        SOURCE_BASENAME=$(basename "$STAGE_DIR")
        PYARMOR_OUTPUT="$TEMP_OBF/$SOURCE_BASENAME"
        
        # Now copy obfuscated Python files to output, preserving structure
        for py_file in "${PYTHON_FILES[@]}"; do
            rel_path="${py_file#$SOURCE_DIR/}"
            output_file="$OUTPUT_DIR/$rel_path"
            output_dir=$(dirname "$output_file")
            obf_file="$PYARMOR_OUTPUT/$rel_path"
            
            # Create directory structure
            if [[ ! -d "$output_dir" ]]; then
                mkdir -p "$output_dir"
                ((DIR_COUNT++))
            fi
            
            ((PY_COUNT++))
            echo "[$PY_COUNT/${#PYTHON_FILES[@]}] Copying obfuscated: $rel_path"
            
            if [[ -f "$obf_file" ]]; then
                cp "$obf_file" "$output_file"
            else
                echo "  WARNING: Obfuscated version not found, copying original" >&2
                cp "$py_file" "$output_file"
            fi
        done
        
        # Copy PyArmor runtime directory
        runtime_dir=$(find "$TEMP_OBF" -maxdepth 1 -type d -name "pyarmor_runtime_*" | head -1)
        if [[ -n "$runtime_dir" ]]; then
            runtime_name=$(basename "$runtime_dir")
            cp -r "$runtime_dir" "$OUTPUT_DIR/$runtime_name"
            echo ""
            echo "✓ Copied PyArmor runtime: $runtime_name"
            
            # Add __init__.py to all subdirectories to ensure imports work
            find "$OUTPUT_DIR" -type d ! -name "pyarmor_runtime_*" -exec sh -c 'test ! -f "$1/__init__.py" && touch "$1/__init__.py"' _ {} \;
            echo "✓ Added __init__.py files to subdirectories"
            
            # Create a README with usage instructions
            cat > "$OUTPUT_DIR/README_OBFUSCATED.txt" << 'README_EOF'
OBFUSCATED WORKSPACE
====================

This directory contains an obfuscated copy of the original workspace.

IMPORTANT NOTES:
- All Python files have been obfuscated using PyArmor
- All other files (configs, docs, etc.) are copied as-is
- The pyarmor_runtime_* directory contains required runtime files

RUNNING PYTHON SCRIPTS:
Method 1 (Recommended - use the run_obfuscated.sh wrapper):
  ./run_obfuscated.sh script.py
  ./run_obfuscated.sh subdir/script.py

Method 2 (Manual - from root directory):
  python3 script.py
  PYTHONPATH="$(pwd)" python3 subdir/script.py

Method 3 (Set PYTHONPATH for your session):
  export PYTHONPATH="$(pwd):$PYTHONPATH"
  python3 subdir/script.py

STRUCTURE:
- Directory structure matches the original workspace exactly
- PyArmor runtime is located in the root directory
- All subdirectories have __init__.py files for package imports

README_EOF
            echo "✓ Created README_OBFUSCATED.txt"
            
            # Create a runner script
            cat > "$OUTPUT_DIR/run_obfuscated.sh" << 'RUNNER_EOF'
#!/usr/bin/env bash
# Wrapper script to run obfuscated Python scripts with correct PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script.py> [args...]"
    echo "Example: $0 source_code/app/hands_app.py"
    exit 1
fi

exec python3 "$@"
RUNNER_EOF
            chmod +x "$OUTPUT_DIR/run_obfuscated.sh"
            echo "✓ Created run_obfuscated.sh wrapper script"
        fi
    else
        echo "WARNING: PyArmor batch obfuscation failed, copying originals" >&2
        for py_file in "${PYTHON_FILES[@]}"; do
            rel_path="${py_file#$SOURCE_DIR/}"
            output_file="$OUTPUT_DIR/$rel_path"
            output_dir=$(dirname "$output_file")
            
            if [[ ! -d "$output_dir" ]]; then
                mkdir -p "$output_dir"
                ((DIR_COUNT++))
            fi
            
            ((PY_COUNT++))
            cp "$py_file" "$output_file"
        done
    fi
    
    echo ""
fi

# Second pass: copy all non-Python files
echo "Copying non-Python files..."
echo ""

while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    
    # Skip ignored patterns
    if should_ignore "$file"; then
        continue
    fi
    
    # Skip Python files (already processed)
    if [[ "$file" == *.py ]]; then
        continue
    fi
    
    rel_path="${file#$SOURCE_DIR/}"
    output_file="$OUTPUT_DIR/$rel_path"
    output_dir=$(dirname "$output_file")
    
    # Create directory structure
    if [[ ! -d "$output_dir" ]]; then
        mkdir -p "$output_dir"
        ((DIR_COUNT++))
    fi
    
    ((OTHER_COUNT++))
    echo "[$OTHER_COUNT] Copying: $rel_path"
    
    # Copy file as-is
    cp "$file" "$output_file"
    
done < "$TEMP_LIST"

rm -f "$TEMP_LIST"

echo ""
echo "=========================================================================="
echo "OBFUSCATION & CLONING COMPLETED"
echo "=========================================================================="
echo ""
echo "Statistics:"
echo "  Directories created:     $DIR_COUNT"
echo "  Python files obfuscated: $PY_COUNT"
echo "  Other files copied:      $OTHER_COUNT"
echo "  Total files:             $((PY_COUNT + OTHER_COUNT))"
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""

# -------------------------------------------------------------------------
# Post-processing: create a shareable package and clean up
# Steps:
# 1) Change into the output directory and run its bundled creator_of_clone.sh
#    (if present) to generate scripts/clone.sh and scripts/clone_hashes.txt.
# 2) Create a `hands_shareable` directory next to the output directory.
# 3) Move the `scripts/` directory, `install.sh` and `requirements.txt` from
#    the output directory into `hands_shareable`.
# 4) Remove the original output directory and its contents.
# -------------------------------------------------------------------------

echo "-- Preparing shareable package..."

if [[ -d "$OUTPUT_DIR" ]]; then
    # Run creator_of_clone.sh inside the output directory if available
    if [[ -x "$OUTPUT_DIR/creator_of_clone.sh" ]]; then
        echo "Running creator_of_clone.sh inside $OUTPUT_DIR to generate clone script and hashes..."
        (cd "$OUTPUT_DIR" && ./creator_of_clone.sh .) || echo "Warning: creator_of_clone.sh returned non-zero" >&2
    else
        echo "Note: no creator_of_clone.sh found inside $OUTPUT_DIR (skipping generation)"
    fi

    # Prepare shareable directory next to OUTPUT_DIR
    PARENT_DIR=$(dirname "$OUTPUT_DIR")
    SHARE_DIR="$PARENT_DIR/hands_shareable"

    if [[ -d "$SHARE_DIR" ]]; then
        BACKUP_NAME="$SHARE_DIR.$(date +%s).bak"
        echo "Existing hands_shareable found — moving it to $BACKUP_NAME"
        mv "$SHARE_DIR" "$BACKUP_NAME"
    fi

    echo "Creating shareable directory: $SHARE_DIR"
    mkdir -p "$SHARE_DIR"

    # Move only clone.sh and clone_hashes.txt from scripts directory
    if [[ -d "$OUTPUT_DIR/scripts" ]]; then
        echo "Preparing to move clone.sh and clone_hashes.txt to $SHARE_DIR/scripts/..."
        mkdir -p "$SHARE_DIR/scripts"
        
        # Move only clone.sh and clone_hashes.txt
        if [[ -f "$OUTPUT_DIR/scripts/clone.sh" ]]; then
            echo "Moving scripts/clone.sh -> $SHARE_DIR/scripts/"
            mv "$OUTPUT_DIR/scripts/clone.sh" "$SHARE_DIR/scripts/"
        fi
        if [[ -f "$OUTPUT_DIR/scripts/clone_hashes.txt" ]]; then
            echo "Moving scripts/clone_hashes.txt -> $SHARE_DIR/scripts/"
            mv "$OUTPUT_DIR/scripts/clone_hashes.txt" "$SHARE_DIR/scripts/"
        fi
        
        echo "Note: verify_clone.py, __init__.py, obfuscate_workspace.sh, and README.md remain in $OUTPUT_DIR/scripts/"
    else
        echo "Warning: $OUTPUT_DIR/scripts not found — nothing to move" >&2
    fi

    # Move install.sh and requirements.txt if present
    if [[ -f "$OUTPUT_DIR/install.sh" ]]; then
        echo "Moving install.sh to $SHARE_DIR/installation/"
        mkdir -p "$SHARE_DIR/installation"
        mv "$OUTPUT_DIR/install.sh" "$SHARE_DIR/installation/"
    else
        echo "Note: $OUTPUT_DIR/install.sh not present" >&2
    fi

    if [[ -f "$OUTPUT_DIR/requirements.txt" ]]; then
        echo "Moving requirements.txt to $SHARE_DIR/installation/"
        mv "$OUTPUT_DIR/requirements.txt" "$SHARE_DIR/installation/"
    else
        echo "Note: $OUTPUT_DIR/requirements.txt not present" >&2
    fi

    # NOTE: Do NOT move creator_of_clone.sh — it remains in the output directory per request

    # Remove the (now mostly empty) original output directory
    echo "Removing original output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"

    # Rename the temporary shareable directory to the original output directory name
    if [[ -d "$SHARE_DIR" ]]; then
        echo "Renaming $SHARE_DIR -> $OUTPUT_DIR"
        mv "$SHARE_DIR" "$OUTPUT_DIR"
        echo "Shareable package ready at: $OUTPUT_DIR/installation"
    else
        echo "ERROR: Expected shareable directory $SHARE_DIR not found after cleanup" >&2
    fi
else
    echo "ERROR: Output directory $OUTPUT_DIR does not exist — cannot prepare shareable package" >&2
fi
