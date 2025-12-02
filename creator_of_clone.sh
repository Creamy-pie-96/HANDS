#!/bin/bash

################################################################################
# creator_of_clone.sh
# 
# Recursively walks through the workspace and creates a clone.sh script that
# can recreate the entire workspace structure and files.
#
# Ignores: .venv, .git, install.sh, requirements.txt, __pycache__, *.pyc
################################################################################

# Get the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_SCRIPT="$SCRIPT_DIR/scripts/clone.sh"

# By default we scan the script directory, but allow an optional scan directory
# Usage: ./creator_of_clone.sh [--scan-dir /path/to/parent] or ./creator_of_clone.sh /path/to/parent
SCAN_DIR="$SCRIPT_DIR"
if [[ "$#" -ge 1 ]]; then
    case "$1" in
        --scan-dir)
            if [[ -n "${2-}" && -d "$2" ]]; then
                SCAN_DIR="$(cd "$2" && pwd)"
                shift 2
            else
                echo "Usage: $0 [--scan-dir DIR]" >&2
                exit 2
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [--scan-dir DIR]" >&2
            exit 0
            ;;
        *)
            # treat first positional arg as scan directory if it exists
            if [[ -d "$1" ]]; then
                SCAN_DIR="$(cd "$1" && pwd)"
                shift
            else
                echo "Unknown argument: $1" >&2
                echo "Usage: $0 [--scan-dir DIR]" >&2
                exit 2
            fi
            ;;
    esac
fi

echo "Scanning root: $SCAN_DIR"

# Files and directories to ignore
IGNORE_PATTERNS=(
    ".venv"
    ".git"
    ".gitignore"
    "install.sh"
    "requirements.txt"
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".DS_Store"
    "creator_of_clone.sh"
    "scripts/clone.sh"
    "scripts/clone_debug.sh"
    "scripts/clone_noerr.sh"
    "test_clone"
    "test_clone_final"
    "test_script.sh"
    "test_partial.sh"
    "test.txt"
    "stdout.log"
    "stdout.txt"
    "stderr.log"
    "stderr.txt"
)

echo "=================================================="
echo "WORKSPACE CLONE SCRIPT GENERATOR"
echo "=================================================="
echo ""
echo "Working directory: $SCRIPT_DIR"
echo "Output script: $OUTPUT_SCRIPT"
echo ""

# Create scripts directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/scripts"

# Function to check if a path should be ignored
should_ignore() {
    local path="$1"
    local basename=$(basename "$path")
    
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        # Check if it's a direct match or glob match
        if [[ "$basename" == "$pattern" ]] || [[ "$basename" == $pattern ]]; then
            return 0  # Should ignore
        fi
        
        # Check if path contains the pattern as a directory component
        if [[ "$path" == *"/$pattern/"* ]] || [[ "$path" == *"/$pattern" ]]; then
            return 0  # Should ignore
        fi
    done
    
    return 1  # Should not ignore
}

# Start writing the clone.sh script
cat > "$OUTPUT_SCRIPT" << 'CLONE_HEADER'
#!/bin/bash

################################################################################
# clone.sh - Workspace Cloning Script
# 
# Recreates the entire workspace structure and files.
# 
# Usage:
#   ./clone.sh                    # Creates in current directory
#   ./clone.sh -d /path/to/dir    # Creates in specified directory
################################################################################

set -uo pipefail

# Parse command line arguments
TARGET_DIR="."

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-d|--directory TARGET_DIR]"
            echo ""
            echo "Options:"
            echo "  -d, --directory DIR    Target directory for cloning (default: current directory)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Convert to absolute path
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

echo "=================================================="
echo "WORKSPACE CLONING IN PROGRESS"
echo "=================================================="
echo ""
echo "Target directory: $TARGET_DIR"
echo ""

# Counter for files created
FILE_COUNT=0
DIR_COUNT=0

# Function to create a file with content
create_file() {
    local rel_path="$1"
    local full_path="$TARGET_DIR/$rel_path"
    local dir_path=$(dirname "$full_path")
    
    # Create directory if needed
    if [[ ! -d "$dir_path" ]]; then
        mkdir -p "$dir_path"
        ((DIR_COUNT++))
    fi
    
    # Create the file (content follows in the next cat command)
    ((FILE_COUNT++))
    echo "[$FILE_COUNT] Creating: $rel_path"
}

CLONE_HEADER

echo "Scanning workspace..."
echo ""

# Find all files (excluding ignored patterns)
FILE_COUNTER=0

# Get list of files
TEMP_LIST=$(mktemp)
find "$SCAN_DIR" -type f | sort > "$TEMP_LIST"

# Process each file
while IFS= read -r file; do
    # Skip empty lines
    [[ -z "$file" ]] && continue
    
    # Get relative path from scan directory
    rel_path="${file#$SCAN_DIR/}"
    
    # Skip if should be ignored
    if should_ignore "$file"; then
        continue
    fi
    
    ((FILE_COUNTER++))
    echo "[$FILE_COUNTER] Found: $rel_path"
    
    # Add file creation command to clone.sh
    echo "" >> "$OUTPUT_SCRIPT"
    echo "# File $FILE_COUNTER: $rel_path" >> "$OUTPUT_SCRIPT"
    echo "create_file \"$rel_path\"" >> "$OUTPUT_SCRIPT"
    
    # Use base64 encoding to avoid delimiter collision issues
    echo "base64 -d > \"\$TARGET_DIR/$rel_path\" << 'END_OF_FILE_${FILE_COUNTER}'" >> "$OUTPUT_SCRIPT"
    base64 < "$file" >> "$OUTPUT_SCRIPT"
    echo "END_OF_FILE_${FILE_COUNTER}" >> "$OUTPUT_SCRIPT"
    
done < "$TEMP_LIST"

# Cleanup
rm -f "$TEMP_LIST"

# Generate hash file for verification
HASH_FILE="$SCRIPT_DIR/scripts/clone_hashes.txt"
echo "Generating SHA256 hashes for verification..."
echo "# SHA256 hashes of original files" > "$HASH_FILE"
echo "# Generated on $(date)" >> "$HASH_FILE"
echo "# Format: <hash> <relative_path>" >> "$HASH_FILE"
echo "" >> "$HASH_FILE"

# Recalculate hashes for all files
TEMP_LIST2=$(mktemp)
find "$SCAN_DIR" -type f | sort > "$TEMP_LIST2"

HASH_COUNTER=0
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    
    rel_path="${file#$SCAN_DIR/}"
    
    if should_ignore "$file"; then
        continue
    fi

    # Skip hashing the generated clone/hashes and the obfuscated verifier
    # (we still include these files in the clone, but their hashes should
    # not be recorded here to avoid self-referential or changing-hash issues)
    if [[ "$rel_path" == "scripts/clone_hashes.txt" ]] || \
       [[ "$rel_path" == "scripts/verify_clone.py" ]]; then
        continue
    fi
    
    ((HASH_COUNTER++))
    # Calculate SHA256 hash
    file_hash=$(sha256sum "$file" | cut -d' ' -f1)
    echo "$file_hash  $rel_path" >> "$HASH_FILE"
    
done < "$TEMP_LIST2"

rm -f "$TEMP_LIST2"

echo "✓ Generated hashes for $HASH_COUNTER files: $HASH_FILE"
echo ""

# Add footer to clone.sh
cat >> "$OUTPUT_SCRIPT" << 'CLONE_FOOTER'

echo ""
echo "=================================================="
echo "CLONING COMPLETED"
echo "=================================================="
echo ""
echo "Directories created: $DIR_COUNT"
echo "Files created: $FILE_COUNT"
echo "Target location: $TARGET_DIR"
echo ""
CLONE_FOOTER

# Make clone.sh executable
chmod +x "$OUTPUT_SCRIPT"

echo ""
echo "=================================================="
echo "CLONE SCRIPT GENERATION COMPLETED"
echo "=================================================="
echo ""
echo "Files scanned: $FILE_COUNTER"
echo "Output script: $OUTPUT_SCRIPT"
echo ""
echo "To clone the workspace, run:"
echo "  $OUTPUT_SCRIPT"
echo "or"
echo "  $OUTPUT_SCRIPT -d /path/to/target/directory"
echo ""
