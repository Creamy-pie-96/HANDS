#!/usr/bin/env python3
"""
Workspace Clone Verification Script

Compares two directory trees and verifies that all files are identical
using SHA256 hashing. Ignores specified patterns like .venv, .git, etc.
"""

import os
import hashlib
import sys
from pathlib import Path
from typing import Set, Dict, Tuple


# Patterns to ignore (same as creator_of_clone.sh)
IGNORE_PATTERNS = {
    ".venv",
    ".git",
    ".gitignore",
    "install.sh",
    "requirements.txt",
    "__pycache__",
    ".DS_Store",
    "creator_of_clone.sh",
    "clone.sh",
    "verify_clone.py",
    "test_clone",
    "test_clone_final",
}

IGNORE_EXTENSIONS = {".pyc", ".pyo", ".pyd"}


def should_ignore(path: Path) -> bool:
    """Check if a path should be ignored."""
    # Check basename
    if path.name in IGNORE_PATTERNS:
        return True
    
    # Check extension
    if path.suffix in IGNORE_EXTENSIONS:
        return True
    
    # Check if any parent directory is in ignore patterns
    for parent in path.parents:
        if parent.name in IGNORE_PATTERNS:
            return True
    
    return False


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {filepath}: {e}")
        return ""


def scan_directory(root_dir: Path) -> Dict[str, str]:
    """
    Scan directory and return dict of {relative_path: hash}.
    Ignores files matching ignore patterns.
    """
    file_hashes = {}
    
    for filepath in root_dir.rglob('*'):
        if filepath.is_file():
            rel_path = filepath.relative_to(root_dir)
            # Check ignore patterns against relative path components
            if should_ignore_relative(rel_path):
                continue
            file_hash = calculate_file_hash(filepath)
            if file_hash:
                file_hashes[str(rel_path)] = file_hash
    
    return file_hashes


def should_ignore_relative(rel_path: Path) -> bool:
    """Check if a relative path should be ignored."""
    # Check each component of the path
    for part in rel_path.parts:
        if part in IGNORE_PATTERNS:
            return True
        # Check extension
        if Path(part).suffix in IGNORE_EXTENSIONS:
            return True
    return False


def compare_directories(original_dir: Path, cloned_dir: Path) -> Tuple[bool, Dict]:
    """
    Compare two directories and return (success, stats).
    
    Returns:
        (bool, dict): (True if identical, statistics dict)
    """
    print("=" * 70)
    print("WORKSPACE CLONE VERIFICATION")
    print("=" * 70)
    print()
    print(f"Original: {original_dir}")
    print(f"Cloned:   {cloned_dir}")
    print()
    
    # Check if directories exist
    if not original_dir.exists():
        print(f"❌ Original directory not found: {original_dir}")
        return False, {}
    
    if not cloned_dir.exists():
        print(f"❌ Cloned directory not found: {cloned_dir}")
        return False, {}
    
    print("Scanning original directory...")
    original_hashes = scan_directory(original_dir)
    print(f"✓ Found {len(original_hashes)} files in original")
    
    print("Scanning cloned directory...")
    cloned_hashes = scan_directory(cloned_dir)
    print(f"✓ Found {len(cloned_hashes)} files in clone")
    print()
    
    # Compare file sets
    original_files = set(original_hashes.keys())
    cloned_files = set(cloned_hashes.keys())
    
    missing_files = original_files - cloned_files
    extra_files = cloned_files - original_files
    common_files = original_files & cloned_files
    
    # Check hashes of common files
    different_files = []
    identical_files = []
    
    for filepath in sorted(common_files):
        if original_hashes[filepath] != cloned_hashes[filepath]:
            different_files.append(filepath)
        else:
            identical_files.append(filepath)
    
    # Display results
    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print()
    
    all_match = (len(missing_files) == 0 and 
                 len(extra_files) == 0 and 
                 len(different_files) == 0)
    
    if all_match:
        print("✅ SUCCESS: All files match perfectly!")
        print()
        print(f"  Total files verified: {len(identical_files)}")
    else:
        print("❌ VERIFICATION FAILED")
        print()
    
    # Show statistics
    print("Statistics:")
    print(f"  Identical files:     {len(identical_files)}")
    print(f"  Different content:   {len(different_files)}")
    print(f"  Missing in clone:    {len(missing_files)}")
    print(f"  Extra in clone:      {len(extra_files)}")
    print()
    
    # Show details if there are issues
    if missing_files:
        print("Missing files in clone:")
        for filepath in sorted(missing_files):
            print(f"  - {filepath}")
        print()
    
    if extra_files:
        print("Extra files in clone (not in original):")
        for filepath in sorted(extra_files):
            print(f"  - {filepath}")
        print()
    
    if different_files:
        print("Files with different content:")
        for filepath in sorted(different_files):
            print(f"  - {filepath}")
            print(f"      Original: {original_hashes[filepath]}")
            print(f"      Cloned:   {cloned_hashes[filepath]}")
        print()
    
    # Show sample of matched files
    if identical_files and all_match:
        print("Sample of verified files:")
        for filepath in sorted(identical_files)[:10]:
            print(f"  ✓ {filepath}")
        if len(identical_files) > 10:
            print(f"  ... and {len(identical_files) - 10} more")
        print()
    
    print("=" * 70)
    
    stats = {
        'identical': len(identical_files),
        'different': len(different_files),
        'missing': len(missing_files),
        'extra': len(extra_files),
        'total_original': len(original_files),
        'total_cloned': len(cloned_files),
    }
    
    return all_match, stats


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python3 verify_clone.py <original_dir> <cloned_dir>")
        print()
        print("Example:")
        print("  python3 verify_clone.py . test_clone")
        sys.exit(1)
    
    original_dir = Path(sys.argv[1]).resolve()
    cloned_dir = Path(sys.argv[2]).resolve()
    
    success, stats = compare_directories(original_dir, cloned_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
