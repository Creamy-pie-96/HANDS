#!/usr/bin/env python3
"""
Compact config.json arrays to single lines.

Transforms:
  "key": [
    value,
    "description"
  ]

To:
  "key": [value, "description"]
"""

import json
import re
import sys
from pathlib import Path


def compact_json(input_path: str, output_path: str = None):
    """
    Compact JSON arrays to single lines while preserving structure.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output file (defaults to input_path)
    """
    if output_path is None:
        output_path = input_path
    
    # Read the JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Custom JSON encoder that compacts small arrays
    def custom_dumps(obj, indent=2, level=0):
        """Recursively dump JSON with compact arrays."""
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for key, value in obj.items():
                key_str = json.dumps(key)
                value_str = custom_dumps(value, indent, level + 1)
                items.append(f'{" " * indent * (level + 1)}{key_str}: {value_str}')
            return "{\n" + ",\n".join(items) + "\n" + " " * indent * level + "}"
        elif isinstance(obj, list):
            # Check if this is a [value, description] pair or small array
            if len(obj) <= 3 and all(not isinstance(item, (dict, list)) or 
                                     (isinstance(item, list) and len(item) <= 3 and 
                                      all(not isinstance(x, (dict, list)) for x in item)) 
                                     for item in obj):
                # Compact array on one line
                return json.dumps(obj)
            else:
                # Multi-line array for complex nested structures
                items = []
                for item in obj:
                    item_str = custom_dumps(item, indent, level + 1)
                    items.append(" " * indent * (level + 1) + item_str)
                return "[\n" + ",\n".join(items) + "\n" + " " * indent * level + "]"
        else:
            return json.dumps(obj)
    
    result = custom_dumps(data)
    
    with open(output_path, 'w') as f:
        f.write(result + '\n')
    
    print(f"✓ Compacted config saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to config.json in parent directory
        config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    else:
        config_path = Path(sys.argv[1])
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    compact_json(str(config_path), output_path)
