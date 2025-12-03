#!/usr/bin/env python3
"""
app_control.py - CLI tool to control HANDS application via config.json

Usage:
    python app_control.py --exit true --pause false
    python app_control.py --exit false
    python app_control.py --pause true
    python app_control.py --config /path/to/config.json --pause true

Modifies the app_control.exit and app_control.pause fields in config.json
to signal the running HANDS application to pause or exit.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def str_to_bool(value: str) -> bool:
    """Convert string to boolean."""
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def update_config(config_path: str, exit_val: Optional[bool] = None, pause_val: Optional[bool] = None) -> bool:
    """
    Update the app_control fields in config.json.
    
    Args:
        config_path: Path to config.json
        exit_val: Value for app_control.exit (None = don't change)
        pause_val: Value for app_control.pause (None = don't change)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure app_control section exists
        if 'app_control' not in config:
            config['app_control'] = {
                'pause': [False, "Set to true to pause gesture control (hot-reloaded)"],
                'exit': [False, "Set to true to gracefully exit the application (hot-reloaded)"]
            }
        
        # Update values
        if exit_val is not None:
            entry = config['app_control'].get('exit', [False, ""])
            if isinstance(entry, list):
                entry[0] = exit_val
                config['app_control']['exit'] = entry
            else:
                config['app_control']['exit'] = [exit_val, "Set to true to gracefully exit the application (hot-reloaded)"]
            print(f"✓ Set app_control.exit = {exit_val}")
        
        if pause_val is not None:
            entry = config['app_control'].get('pause', [False, ""])
            if isinstance(entry, list):
                entry[0] = pause_val
                config['app_control']['pause'] = entry
            else:
                config['app_control']['pause'] = [pause_val, "Set to true to pause gesture control (hot-reloaded)"]
            print(f"✓ Set app_control.pause = {pause_val}")
        
        # Write back with proper formatting
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Config saved to {config_path}")
        return True
        
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in config file: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Error updating config: {e}", file=sys.stderr)
        return False


def get_default_config_path() -> str:
    """Get the default config.json path."""
    # Try relative to this script first
    script_dir = Path(__file__).parent
    
    # Check common locations
    candidates = [
        script_dir.parent / 'config' / 'config.json',  # source_code/config/config.json
        script_dir / 'config.json',  # Same directory
        Path.cwd() / 'source_code' / 'config' / 'config.json',  # From project root
        Path.cwd() / 'config.json',  # Current directory
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    # Default to first candidate
    return str(candidates[0])


def main():
    parser = argparse.ArgumentParser(
        description="Control HANDS application via config.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
python app_control.py --exit true           # Signal app to exit
python app_control.py --pause true          # Pause gesture control
python app_control.py --pause false         # Resume gesture control
python app_control.py --exit true --pause false  # Both at once
python app_control.py --config /path/to/config.json --exit true
python app_control.py --status              # Show current values
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--exit', '-e',
        type=str,
        default=None,
        metavar='BOOL',
        help='Set app_control.exit (true/false)'
    )
    
    parser.add_argument(
        '--pause', '-p',
        type=str,
        default=None,
        metavar='BOOL',
        help='Set app_control.pause (true/false)'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current app_control values'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    config_path = args.config if args.config else get_default_config_path()
    
    # Status mode
    if args.status:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            app_control = config.get('app_control', {})
            pause = app_control.get('pause', [False])[0] if isinstance(app_control.get('pause'), list) else app_control.get('pause', False)
            exit_val = app_control.get('exit', [False])[0] if isinstance(app_control.get('exit'), list) else app_control.get('exit', False)
            
            print(f"Config: {config_path}")
            print(f"  app_control.pause = {pause}")
            print(f"  app_control.exit  = {exit_val}")
            return 0
        except Exception as e:
            print(f"✗ Error reading config: {e}", file=sys.stderr)
            return 1
    
    # Validate arguments
    if args.exit is None and args.pause is None:
        parser.print_help()
        print("\nError: At least one of --exit or --pause must be specified", file=sys.stderr)
        return 1
    
    # Parse boolean values
    exit_val = None
    pause_val = None
    
    try:
        if args.exit is not None:
            exit_val = str_to_bool(args.exit)
        if args.pause is not None:
            pause_val = str_to_bool(args.pause)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Update config
    success = update_config(config_path, exit_val=exit_val, pause_val=pause_val)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
