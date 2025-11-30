"""
Configuration Management for HANDS System

Loads and provides access to configuration from config.json.
Allows runtime configuration of all gesture thresholds and system parameters.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Singleton configuration manager that loads from config.json
    """
    _instance = None
    _config_data: Dict[str, Any] = {}
    _config_path: str = ""
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __new__(cls, *args, **kwargs):
        # Allow passing through extra args (e.g., Config(path)) without
        # breaking the singleton __new__ signature. This makes calls like
        # `Config(args.config)` safe.
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from JSON file.

        Args:
            config_path: Path to config.json. If provided, force reload from
                         that path. If None, load from the default location
                         only on first initialization.
        """
        # If caller explicitly provided a path, honor it and reload.
        if config_path is not None:
            self._config_path = str(config_path)
            self.reload()
            return

        # Otherwise load once using defaults (existing behavior)
        if not self._config_data:  # Only load once
            # Look for config.json in the script's directory
            config_path = Path(__file__).parent / "config.json"
            self._config_path = str(config_path)
            self.reload()
    
    def reload(self):
        """Reload configuration from file."""
        try:
            with open(self._config_path, 'r') as f:
                self._config_data = json.load(f)
            print(f"✓ Loaded configuration from {self._config_path}")
        except FileNotFoundError:
            print(f"⚠ Config file not found: {self._config_path}")
            print("  Using default values")
            self._config_data = self._get_defaults()
        except json.JSONDecodeError as e:
            print(f"⚠ Error parsing config file: {e}")
            print("  Using default values")
            self._config_data = self._get_defaults()
    
    def save(self):
        """Save current configuration back to file."""
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self._config_data, f, indent=2)
            print(f"✓ Saved configuration to {self._config_path}")
        except Exception as e:
            print(f"✗ Error saving config: {e}")
    
    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.
        
        Examples:
            config.get('camera', 'width')  # Returns 640
            config.get('gesture_thresholds', 'pinch', 'threshold_rel')
        
        Args:
            keys: Path to value (e.g., 'gesture_thresholds', 'pinch', 'threshold_rel')
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default
        """
        current = self._config_data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def set(self, *keys, value):
        """
        Set configuration value using dot notation.
        
        Example:
            config.set('camera', 'width', value=1280)
        """
        if len(keys) == 0:
            return
        
        current = self._config_data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_defaults(self) -> Dict:
        """Return default configuration values."""
        return {
            "gesture_thresholds": {
                "pinch": {
                    "threshold_rel": 0.055,
                    "hold_frames": 5,
                    "cooldown_seconds": 0.6
                },
                "pointing": {
                    "min_extension_ratio": 0.12,
                    "max_speed": 0.5,
                    "max_extra_fingers": 1
                },
                "swipe": {
                    "velocity_threshold": 0.8,
                    "cooldown_seconds": 0.5,
                    "history_size": 8
                },
                "zoom": {
                    "scale_threshold": 0.15,
                    "finger_gap_threshold": 0.06,
                    "history_size": 5
                },
                "open_hand": {
                    "min_fingers": 4,
                    "pinch_exclusion_distance": 0.08
                }
            },
            "system_control": {
                "cursor": {
                    "smoothing_factor": 0.3,
                    "speed_multiplier": 1.5,
                    "bounds_padding": 10,
                    "precision_damping": 0.3
                },
                "click": {
                    "double_click_timeout": 0.5,
                    "drag_hold_duration": 1.0
                },
                "scroll": {
                    "sensitivity": 30,
                    "horizontal_enabled": True,
                    "vertical_enabled": True
                },
                "zoom": {
                    "sensitivity": 5,
                    "use_system_zoom": True
                }
            },
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 60
            },
            "performance": {
                "show_fps": True,
                "show_debug_info": False,
                "max_hands": 2,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.3
            }
        }
    
    @property
    def data(self) -> Dict:
        """Get entire configuration dictionary."""
        return self._config_data


# Global configuration instance
config = Config()


# Convenience functions for common access patterns
def get_gesture_threshold(gesture_name: str, param_name: str, default=None):
    """Get a gesture threshold parameter."""
    return config.get('gesture_thresholds', gesture_name, param_name, default=default)


def get_system_control(category: str, param_name: str, default=None):
    """Get a system control parameter."""
    return config.get('system_control', category, param_name, default=default)


def get_visual_setting(param_name: str, default=None):
    """Get a visual feedback setting."""
    return config.get('visual_feedback', param_name, default=default)


if __name__ == "__main__":
    # Test configuration loading
    print("\n=== Configuration Test ===\n")
    
    print("Gesture Thresholds:")
    print(f"  Pinch threshold: {get_gesture_threshold('pinch', 'threshold_rel')}")
    print(f"  Zoom scale threshold: {get_gesture_threshold('zoom', 'scale_threshold')}")
    print(f"  Swipe velocity: {get_gesture_threshold('swipe', 'velocity_threshold')}")
    
    print("\nSystem Control:")
    print(f"  Cursor smoothing: {get_system_control('cursor', 'smoothing_factor')}")
    print(f"  Scroll sensitivity: {get_system_control('scroll', 'sensitivity')}")
    
    print("\nCamera:")
    print(f"  Resolution: {config.get('camera', 'width')}x{config.get('camera', 'height')}")
    
    print("\n✓ Configuration system working!")
