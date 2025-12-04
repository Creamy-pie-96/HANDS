"""
Configuration Management for HANDS System

Loads and provides access to configuration from config.json.
Allows runtime configuration of all gesture thresholds and system parameters.
Supports both old format (direct values) and new format ([value, description]).
"""

import json
import os
from typing import Dict, Any, Optional, Tuple
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
        Handles both old format (direct values) and new format ([value, description]).
        
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
        
        # Handle new [value, description] format
        if isinstance(current, list) and len(current) >= 1:
            return current[0]  # Return the value part
        
        return current
    
    def get_with_description(self, *keys, default=None) -> Tuple[Any, str]:
        """
        Get configuration value AND description.
        
        Returns:
            Tuple of (value, description) or (default, "")
        """
        current = self._config_data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return (default, "")
        
        # Handle new [value, description] format
        if isinstance(current, list):
            if len(current) >= 2:
                return (current[0], current[1])
            elif len(current) == 1:
                return (current[0], "")
        
        # Old format or plain value
        return (current, "")
    
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
                    "max_extra_fingers": 1,
                    "ewma_alpha": 0.4
                },
                "swipe": {
                    "ewma_alpha": 0.3,
                    "velocity_threshold_x": 0.4,
                    "velocity_threshold_y": 0.4,
                    "confidence_ramp_up": 0.25,
                    "confidence_decay": 0.15,
                    "confidence_threshold": 0.6,
                    "max_velocity_x": 2.0,
                    "max_velocity_y": 2.0
                },
                "finger_extension": {
                    "open_ratio": 1.20,
                    "close_ratio": 1.10,
                    "motion_speed_threshold": 0.15,
                    "motion_sigmoid_k": 20.0
                },
                "zoom": {
                    "finger_gap_threshold": 0.10,
                    "ewma_alpha": 0.3,
                    "velocity_threshold": 0.08,
                    "confidence_ramp_up": 0.25,
                    "confidence_decay": 0.15,
                    "confidence_threshold": 0.6,
                    "max_velocity": 1.5,
                    "require_fingers_extended": False
                },
                "open_hand": {
                    "min_fingers": 4,
                    "pinch_exclusion_distance": 0.08
                },
                "thumbs": {
                    "velocity_threshold": 0.2,
                    "ewma_alpha": 0.3,
                    "hold_frames": 5,
                    "confidence_ramp_up": 0.3,
                    "confidence_decay": 0.2,
                    "confidence_threshold": 0.6
                }
            },
            "system_control": {
                "cursor": {
                    "smoothing_factor": 0.3,
                    "speed_multiplier": 1.5,
                    "bounds_padding": 10,
                    "precision_damping": 0.3,
                    "dead_zone": 0.02,
                    "magnetic_radius": 0.05,
                    "magnetic_factor": 0.4,
                    "screen_bounds_padding": 10,
                    "fallback_screen_width": 1920,
                    "fallback_screen_height": 1080
                },
                "click": {
                    "double_click_timeout": 0.5,
                    "drag_hold_duration": 1.0
                },
                "scroll": {
                    "sensitivity": 30,
                    "speed_neutral": 1.0,
                    "speed_factor": 0.2,
                    "horizontal_enabled": True,
                    "vertical_enabled": True
                },
                "zoom": {
                    "sensitivity": 5,
                    "speed_neutral": 1.0,
                    "speed_factor": 0.2,
                    "use_system_zoom": True
                }
            },
            "bimanual_gestures": {
                "hand_still_threshold": 0.3,
                "pan_velocity_threshold": 0.4,
                "draw_velocity_threshold": 0.2,
                "precision_damping_factor": 0.3,
                "two_hand_distance_threshold": 0.1,
                "warp_min_distance": 0.3
            },
            "visual_feedback": {
                "enabled": True,
                "show_hand_skeleton": True,
                "show_fingertips": True,
                "show_cursor_preview": True,
                "show_gesture_name": True,
                "overlay_opacity": 0.7,
                "velocity_arrow_scale": 0.25,
                "velocity_threshold_highlight": 0.8,
                "fingertip_dim_factor": 0.4,
                "debug_panel": {
                    "margin": 12,
                    "spacing": 8,
                    "base_width": 240,
                    "line_height": 16,
                    "title_height": 20,
                    "background_alpha": 0.45,
                    "spacing_buffer": 4,
                    "start_y_offset": 8,
                    "scan_step_horizontal": 30,
                    "scan_step_vertical": 20
                },
                "gesture_panel": {
                    "max_height": 200,
                    "panel_y": 10,
                    "panel_left_x": 10,
                    "panel_width": 300,
                    "overlay_alpha": 0.7,
                    "frame_blend": 0.3,
                    "title_y_offset": 30,
                    "title_x": 20,
                    "separator_y_offset": 30,
                    "separator_left_x": 20,
                    "separator_right_x": 290,
                    "separator_bottom_spacing": 20,
                    "max_gestures_display": 4,
                    "gesture_indicator_x": 25,
                    "gesture_indicator_radius": 5,
                    "gesture_name_x": 40,
                    "gesture_name_y_adjust": 5,
                    "hint_x": 180,
                    "line_spacing": 20,
                    "param_indent_x": 45,
                    "param_line_spacing": 15,
                    "spacing_with_hint": 25,
                    "spacing_no_hint": 25,
                    "no_gesture_x": 40
                },
                "cursor_preview": {
                    "trail_fade_time": 0.5,
                    "circle_radius": 15,
                    "crosshair_length": 20,
                    "crosshair_gap": 8
                },
                "animation": {
                    "pulse_frequency": 2.0
                }
            },
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 60
            },
            "performance": {
                "use_gpu": True,
                "show_fps": True,
                "show_debug_info": False,
                "max_hands": 2,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.3,
                "gesture_history_maxlen": 16,
                "bimanual_history_maxlen": 10
            },
            "display": {
                "window_width": 1280,
                "window_height": 720,
                "flip_horizontal": True,
                "fps_update_interval": 30,
                "visual_mode": "full",
                "status_indicator": {
                    "enabled": True,
                    "size": 64,
                    "opacity": 0.8,
                    "position": "top-right",
                    "margin_x": 20,
                    "margin_y": 20,
                    "stickers_base_path": "../gui/stickers",
                    "stickers": {
                        "pointing": "pointing.png",
                        "pinch": "pinch.png",
                        "zoom_in": "zoom_in.png",
                        "zoom_out": "zoom_out.png",
                        "swipe_up": "swipe_up.png",
                        "swipe_down": "swipe_down.png",
                        "swipe_left": "swipe_left.png",
                        "swipe_right": "swipe_right.png",
                        "open_hand": "open_hand.png",
                        "thumbs_up": "thumbs_up.png",
                        "thumbs_down": "thumbs_down.png",
                        "thumbs_up_moving_up": "thumbs_up_moving_up.png",
                        "thumbs_up_moving_down": "thumbs_up_moving_down.png",
                        "thumbs_down_moving_up": "thumbs_down_moving_up.png",
                        "thumbs_down_moving_down": "thumbs_down_moving_down.png"
                    },
                    "colors": {
                        "red": [255, 50, 50],
                        "yellow": [255, 200, 0],
                        "blue": [50, 150, 255]
                    }
                }
            },
            "app_control": {
                "pause": False,
                "exit": False
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


def get_bimanual_setting(param_name: str, default=None):
    """Get a bimanual gesture parameter."""
    return config.get('bimanual_gestures', param_name, default=default)


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
