"""
System Controller for HANDS

Interfaces between gesture detection and actual system control (mouse/keyboard).
Uses pynput for reliable cross-platform input control.
"""

import time
import threading
from typing import Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np

try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠ pynput not available. Install with: pip install pynput")

# Decorator to mark methods as exposed for the config GUI
def exposed_action(func):
    func._is_exposed_action = True
    return func

try:
    import screeninfo
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("⚠ screeninfo not available. Install with: pip install screeninfo")

from source_code.utils.math_utils import EWMA


@dataclass
class VelocitySensitivityConfig:
    """
    Configuration for velocity-based sensitivity modulation.
    
    This is the core algorithm used to adjust action rates based on gesture velocity.
    Formula:
        M = 1.0 + speed_factor * (velocity_norm - speed_neutral)
        M_clamped = clamp(M, 1 - speed_factor, 1 + speed_factor)
        effective_sensitivity = base_sensitivity * M_clamped
        min_delay = base_delay / max(0.05, effective_sensitivity)
    
    Attributes:
        base_sensitivity: Base sensitivity multiplier for the action
        speed_neutral: Velocity at which no modulation occurs (velocity_norm == neutral → M=1)
        speed_factor: How much velocity affects the rate (±speed_factor from neutral)
        base_delay: Base delay between actions in seconds
    """
    base_sensitivity: float = 1.0
    speed_neutral: float = 1.0
    speed_factor: float = 0.2
    base_delay: float = 0.5


class VelocitySensitivity:
    """
    Reusable velocity-based sensitivity calculator.
    
    Implements the sophisticated velocity modulation formula that adjusts
    action rates based on gesture velocity. Faster gestures = faster actions.
    
    Used for: swipe, zoom, thumbs_moving_up/down, and any other velocity-dependent gesture.
    """
    
    def __init__(self, config: VelocitySensitivityConfig):
        """
        Initialize velocity sensitivity calculator.
        
        Args:
            config: VelocitySensitivityConfig with modulation parameters
        """
        self.config = config
        self._last_action_time = 0.0
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'VelocitySensitivity':
        """
        Create VelocitySensitivity from a dictionary of parameters.
        
        Args:
            params: Dict with keys: sensitivity, speed_neutral, speed_factor, base_delay
        
        Returns:
            VelocitySensitivity instance
        """
        config = VelocitySensitivityConfig(
            base_sensitivity=float(params.get('sensitivity', 1.0)),
            speed_neutral=float(params.get('speed_neutral', 1.0)),
            speed_factor=float(params.get('speed_factor', 0.2)),
            base_delay=float(params.get('base_delay', 0.5))
        )
        return cls(config)
    
    def calculate_effective_sensitivity(self, velocity_norm: float) -> float:
        """
        Calculate effective sensitivity based on current velocity.
        
        Args:
            velocity_norm: Normalized gesture velocity
        
        Returns:
            Effective sensitivity value (base_sensitivity * modulation)
        """
        m = 1.0 + self.config.speed_factor * (velocity_norm - self.config.speed_neutral)
        m_clamped = max(1.0 - self.config.speed_factor, min(1.0 + self.config.speed_factor, m))
        return self.config.base_sensitivity * m_clamped
    
    def calculate_min_delay(self, velocity_norm: float) -> float:
        """
        Calculate minimum delay between actions based on velocity.
        
        Args:
            velocity_norm: Normalized gesture velocity
        
        Returns:
            Minimum delay in seconds
        """
        s_eff = self.calculate_effective_sensitivity(velocity_norm)
        return self.config.base_delay / max(0.05, s_eff)
    
    def should_act(self, velocity_norm: float) -> bool:
        """
        Check if enough time has passed since last action for velocity-modulated rate.
        
        Args:
            velocity_norm: Normalized gesture velocity
        
        Returns:
            True if action should be performed, False if rate-limited
        """
        now = time.time()
        min_delay = self.calculate_min_delay(velocity_norm)
        
        if now - self._last_action_time < min_delay:
            return False
        
        return True
    
    def record_action(self):
        """Record that an action was performed (for rate limiting)."""
        self._last_action_time = time.time()
    
    def try_act(self, velocity_norm: float) -> bool:
        """
        Check if action is allowed and record it if so.
        
        Combines should_act() and record_action() for convenience.
        
        Args:
            velocity_norm: Normalized gesture velocity
        
        Returns:
            True if action was allowed (and recorded), False if rate-limited
        """
        if self.should_act(velocity_norm):
            self.record_action()
            return True
        return False
    
    def reset(self):
        """Reset the last action time (e.g., when gesture ends)."""
        self._last_action_time = 0.0


@dataclass
class ScreenBounds:
    """Screen dimensions and boundaries."""
    width: int
    height: int
    padding: int
    
    def contains(self, x: int, y: int) -> bool:
        """Check if point is within bounds."""
        return (self.padding <= x < self.width - self.padding and
                self.padding <= y < self.height - self.padding)
    
    def clamp(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to screen bounds."""
        x = max(self.padding, min(x, self.width - self.padding - 1))
        y = max(self.padding, min(y, self.height - self.padding - 1))
        return (x, y)


class SystemController:
    """
    Controls system mouse and keyboard based on gestures.
    Provides smooth cursor movement, click detection, and keyboard shortcuts.
    """
    
    def __init__(self, config=None):
        """
        Initialize system controller.
        
        Args:
            config: Configuration object (from config_manager)
        """
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput library is required. Install with: pip install pynput")
        
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        
        # Load configuration
        if config:
            from source_code.config.config_manager import get_system_control, get_velocity_sensitivity_config
            self.cursor_smoothing = get_system_control('cursor', 'smoothing_factor', 0.3)
            self.cursor_speed = get_system_control('cursor', 'speed_multiplier', 1.5)
            self.precision_damping = get_system_control('cursor', 'precision_damping', 0.3)
            self.dead_zone = get_system_control('cursor', 'dead_zone', 0.02)
            self.magnetic_radius = get_system_control('cursor', 'magnetic_radius', 0.05)
            self.magnetic_factor = get_system_control('cursor', 'magnetic_factor', 0.4)
            self.screen_bounds_padding = get_system_control('cursor', 'screen_bounds_padding', 10)
            self.fallback_width = get_system_control('cursor', 'fallback_screen_width', 1920)
            self.fallback_height = get_system_control('cursor', 'fallback_screen_height', 1080)
            self.double_click_timeout = get_system_control('click', 'double_click_timeout', 0.5)
            self.drag_hold_duration = get_system_control('click', 'drag_hold_duration', 1.0)
            
            # Initialize velocity sensitivity for each velocity-dependent gesture type
            # Each uses the reusable VelocitySensitivity calculator
            # Note: Scroll uses swipe_up/swipe_down sensitivity (scroll = swipe up/down)
            #       Workspace switch uses swipe_left/swipe_right sensitivity
            self.velocity_sensitivity = {
                'zoom': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('zoom')),
                'swipe_left': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('swipe_left')),
                'swipe_right': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('swipe_right')),
                'swipe_up': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('swipe_up')),
                'swipe_down': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('swipe_down')),
                'thumbs_up_moving_up': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('thumbs_up_moving_up')),
                'thumbs_up_moving_down': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('thumbs_up_moving_down')),
                'thumbs_down_moving_up': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('thumbs_down_moving_up')),
                'thumbs_down_moving_down': VelocitySensitivity.from_dict(get_velocity_sensitivity_config('thumbs_down_moving_down')),
            }
        else:
            self.cursor_smoothing = 0.3
            self.cursor_speed = 1.5
            self.precision_damping = 0.3
            self.dead_zone = 0.02
            self.magnetic_radius = 0.05
            self.magnetic_factor = 0.4
            self.screen_bounds_padding = 10
            self.fallback_width = 1920
            self.fallback_height = 1080
            self.double_click_timeout = 0.5
            self.drag_hold_duration = 1.0

            # Default velocity sensitivity configurations
            # Note: swipe_up/down control scroll, swipe_left/right control workspace switch
            scroll_config = VelocitySensitivityConfig(base_sensitivity=5.0, speed_neutral=1.0, speed_factor=0.3, base_delay=0.05)
            workspace_config = VelocitySensitivityConfig(base_sensitivity=1.0, speed_neutral=1.0, speed_factor=0.3, base_delay=0.3)
            zoom_config = VelocitySensitivityConfig(base_sensitivity=2.0, speed_neutral=1.0, speed_factor=0.2, base_delay=0.1)
            thumbs_config = VelocitySensitivityConfig(base_sensitivity=1.0, speed_neutral=0.5, speed_factor=0.4, base_delay=0.15)
            self.velocity_sensitivity = {
                'zoom': VelocitySensitivity(zoom_config),
                'swipe_left': VelocitySensitivity(workspace_config),
                'swipe_right': VelocitySensitivity(workspace_config),
                'swipe_up': VelocitySensitivity(scroll_config),
                'swipe_down': VelocitySensitivity(scroll_config),
                'thumbs_up_moving_up': VelocitySensitivity(thumbs_config),
                'thumbs_up_moving_down': VelocitySensitivity(thumbs_config),
                'thumbs_down_moving_up': VelocitySensitivity(thumbs_config),
                'thumbs_down_moving_down': VelocitySensitivity(thumbs_config),
            }
        
        # Get screen dimensions
        self.screen = self._get_screen_bounds()
        
        # Cursor smoothing filter
        self.cursor_filter_x = EWMA(alpha=self.cursor_smoothing)
        self.cursor_filter_y = EWMA(alpha=self.cursor_smoothing)
        
        # Click state tracking
        self.last_click_time = 0.0
        self.click_count = 0
        self.is_dragging = False
        self.pinch_start_time = None
        
        # Pause state
        self.paused = False
        
        # Current cursor position (normalized)
        self.current_norm_pos = (0.5, 0.5)
    def _get_screen_bounds(self) -> ScreenBounds:
        """Get screen dimensions."""
        if SCREENINFO_AVAILABLE:
            try:
                monitors = screeninfo.get_monitors()
                if monitors:
                    primary = monitors[0]
                    return ScreenBounds(width=primary.width, height=primary.height, padding=self.screen_bounds_padding)
            except Exception as e:
                print(f"⚠ Error getting screen info: {e}")
        
        print(f"⚠ Using default screen resolution {self.fallback_width}x{self.fallback_height}")
        return ScreenBounds(width=self.fallback_width, height=self.fallback_height, padding=self.screen_bounds_padding)
    
    @exposed_action
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        return self.paused
    
    def normalized_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Convert normalized coordinates (0..1) to screen pixels.
        
        Args:
            norm_x: Normalized X coordinate (0 = left, 1 = right)
            norm_y: Normalized Y coordinate (0 = top, 1 = bottom)
        
        Returns:
            (screen_x, screen_y) in pixels
        """
        screen_x = int(norm_x * self.screen.width)
        screen_y = int(norm_y * self.screen.height)
        return self.screen.clamp(screen_x, screen_y)
    
    @exposed_action
    def move_cursor(self, norm_x: float, norm_y: float, precision_mode: bool = False):
        """
        Move cursor to normalized position with smoothing and stabilization.
        
        Args:
            norm_x: Target X position (0..1) - normalized to camera frame
            norm_y: Target Y position (0..1) - normalized to camera frame
            precision_mode: If True, apply extra damping for fine control
        """
        if self.paused:
            return
        
        # Apply smoothing
        smooth_x = self.cursor_filter_x.update(norm_x)
        smooth_y = self.cursor_filter_y.update(norm_y)
        
        # MAGNETIC STABILIZATION: Create dead zones for easier control
        # When cursor is near current position, require larger movement to break free
        current_x, current_y = self.current_norm_pos
        dx = smooth_x - current_x
        dy = smooth_y - current_y
        distance = np.hypot(dx, dy)
        
        # Dead zone threshold - small movements are ignored (reduces shake)
        if distance < self.dead_zone and not precision_mode:
            # Too small - don't move (magnetic sticking)
            return
        
        # Magnetic damping: Movement near current position is slower
        if distance < self.magnetic_radius:
            # Apply extra damping (magnet effect)
            smooth_x = current_x + dx * self.magnetic_factor
            smooth_y = current_y + dy * self.magnetic_factor
        
        # Apply precision damping if needed
        if precision_mode:
            # Interpolate between current and target
            smooth_x = current_x + (smooth_x - current_x) * self.precision_damping
            smooth_y = current_y + (smooth_y - current_y) * self.precision_damping
        
        # Store current normalized position
        self.current_norm_pos = (smooth_x, smooth_y)
        
        # Convert to screen coordinates
        # FIXED: Map the entire camera view (0..1) to entire screen (0..screen_size)
        # This ensures hand at edge of camera reaches edge of screen
        screen_x, screen_y = self.normalized_to_screen(smooth_x, smooth_y)
        
        # Move mouse
        try:
            self.mouse.position = (screen_x, screen_y)
        except Exception as e:
            print(f"⚠ Error moving cursor: {e}")
    
    @exposed_action
    def click(self, button: str = 'left', double: bool = False):
        """
        Perform mouse click.
        
        Args:
            button: 'left', 'right', or 'middle'
            double: If True, perform double click
        """
        if self.paused:
            return
        
        btn = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }.get(button, Button.left)
        
        try:
            if double:
                self.mouse.click(btn, 2)
            else:
                self.mouse.click(btn, 1)
        except Exception as e:
            print(f"⚠ Error clicking: {e}")
    
    @exposed_action
    def execute_key_combo(self, key_string: str):
        """
        Execute a key combination from a string description.
        
        Format example: "ctrl+shift+p", "alt+tab", "a", "enter"
        
        Args:
            key_string: String describing the key combo
        """
        if self.paused or not PYNPUT_AVAILABLE:
            return

        # Normalize string
        parts = [p.strip().lower() for p in key_string.split('+')]
        if not parts:
            return
            
        try:
            keys_to_press = []
            
            # Helper to get key object
            def get_key(k_str):
                # Check for special keys in pynput.keyboard.Key
                if hasattr(Key, k_str):
                    return getattr(Key, k_str)
                # Check for single char
                if len(k_str) == 1:
                    return k_str
                # Handle aliases/special cases
                aliases = {
                    'ctrl': Key.ctrl,
                    'control': Key.ctrl,
                    'shift': Key.shift,
                    'alt': Key.alt,
                    'win': Key.cmd,
                    'cmd': Key.cmd,
                    'super': Key.cmd, 
                    'enter': Key.enter,
                    'return': Key.enter,
                    'esc': Key.esc,
                    'escape': Key.esc,
                    'space': Key.space,
                    'tab': Key.tab,
                    'backspace': Key.backspace,
                    'del': Key.delete,
                    'delete': Key.delete,
                    'up': Key.up,
                    'down': Key.down,
                    'left': Key.left,
                    'right': Key.right,
                    'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
                    'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
                    'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
                }
                return aliases.get(k_str)

            # Identify valid keys
            for part in parts:
                k = get_key(part)
                if k:
                    keys_to_press.append(k)
                else:
                    print(f"⚠ Unknown key in combo: {part}")
                    return

            # Press all in order
            for k in keys_to_press:
                self.keyboard.press(k)
            
            # Release all in reverse order
            for k in reversed(keys_to_press):
                self.keyboard.release(k)
                
        except Exception as e:
            print(f"⚠ Error executing key combo '{key_string}': {e}")
        """Start drag operation (press and hold)."""
        if self.paused or self.is_dragging:
            return
        
        try:
            self.mouse.press(Button.left)
            self.is_dragging = True
        except Exception as e:
            print(f"⚠ Error starting drag: {e}")
    
    def stop_drag(self):
        """Stop drag operation (release)."""
        if not self.is_dragging:
            return
        
        try:
            self.mouse.release(Button.left)
            self.is_dragging = False
        except Exception as e:
            print(f"⚠ Error stopping drag: {e}")
    
    @exposed_action
    def scroll(self, dx: int = 0, dy: int = 0, velocity_norm: float = 1.0):
        """
        Perform scroll action with velocity-modulated rate-limiting.
        
        Scrolling is triggered by swipe gestures:
        - swipe_up → scroll up (dy < 0)
        - swipe_down → scroll down (dy > 0)
        
        Args:
            dx: Horizontal scroll amount
            dy: Vertical scroll amount (positive = down, negative = up)
            velocity_norm: Normalized gesture velocity for modulating scroll rate
        """
        if self.paused:
            return
        
        try:
            if dx == 0 and dy == 0:
                return
            
            # Determine which swipe direction sensitivity to use
            # swipe_up triggers scroll up (dy < 0), swipe_down triggers scroll down (dy > 0)
            if dy != 0:
                gesture_name = 'swipe_down' if dy > 0 else 'swipe_up'
            else:
                gesture_name = 'swipe_right' if dx > 0 else 'swipe_left'
            
            # Use the appropriate swipe sensitivity for rate limiting
            sensitivity = self.velocity_sensitivity.get(gesture_name)
            if sensitivity and not sensitivity.try_act(velocity_norm):
                return
            
            self.mouse.scroll(dx, dy)
        except Exception as e:
            print(f"⚠ Error scrolling: {e}")
    
    @exposed_action
    def zoom(self, zoom_in: bool = True, velocity_norm: float = 1.0):
        """
        Perform system zoom (Ctrl + +/-) with velocity-modulated rate-limiting.
        
        Args:
            zoom_in: If True, zoom in. If False, zoom out.
            velocity_norm: Normalized gesture velocity for modulating zoom rate.
        """
        if self.paused:
            return

        try:
            # Use the VelocitySensitivity calculator for rate limiting
            if not self.velocity_sensitivity['zoom'].try_act(velocity_norm):
                return

            with self.keyboard.pressed(Key.ctrl):
                if zoom_in:
                    # Press Ctrl + + (Shift + =)
                    self.keyboard.press(Key.shift)
                    self.keyboard.press('=')  # Shift+= is +
                    self.keyboard.release('=')
                    self.keyboard.release(Key.shift)
                else:
                    self.keyboard.press('-')
                    self.keyboard.release('-')
        except Exception as e:
            print(f"⚠ Error zooming: {e}")
    
    def workspace_switch(self, direction: str):
        """
        Switch workspace (Ctrl+Alt+Arrow).
        
        Args:
            direction: 'left', 'right', 'up', or 'down'
        """
        if self.paused:
            return
        
        key_map = {
            'left': Key.left,
            'right': Key.right,
            'up': Key.up,
            'down': Key.down
        }
        
        arrow_key = key_map.get(direction)
        if not arrow_key:
            return
        
        try:
            with self.keyboard.pressed(Key.ctrl):
                with self.keyboard.pressed(Key.alt):
                    self.keyboard.press(arrow_key)
                    self.keyboard.release(arrow_key)
        except Exception as e:
            print(f"⚠ Error switching workspace: {e}")
    
    def get_velocity_sensitivity(self, gesture_name: str) -> Optional[VelocitySensitivity]:
        """
        Get the VelocitySensitivity calculator for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture (e.g., 'swipe_left', 'zoom', 'thumbs_up_moving_up')
        
        Returns:
            VelocitySensitivity instance or None if not found
        """
        return self.velocity_sensitivity.get(gesture_name)
    
    def get_base_sensitivity(self, gesture_name: str, default: float = 1.0) -> float:
        """
        Get the base sensitivity value for a gesture.
        
        This returns the raw sensitivity multiplier from the velocity sensitivity config,
        useful for scaling actions like scroll amount.
        
        Args:
            gesture_name: Name of the gesture (e.g., 'swipe_up', 'swipe_down')
            default: Default value if gesture not found
        
        Returns:
            Base sensitivity value (float)
        """
        sensitivity = self.velocity_sensitivity.get(gesture_name)
        if sensitivity:
            return sensitivity.config.base_sensitivity
        return default
    
    def perform_velocity_action(self, gesture_name: str, velocity_norm: float, action_callback) -> bool:
        """
        Perform an action with velocity-based rate limiting.
        
        This is the generic method for any velocity-dependent gesture.
        Uses the VelocitySensitivity calculator to determine if enough time
        has passed since the last action.
        
        Args:
            gesture_name: Name of the gesture for config lookup
            velocity_norm: Normalized gesture velocity
            action_callback: Function to call if action is allowed (no args)
        
        Returns:
            True if action was performed, False if rate-limited
        """
        if self.paused:
            return False
        
        sensitivity = self.velocity_sensitivity.get(gesture_name)
        if sensitivity is None:
            # No config for this gesture - just perform the action
            action_callback()
            return True
        
        if sensitivity.try_act(velocity_norm):
            action_callback()
            return True
        
        return False
    
    @exposed_action
    def swipe(self, direction: str, velocity_norm: float = 1.0) -> bool:
        """
        Perform swipe action with velocity-modulated rate-limiting.
        
        Swipe actions trigger workspace switches or navigation:
        - swipe_left/right: Switch workspace left/right
        - swipe_up/down: Can be configured for various actions
        
        Args:
            direction: 'left', 'right', 'up', or 'down'
            velocity_norm: Normalized gesture velocity
        
        Returns:
            True if action was performed, False if rate-limited
        """
        gesture_name = f'swipe_{direction}'
        return self.perform_velocity_action(
            gesture_name,
            velocity_norm,
            lambda: self.workspace_switch(direction)
        )
    
    @exposed_action
    def thumbs_action(self, gesture_name: str, velocity_norm: float = 1.0, action_callback=None) -> bool:
        """
        Perform thumbs gesture action with velocity-modulated rate-limiting.
        
        Thumbs gestures with movement (thumbs_up_moving_up, etc.) use velocity
        to control action rate. Static thumbs_up/thumbs_down don't use this.
        
        Args:
            gesture_name: Full gesture name (e.g., 'thumbs_up_moving_up')
            velocity_norm: Normalized gesture velocity
            action_callback: Optional custom action. If None, uses default volume/brightness.
        
        Returns:
            True if action was performed, False if rate-limited
        """
        if action_callback is None:
            # Default actions for thumbs gestures
            action_map = {
                'thumbs_up_moving_up': lambda: self._volume_change(+5),
                'thumbs_up_moving_down': lambda: self._volume_change(-5),
                'thumbs_down_moving_up': lambda: self._brightness_change(+5),
                'thumbs_down_moving_down': lambda: self._brightness_change(-5),
            }
            action_callback = action_map.get(gesture_name, lambda: None)
        
        return self.perform_velocity_action(gesture_name, velocity_norm, action_callback)
    
    def _volume_change(self, delta: int):
        """Change system volume using media keys or platform-specific methods."""
        try:
            # Use XF86 media keys - works on most Linux desktops
            if delta > 0:
                self.keyboard.press(Key.media_volume_up)
                self.keyboard.release(Key.media_volume_up)
            else:
                self.keyboard.press(Key.media_volume_down)
                self.keyboard.release(Key.media_volume_down)
        except Exception:
            # Fallback: try pactl on Linux
            try:
                import subprocess
                if delta > 0:
                    subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '+5%'], 
                                 capture_output=True, timeout=1)
                else:
                    subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '-5%'], 
                                 capture_output=True, timeout=1)
            except Exception:
                pass  # Volume control not available
    
    def _brightness_change(self, delta: int):
        """Change screen brightness using platform-specific methods.
        
        On Linux, tries (in order):
        1. brightnessctl - works on most systems
        2. xbacklight - legacy X11 method
        3. DBus - GNOME/KDE brightness interface
        """
        import subprocess
        import shutil
        
        # Method 1: brightnessctl (most common on modern Linux)
        if shutil.which('brightnessctl'):
            try:
                if delta > 0:
                    result = subprocess.run(['brightnessctl', 'set', '+5%'], 
                                           capture_output=True, timeout=1)
                else:
                    result = subprocess.run(['brightnessctl', 'set', '5%-'], 
                                           capture_output=True, timeout=1)
                if result.returncode == 0:
                    return  # Success
                else:
                    print(f"⚠ brightnessctl failed: {result.stderr.decode()}")
            except subprocess.TimeoutExpired:
                print("⚠ brightnessctl timed out")
        
        # Method 2: xbacklight (legacy X11)
        if shutil.which('xbacklight'):
            try:
                if delta > 0:
                    result = subprocess.run(['xbacklight', '-inc', '5'], 
                                           capture_output=True, timeout=1)
                else:
                    result = subprocess.run(['xbacklight', '-dec', '5'], 
                                           capture_output=True, timeout=1)
                if result.returncode == 0:
                    return  # Success
                else:
                    print(f"⚠ xbacklight failed: {result.stderr.decode()}")
            except subprocess.TimeoutExpired:
                print("⚠ xbacklight timed out")
        
        # Method 3: DBus (GNOME/KDE) - requires python-dbus
        try:
            import dbus
            bus = dbus.SessionBus()
            # Try GNOME
            try:
                brightness_proxy = bus.get_object('org.gnome.SettingsDaemon.Power',
                    '/org/gnome/SettingsDaemon/Power')
                brightness_iface = dbus.Interface(brightness_proxy,
                    'org.gnome.SettingsDaemon.Power.Screen')
                current = brightness_iface.GetPercentage()
                new_val = max(5, min(100, current + (5 if delta > 0 else -5)))
                brightness_iface.SetPercentage(new_val)
                return
            except dbus.DBusException as e:
                print(f"⚠ DBus brightness failed: {e}")
        except ImportError:
            pass
        
        # No brightness control method available
        print("⚠ No brightness control method available. Install brightnessctl: sudo apt install brightnessctl")
    
    def handle_pinch_gesture(self, pinch_detected: bool):
        """
        Handle pinch gesture for click/drag logic.
        
        Args:
            pinch_detected: True if pinch is currently detected
        """
        now = time.time()
        
        if pinch_detected:
            if self.pinch_start_time is None:
                # Pinch just started
                self.pinch_start_time = now
                
                # Check for double click
                if (self.click_count > 0 and 
                    now - self.last_click_time < self.double_click_timeout):
                    self.click(double=True)
                    self.click_count = 0
                else:
                    # Single click
                    self.click()
                    self.click_count = 1
                    self.last_click_time = now
        else:
            if self.pinch_start_time is not None:
                # Pinch just released
                pinch_duration = now - self.pinch_start_time
                
                # If held long enough, it was a drag
                if pinch_duration >= self.drag_hold_duration:
                    self.stop_drag()
                
                self.pinch_start_time = None
        
        # Check if we should start dragging
        if (self.pinch_start_time is not None and 
            not self.is_dragging and
            now - self.pinch_start_time >= self.drag_hold_duration):
            self.start_drag()


# Singleton instance
_controller_instance = None

def get_system_controller(config=None) -> SystemController:
    """Get or create singleton system controller instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = SystemController(config)
    return _controller_instance


if __name__ == "__main__":
    # Test system controller
    print("\n=== System Controller Test ===\n")
    
    if not PYNPUT_AVAILABLE:
        print("✗ pynput not installed. Run: pip install pynput")
        exit(1)
    
    controller = SystemController()
    
    print("\nMoving cursor to center...")
    controller.move_cursor(0.5, 0.5)
    time.sleep(0.5)
    
    print("Moving cursor in circle...")
    for angle in np.linspace(0, 2*np.pi, 20):
        x = 0.5 + 0.2 * np.cos(angle)
        y = 0.5 + 0.2 * np.sin(angle)
        controller.move_cursor(x, y)
        time.sleep(0.05)
    
    print("\n✓ System Controller test complete!")
    print("  (Your cursor should have moved to center and drawn a circle)")
