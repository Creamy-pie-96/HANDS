"""
System Controller for HANDS

Interfaces between gesture detection and actual system control (mouse/keyboard).
Uses pynput for reliable cross-platform input control.
"""

import time
import threading
from typing import Tuple, Optional
from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠ pynput not available. Install with: pip install pynput")

try:
    import screeninfo
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("⚠ screeninfo not available. Install with: pip install screeninfo")

from source_code.utils.math_utils import EWMA


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
            from source_code.config.config_manager import get_system_control
            self.cursor_smoothing = get_system_control('cursor', 'smoothing_factor', 0.3)
            self.cursor_speed = get_system_control('cursor', 'speed_multiplier', 1.5)
            self.precision_damping = get_system_control('cursor', 'precision_damping', 0.3)
            self.dead_zone = get_system_control('cursor', 'dead_zone', 0.02)
            self.magnetic_radius = get_system_control('cursor', 'magnetic_radius', 0.05)
            self.magnetic_factor = get_system_control('cursor', 'magnetic_factor', 0.4)
            self.screen_bounds_padding = get_system_control('cursor', 'screen_bounds_padding', 10)
            self.fallback_width = get_system_control('cursor', 'fallback_screen_width', 1920)
            self.fallback_height = get_system_control('cursor', 'fallback_screen_height', 1080)
            self.scroll_sensitivity = get_system_control('scroll', 'sensitivity', 30)
            self.zoom_sensitivity = get_system_control('zoom', 'sensitivity', 5)
            self.double_click_timeout = get_system_control('click', 'double_click_timeout', 0.5)
            self.drag_hold_duration = get_system_control('click', 'drag_hold_duration', 1.0)
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
            self.scroll_sensitivity = 30
            self.zoom_sensitivity = 5
            self.double_click_timeout = 0.5
            self.drag_hold_duration = 1.0
        
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
        # Last time a zoom keypress was issued (rate-limiting)
        self._last_zoom_time = 0.0
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
    
    def start_drag(self):
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
    
    def scroll(self, dx: int = 0, dy: int = 0):
        """
        Perform scroll action.
        
        Args:
            dx: Horizontal scroll amount
            dy: Vertical scroll amount (positive = down, negative = up)
        """
        if self.paused:
            return
        
        try:
            if dx != 0 or dy != 0:
                self.mouse.scroll(dx, dy)
        except Exception as e:
            print(f"⚠ Error scrolling: {e}")
    
    def zoom(self, zoom_in: bool = True):
        """
        Perform system zoom (Ctrl + +/-).
        
        Args:
            zoom_in: If True, zoom in. If False, zoom out.
        """
        if self.paused:
            return

        # Rate-limit zoom keypresses to avoid extremely fast repeated zooms
        # The configured `zoom_sensitivity` scales the repeat rate: higher sensitivity -> more frequent
        # Default behavior: one zoom keypress every 0.5s when sensitivity==1
        try:
            now = time.time()
            sensitivity = float(getattr(self, 'zoom_sensitivity', 1.0) or 1.0)
            base_delay = 0.5
            # Allow fractional sensitivity values <1.0 to reduce zoom frequency (larger delay).
            # Clamp sensitivity to a small positive floor to avoid division by zero.
            min_delay = base_delay / max(0.05, sensitivity)

            if now - self._last_zoom_time < min_delay:
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

            self._last_zoom_time = now
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
