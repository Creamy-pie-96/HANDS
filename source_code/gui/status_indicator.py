"""
Status Indicator GUI for HANDS

Displays floating status indicators for left and right hand gestures.
Supports gesture stickers loaded from config paths.

EXTENSIBLE DESIGN:
- To add new gestures: just add stickers to config.json under status_indicator.stickers
- The gesture detection sends gesture names, which are auto-matched to stickers
- If no sticker is found, an emoji fallback is shown (add to emoji_map for custom emojis)
"""

import sys
import os
import time
import queue
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRect
from PyQt6.QtGui import QColor, QPainter, QBrush, QFont, QPixmap, QImage


class StatusSignals(QObject):
    """Qt signals for thread-safe status updates."""
    update_hands = pyqtSignal(dict)      # {'left': {...}, 'right': {...}}
    update_frame = pyqtSignal(object)    # frame (numpy array)


class CameraWindow(QWidget):
    """Optional camera preview window."""
    
    def __init__(self, config, key_queue=None):
        super().__init__()
        self.config = config
        self.key_queue = key_queue
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("HANDS Camera View")
        width = self.config.get('display', 'window_width', default=640)
        height = self.config.get('display', 'window_height', default=480)
        self.resize(width, height)
        
        self.image_label = QLabel(self)
        self.image_label.resize(width, height)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")

    def update_frame(self, frame):
        if frame is None:
            return
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
    def resizeEvent(self, event):
        self.image_label.resize(self.size())
        super().resizeEvent(event)
    
    def keyPressEvent(self, event):
        if self.key_queue:
            try:
                self.key_queue.put_nowait(event.key())
            except:
                pass
        super().keyPressEvent(event)


# Shared emoji map for fallback display when no sticker is available
# EXTENSIBLE: Add new gesture -> emoji mappings here
EMOJI_MAP = {
    'pointing': '👆', 'pinch': '🤏',
    'zoom': '🤌', 'zoom_in': '🔍+', 'zoom_out': '🔍-',
    'swipe': '👋', 'swipe_up': '👋↑', 'swipe_down': '👋↓',
    'swipe_left': '👋←', 'swipe_right': '👋→',
    'open_hand': '✋',
    'thumbs_up': '👍', 'thumbs_down': '👎',
    'thumbs_up_moving_up': '👍↑', 'thumbs_up_moving_down': '👍↓',
    'thumbs_down_moving_up': '👎↑', 'thumbs_down_moving_down': '👎↓',
    'precision_cursor': '🎯', 'pan': '↔️',
    'paused': '⏸', 'dry_run': 'DRY',
    '': '', 'none': ''
}


class HandIndicator(QWidget):
    """
    Single hand status indicator widget.
    
    Shows a gesture sticker or fallback colored circle with emoji.
    Automatically hides when no hand is detected.
    Shows a small red dot when the displayed gesture is disabled.
    """
    
    def __init__(self, hand_label: str, config, sticker_cache: dict, color_map: dict, debug: bool = False):
        """
        Initialize a hand indicator.
        
        Args:
            hand_label: 'left' or 'right'
            config: Config instance
            sticker_cache: Shared dict of {gesture_name: QPixmap}
            color_map: Shared dict of {color_name: QColor}
            debug: Enable debug logging
        """
        super().__init__()
        self.hand_label = hand_label
        self.config = config
        self.sticker_cache = sticker_cache
        self.color_map = color_map
        self.debug = debug
        
        # Current state
        self.current_color = QColor(100, 100, 100)
        self.current_text = ""
        self.current_gesture = None
        self.is_detected = False
        self.is_disabled = False  # NEW: Track if current gesture is disabled
        
        # Track warned gestures (avoid repeated debug logs)
        self._warned_gestures = set()
        
        self.initUI()
    
    def initUI(self):
        """Initialize the UI."""
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Additional attributes for better click-through support
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_X11DoNotAcceptFocus, True)
        
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
        self.update_position()
        
        # Initially hidden (will be shown when hand is detected)
        self.hide()
        
        # Set up click-through after window is created
        self._setup_click_through()
    
    def update_position(self):
        """Update widget position based on config."""
        primary_screen = QApplication.primaryScreen()
        if primary_screen is None:
            screen_width = self.config.get('system_control', 'cursor', 'fallback_screen_width', default=1920)
            screen_height = self.config.get('system_control', 'cursor', 'fallback_screen_height', default=1080)
            screen = QRect(0, 0, screen_width, screen_height)
        else:
            screen = primary_screen.geometry()
        
        # Get per-hand position settings
        hand_config = self.config.get('display', 'status_indicator', f'{self.hand_label}_hand', default={})
        
        # Unwrap [value, description] format and ensure correct types
        pos_raw = hand_config.get('position', 'top-right')
        pos_setting = pos_raw[0] if isinstance(pos_raw, list) else pos_raw
        pos_setting = str(pos_setting) if pos_setting else 'top-right'
        
        mx_raw = hand_config.get('margin_x', 20)
        margin_x = mx_raw[0] if isinstance(mx_raw, list) else mx_raw
        margin_x = int(margin_x) if margin_x is not None else 20
        
        my_raw = hand_config.get('margin_y', 20)
        margin_y = my_raw[0] if isinstance(my_raw, list) else my_raw
        margin_y = int(margin_y) if margin_y is not None else 20
        
        x, y = 0, 0
        if 'right' in pos_setting:
            x = screen.width() - self.width() - margin_x
        else:
            x = margin_x
            
        if 'bottom' in pos_setting:
            y = screen.height() - self.height() - margin_y
        else:
            y = margin_y
            
        self.move(x, y)
        
        if self.debug:
            print(f"[{self.hand_label.upper()}] Position: ({x}, {y}) | config: pos={pos_setting}, mx={margin_x}, my={margin_y}")
    
    def paintEvent(self, event):
        """Draw the indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        sticker_drawn = False
        
        # Try to draw sticker if available
        if self.current_gesture and self.current_gesture in self.sticker_cache:
            pixmap = self.sticker_cache[self.current_gesture]
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.width(), self.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                x_offset = (self.width() - scaled.width()) // 2
                y_offset = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled)
                sticker_drawn = True
        
        if not sticker_drawn:
            # Fallback: colored circle with emoji
            painter.setBrush(QBrush(self.current_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, self.width(), self.height())
            
            if self.current_text:
                painter.setPen(QColor(255, 255, 255))
                font = QFont("Segoe UI Emoji", int(self.width() * 0.4))
                font.setStyleHint(QFont.StyleHint.SansSerif)
                painter.setFont(font)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.current_text)
        
        # Draw disabled indicator (small red dot in bottom-right corner)
        if self.is_disabled:
            dot_size = max(8, self.width() // 5)  # At least 8px, or 1/5 of indicator size
            dot_x = self.width() - dot_size - 2  # 2px padding from edge
            dot_y = self.height() - dot_size - 2
            
            # Draw red dot with black outline for visibility
            painter.setPen(QColor(0, 0, 0))
            painter.setBrush(QBrush(QColor(255, 50, 50)))  # Bright red
            painter.drawEllipse(dot_x, dot_y, dot_size, dot_size)
    
    def update_state(self, state: str, gesture: str, disabled: bool = False):
        """
        Update the indicator state.
        
        Args:
            state: 'red', 'yellow', 'blue', or 'hidden'
            gesture: Gesture name for sticker lookup
            disabled: If True, show a red dot to indicate gesture is disabled
        """
        if state == 'hidden':
            self.is_detected = False
            self.is_disabled = False
            if self.isVisible():
                self.hide()
                if self.debug:
                    print(f"[{self.hand_label.upper()}] Hidden")
            return
        
        # Show the indicator
        was_hidden = not self.isVisible()
        if not self.isVisible():
            self.show()
        
        self.is_detected = True
        self.is_disabled = disabled  # Track disabled state
        self.current_color = self.color_map.get(state, QColor(100, 100, 100))
        
        # Handle exit countdown (e.g., "exit_3")
        if gesture.startswith('exit_'):
            countdown = gesture.split('_')[1] if '_' in gesture else ''
            self.current_text = f"🚪{countdown}"
            self.current_gesture = 'thumbs_down'
        else:
            self.current_text = EMOJI_MAP.get(gesture, gesture[:2].upper() if gesture else '')
            self.current_gesture = gesture if gesture else None
        
        # Debug logging
        if self.debug:
            sticker_found = self.current_gesture in self.sticker_cache if self.current_gesture else False
            disabled_str = " [DISABLED]" if disabled else ""
            if was_hidden:
                print(f"[{self.hand_label.upper()}] Shown: state={state}, gesture={gesture}{disabled_str}, sticker={'YES' if sticker_found else 'NO'}")
            elif self.current_gesture and not sticker_found:
                if self.current_gesture not in self._warned_gestures:
                    self._warned_gestures.add(self.current_gesture)
                    print(f"[{self.hand_label.upper()}] No sticker for '{self.current_gesture}' (emoji fallback)")
        
        self.update()
    
    def _setup_click_through(self):
        """
        Platform-specific setup for true click-through on Linux/X11.
        
        Qt's WA_TransparentForMouseEvents doesn't work properly on X11,
        so we use the XShape extension to set an empty input region.
        This makes ALL mouse events pass through to windows below.
        """
        try:
            self._setup_click_through_ctypes()
        except Exception as e:
            if self.debug:
                print(f"[{self.hand_label.upper()}] Click-through setup failed: {e}")
    
    def _setup_click_through_ctypes(self):
        """Fallback click-through using ctypes to call X11 directly."""
        import ctypes
        import ctypes.util
        
        # Load X11 libraries
        x11 = ctypes.CDLL(ctypes.util.find_library('X11'))
        xext = ctypes.CDLL(ctypes.util.find_library('Xext'))
        
        # Get display and window
        display_name = ctypes.c_char_p(None)
        dpy = x11.XOpenDisplay(display_name)
        if not dpy:
            return
        
        window_id = int(self.winId())
        
        # XShapeCombineRectangles with no rectangles = empty input region
        # ShapeInput = 2, ShapeSet = 0
        xext.XShapeCombineRectangles(
            dpy,                    # Display
            window_id,              # Window
            2,                      # ShapeInput
            0, 0,                   # x, y offset
            None,                   # rectangles (NULL = empty)
            0,                      # count
            0,                      # ShapeSet
            0                       # Unsorted
        )
        
        x11.XFlush(dpy)
        x11.XCloseDisplay(dpy)
        
        if self.debug:
            print(f"[{self.hand_label.upper()}] X11 click-through enabled via ctypes")
    
    def showEvent(self, event):
        """Re-apply click-through when window becomes visible."""
        super().showEvent(event)
        # Delay slightly to ensure window is mapped
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self._setup_click_through)


class DualHandIndicator:
    """
    Manager for dual hand status indicators.
    
    Creates and manages two HandIndicator widgets - one for each hand.
    Each indicator only appears when that hand is detected.
    
    EXTENSIBLE: Stickers are shared between both hands. Just add stickers
    to config.json and they'll be available for both indicators.
    """
    
    def __init__(self, config):
        """Initialize dual hand indicators."""
        self.config = config
        self.config_path = config._config_path
        self.debug = config.get('display', 'status_indicator', 'debug', default=False)
        
        # Shared sticker cache and color map (loaded once, used by both indicators)
        self.sticker_cache = {}
        self.color_map = {}
        
        # Load resources
        self._load_stickers()
        self._load_colors()
        
        # Check which hands are enabled
        right_enabled = self._unwrap_config_value(
            config.get('display', 'status_indicator', 'right_hand', 'enabled', default=True), True
        )
        left_enabled = self._unwrap_config_value(
            config.get('display', 'status_indicator', 'left_hand', 'enabled', default=True), True
        )
        
        # Create indicators
        self.right_indicator = None
        self.left_indicator = None
        
        if right_enabled:
            self.right_indicator = HandIndicator('right', config, self.sticker_cache, self.color_map, self.debug)
        
        if left_enabled:
            self.left_indicator = HandIndicator('left', config, self.sticker_cache, self.color_map, self.debug)
        
        if self.debug:
            print(f"✓ DualHandIndicator initialized: right={right_enabled}, left={left_enabled}")
            print(f"  Stickers loaded: {len(self.sticker_cache)}")
    
    def _unwrap_config_value(self, value, default):
        """Unwrap [value, description] format from config."""
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value if value is not None else default
    
    def _load_stickers(self):
        """Load gesture sticker images from config paths."""
        sticker_paths = self.config.get('display', 'status_indicator', 'stickers', default={})
        stickers_base = self.config.get('display', 'status_indicator', 'stickers_base_path', default='')
        
        if not sticker_paths:
            print("⚠ No sticker paths configured")
            return
        
        # Resolve base path
        if stickers_base:
            if not os.path.isabs(stickers_base):
                config_dir = Path(self.config_path).parent
                stickers_base = str(config_dir / stickers_base)
        else:
            stickers_base = str(Path(self.config_path).parent / 'stickers')
        
        print(f"🔍 Loading stickers from: {stickers_base}")
        
        self.sticker_cache.clear()
        loaded_count = 0
        failed = []
        
        for gesture_name, filename_entry in sticker_paths.items():
            filename = filename_entry[0] if isinstance(filename_entry, list) else filename_entry
            if not filename:
                continue
            
            full_path = filename if os.path.isabs(filename) else os.path.join(stickers_base, filename)
            
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    self.sticker_cache[gesture_name] = pixmap
                    loaded_count += 1
                else:
                    failed.append(f"{gesture_name}: invalid image")
            else:
                failed.append(f"{gesture_name}: not found at {full_path}")
        
        if loaded_count > 0:
            print(f"✓ Loaded {loaded_count} gesture stickers: {list(self.sticker_cache.keys())}")
        else:
            print("⚠ No stickers loaded!")
        
        if failed and self.debug:
            print(f"⚠ Failed to load {len(failed)} stickers:")
            for f in failed[:5]:
                print(f"  - {f}")
    
    def _load_colors(self):
        """Load indicator colors from config."""
        colors_cfg = self.config.get('display', 'status_indicator', 'colors', default={})
        
        defaults = {
            'red': [255, 50, 50],
            'yellow': [255, 200, 0],
            'blue': [50, 150, 255]
        }
        
        self.color_map.clear()
        for color_name, default_rgb in defaults.items():
            rgb = colors_cfg.get(color_name, default_rgb)
            if isinstance(rgb, list) and len(rgb) >= 3:
                if isinstance(rgb[0], list):
                    rgb = rgb[0]
                self.color_map[color_name] = QColor(rgb[0], rgb[1], rgb[2])
            else:
                self.color_map[color_name] = QColor(*default_rgb)
    
    def update_hands(self, hands_data: dict):
        """
        Update both hand indicators.
        
        Args:
            hands_data: Dict with structure:
                {
                    'left': {'detected': bool, 'state': str, 'gesture': str, 'disabled': bool},
                    'right': {'detected': bool, 'state': str, 'gesture': str, 'disabled': bool}
                }
        """
        # Update right hand
        if self.right_indicator:
            right_data = hands_data.get('right', {})
            if right_data.get('detected', False):
                self.right_indicator.update_state(
                    right_data.get('state', 'blue'),
                    right_data.get('gesture', ''),
                    right_data.get('disabled', False)
                )
            else:
                self.right_indicator.update_state('hidden', '')
        
        # Update left hand (only show when detected)
        if self.left_indicator:
            left_data = hands_data.get('left', {})
            if left_data.get('detected', False):
                self.left_indicator.update_state(
                    left_data.get('state', 'blue'),
                    left_data.get('gesture', ''),
                    left_data.get('disabled', False)
                )
            else:
                self.left_indicator.update_state('hidden', '')
    
    def show(self):
        """Show indicators (they start hidden and appear when hands are detected)."""
        # Initially both are hidden, they show when hands are detected
        pass
    
    def hide(self):
        """Hide all indicators."""
        if self.right_indicator:
            self.right_indicator.hide()
        if self.left_indicator:
            self.left_indicator.hide()


# =============================================================================
# LEGACY COMPATIBILITY: Keep StatusIndicator class for backwards compatibility
# =============================================================================

class StatusIndicator(QWidget):
    """
    Legacy single-hand status indicator (kept for backwards compatibility).
    Use DualHandIndicator for new code.
    """
    
    def __init__(self, config, status_queue=None):
        super().__init__()
        self.config = config
        self.status_queue = status_queue
        self.config_path = config._config_path
        
        try:
            self.last_config_mtime = os.path.getmtime(self.config_path)
        except Exception:
            self.last_config_mtime = 0
        
        self.current_color = QColor(255, 0, 0)
        self.current_text = "⏸"
        self.current_gesture = None
        
        self.sticker_cache = {}
        self.stickers_enabled = False
        self._warned_gestures = set()
        self.color_map = {}
        
        self._load_stickers()
        self._load_colors()
        self.initUI()
        
    def _load_stickers(self):
        """Load gesture sticker images from config paths."""
        sticker_paths = self.config.get('display', 'status_indicator', 'stickers', default={})
        stickers_base = self.config.get('display', 'status_indicator', 'stickers_base_path', default='')
        
        if not sticker_paths:
            self.stickers_enabled = False
            return
        
        if stickers_base:
            if not os.path.isabs(stickers_base):
                config_dir = Path(self.config_path).parent
                stickers_base = str(config_dir / stickers_base)
        else:
            stickers_base = str(Path(self.config_path).parent / 'stickers')
        
        self.sticker_cache.clear()
        loaded_count = 0
        
        for gesture_name, filename_entry in sticker_paths.items():
            filename = filename_entry[0] if isinstance(filename_entry, list) else filename_entry
            if not filename:
                continue
            
            full_path = filename if os.path.isabs(filename) else os.path.join(stickers_base, filename)
            
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    self.sticker_cache[gesture_name] = pixmap
                    loaded_count += 1
        
        self.stickers_enabled = loaded_count > 0
        
    def initUI(self):
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        self.update_position()
        
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
    def update_position(self):
        primary_screen = QApplication.primaryScreen()
        if primary_screen is None:
            screen_width = self.config.get('system_control', 'cursor', 'fallback_screen_width', default=1920)
            screen_height = self.config.get('system_control', 'cursor', 'fallback_screen_height', default=1080)
            screen = QRect(0, 0, screen_width, screen_height)
        else:
            screen = primary_screen.geometry()
        
        pos_setting = self.config.get('display', 'status_indicator', 'position', default='top-right')
        margin_x = self.config.get('display', 'status_indicator', 'margin_x', default=20)
        margin_y = self.config.get('display', 'status_indicator', 'margin_y', default=20)
        
        x, y = 0, 0
        if 'right' in pos_setting:
            x = screen.width() - self.width() - margin_x
        else:
            x = margin_x
            
        if 'bottom' in pos_setting:
            y = screen.height() - self.height() - margin_y
        else:
            y = margin_y
            
        self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        sticker_drawn = False
        if self.stickers_enabled and self.current_gesture:
            if self.current_gesture in self.sticker_cache:
                pixmap = self.sticker_cache[self.current_gesture]
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        self.width(), self.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    x_offset = (self.width() - scaled.width()) // 2
                    y_offset = (self.height() - scaled.height()) // 2
                    painter.drawPixmap(x_offset, y_offset, scaled)
                    sticker_drawn = True
        
        if not sticker_drawn:
            painter.setBrush(QBrush(self.current_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, self.width(), self.height())
            
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Segoe UI Emoji", int(self.width() * 0.5))
            font.setStyleHint(QFont.StyleHint.SansSerif)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.current_text)

    def _load_colors(self):
        colors_cfg = self.config.get('display', 'status_indicator', 'colors', default={})
        
        defaults = {
            'red': [255, 50, 50],
            'yellow': [255, 200, 0],
            'blue': [50, 150, 255]
        }
        
        self.color_map = {}
        for color_name, default_rgb in defaults.items():
            rgb = colors_cfg.get(color_name, default_rgb)
            if isinstance(rgb, list) and len(rgb) >= 3:
                if isinstance(rgb[0], list):
                    rgb = rgb[0]
                self.color_map[color_name] = QColor(rgb[0], rgb[1], rgb[2])
            else:
                self.color_map[color_name] = QColor(*default_rgb)
    
    def update_status(self, color_name: str, text: str):
        self.current_color = self.color_map.get(color_name, QColor(100, 100, 100))
        
        if text.startswith('exit_'):
            countdown = text.split('_')[1] if '_' in text else ''
            self.current_text = f"🚪{countdown}"
            self.current_gesture = 'thumbs_down'
        else:
            self.current_text = EMOJI_MAP.get(text, text[:2].upper() if text else '')
            self.current_gesture = text if text else None
        
        self.update()
    
    def reload_config(self):
        self.config.reload()
        self._load_stickers()
        self._load_colors()
        
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
        self.update_position()
        self.update()
    
    def check_config_changed(self) -> bool:
        try:
            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime != self.last_config_mtime:
                self.last_config_mtime = current_mtime
                return True
        except Exception:
            pass
        return False


# =============================================================================
# RUN_GUI: Main entry point
# =============================================================================

def run_gui(config, status_queue, frame_queue=None, key_queue=None):
    """
    Run the status indicator GUI with dual hand support.
    
    Args:
        config: Config instance
        status_queue: Queue for receiving status updates from main app
                     
                     NEW FORMAT (preferred):
                     {'left': {'detected': bool, 'state': str, 'gesture': str},
                      'right': {'detected': bool, 'state': str, 'gesture': str}}
                     
                     LEGACY FORMAT (still supported):
                     (state, gesture_name) - interpreted as right hand only
                     
        frame_queue: Optional queue for receiving camera frames
        key_queue: Optional queue for sending keyboard events to main app
    """
    app = QApplication(sys.argv)
    
    # Prevent app from quitting when windows close
    app.setQuitOnLastWindowClosed(False)
    
    debug = config.get('display', 'status_indicator', 'debug', default=False)
    
    # Create dual hand indicator
    indicator = DualHandIndicator(config)
    indicator.show()
    
    # Camera Window (Optional)
    camera_window = None
    if frame_queue is not None:
        camera_window = CameraWindow(config, key_queue)
        camera_window.show()
    
    signals = StatusSignals()
    signals.update_hands.connect(indicator.update_hands)
    if camera_window:
        signals.update_frame.connect(camera_window.update_frame)
    
    # Timer to check queues
    timer = QTimer()
    
    def check_queues():
        # Check Status Queue
        try:
            while True:
                data = status_queue.get_nowait()
                
                # Check for shutdown sentinel
                if isinstance(data, tuple) and len(data) == 2:
                    state, text = data
                    if state == 'shutdown' and text == 'shutdown':
                        print("🛑 Shutdown signal received, closing GUI...")
                        try:
                            app_instance = QApplication.instance()
                            if app_instance:
                                app_instance.quit()
                        except Exception as e:
                            print(f"⚠ Error during quit: {e}")
                            sys.exit(0)
                        return
                    
                    # Legacy format: convert to new format for backwards compatibility
                    # Assume it's for right hand if legacy format
                    hands_data = {
                        'right': {'detected': True, 'state': state, 'gesture': text},
                        'left': {'detected': False, 'state': 'hidden', 'gesture': ''}
                    }
                    signals.update_hands.emit(hands_data)
                    
                    if debug:
                        print(f"[LEGACY] state={state}, gesture={text}")
                
                elif isinstance(data, dict):
                    # New format: {'left': {...}, 'right': {...}}
                    signals.update_hands.emit(data)
                    
                    if debug:
                        left_g = data.get('left', {}).get('gesture', '')
                        right_g = data.get('right', {}).get('gesture', '')
                        left_d = data.get('left', {}).get('detected', False)
                        right_d = data.get('right', {}).get('detected', False)
                        print(f"[UPDATE] L: {left_g if left_d else 'hidden'} | R: {right_g if right_d else 'hidden'}")
                    
        except queue.Empty:
            pass
        
        # Check Frame Queue
        if frame_queue:
            try:
                frame = None
                while True:
                    frame = frame_queue.get_nowait()
            except queue.Empty:
                if frame is not None:
                    signals.update_frame.emit(frame)
            except Exception:
                pass

    timer.timeout.connect(check_queues)
    timer.start(30)  # Check every 30ms (~33 FPS)
    
    exit_code = app.exec()
    print(f"📺 GUI closed with exit code: {exit_code}")
    sys.exit(exit_code)