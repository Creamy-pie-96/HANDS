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
    update_state = pyqtSignal(str, str)  # gesture_name, state (color)
    update_frame = pyqtSignal(object)    # frame (numpy array)

class CameraWindow(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("HANDS Camera View")
        width = self.config.get('display', 'window_width', default=640)
        height = self.config.get('display', 'window_height', default=480)
        self.resize(width, height)
        
        self.image_label = QLabel(self)
        self.image_label.resize(width, height)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Style
        self.setStyleSheet("background-color: black;")

    def update_frame(self, frame):
        if frame is None:
            return
            
        # Convert CV2 frame (BGR) to QImage (RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale to window size
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
    def resizeEvent(self, event):
        self.image_label.resize(self.size())
        super().resizeEvent(event)


class StatusIndicator(QWidget):
    """
    Floating status indicator with gesture sticker support.
    
    Features:
    - Displays gesture stickers from config paths
    - Config hot-reload (checks file modification time)
    - Pause/exit control via config fields
    """
    
    def __init__(self, config, status_queue=None):
        super().__init__()
        self.config = config
        self.status_queue = status_queue
        self.config_path = config._config_path
        
        # Track config file modification time for hot-reload
        try:
            self.last_config_mtime = os.path.getmtime(self.config_path)
        except Exception:
            self.last_config_mtime = 0
        
        # Current state
        self.current_color = QColor(255, 0, 0)  # Default Red (paused)
        self.current_text = "⏸"
        self.current_gesture = None  # Active gesture name
        
        # Sticker cache: {gesture_name: QPixmap}
        self.sticker_cache = {}
        self.stickers_enabled = False
        
        # Color map: {color_name: QColor}
        self.color_map = {}
        
        # Load stickers and colors from config
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
        
        # Resolve base path (relative to config file or absolute)
        if stickers_base:
            if not os.path.isabs(stickers_base):
                config_dir = Path(self.config_path).parent
                stickers_base = str(config_dir / stickers_base)
        else:
            stickers_base = str(Path(self.config_path).parent / 'stickers')
        
        self.sticker_cache.clear()
        loaded_count = 0
        
        for gesture_name, filename_entry in sticker_paths.items():
            # Handle [value, description] format from config
            if isinstance(filename_entry, list):
                filename = filename_entry[0] if filename_entry else None
            else:
                filename = filename_entry
            
            if not filename:
                continue
            
            # Construct full path
            if os.path.isabs(filename):
                full_path = filename
            else:
                full_path = os.path.join(stickers_base, filename)
            
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    self.sticker_cache[gesture_name] = pixmap
                    loaded_count += 1
        
        self.stickers_enabled = loaded_count > 0
        if loaded_count > 0:
            print(f"✓ Loaded {loaded_count} gesture stickers")
        
    def initUI(self):
        """Initialize the UI with window flags and positioning."""
        # Window flags for always on top, frameless, and transparent background
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Click-through support
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Size and Position
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        self.update_position()
        
        # Opacity
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
    def update_position(self):
        """Update widget position based on config settings."""
        primary_screen = QApplication.primaryScreen()
        if primary_screen is None:
            # Fallback to default screen dimensions
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
        """Draw the status indicator (sticker or colored circle with emoji)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Try to draw sticker if available for current gesture
        sticker_drawn = False
        if self.stickers_enabled and self.current_gesture and self.current_gesture in self.sticker_cache:
            pixmap = self.sticker_cache[self.current_gesture]
            if not pixmap.isNull():
                # Scale sticker to fit widget size
                scaled = pixmap.scaled(
                    self.width(), self.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                # Center the scaled pixmap
                x_offset = (self.width() - scaled.width()) // 2
                y_offset = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled)
                sticker_drawn = True
        
        if not sticker_drawn:
            # Draw colored circle as fallback
            painter.setBrush(QBrush(self.current_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, self.width(), self.height())
            
            # Draw text/emoji
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Segoe UI Emoji", int(self.width() * 0.5))
            font.setStyleHint(QFont.StyleHint.SansSerif)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.current_text)

    def _load_colors(self):
        """Load indicator colors from config."""
        colors_cfg = self.config.get('display', 'status_indicator', 'colors', default={})
        
        # Default colors
        defaults = {
            'red': [255, 50, 50],
            'yellow': [255, 200, 0],
            'blue': [50, 150, 255]
        }
        
        self.color_map = {}
        for color_name, default_rgb in defaults.items():
            rgb = colors_cfg.get(color_name, default_rgb)
            # Handle [value, description] format
            if isinstance(rgb, list) and len(rgb) >= 3:
                if isinstance(rgb[0], list):
                    # Nested list format: [[r,g,b], "description"]
                    rgb = rgb[0]
                self.color_map[color_name] = QColor(rgb[0], rgb[1], rgb[2])
            else:
                self.color_map[color_name] = QColor(*default_rgb)
    
    def update_status(self, color_name: str, text: str):
        """
        Update the indicator state.
        
        Args:
            color_name: 'red', 'yellow', 'blue' for state color
            text: Gesture name (for sticker lookup) or special state
        """
        self.current_color = self.color_map.get(color_name, QColor(100, 100, 100))
        
        # Map gesture name to emoji for fallback display
        emoji_map = {
            'pointing': '👆',
            'pinch': '🤏',
            'zoom': '🤌',
            'swipe': '👋',
            'open_hand': '✋',
            'thumbs_up': '👍',
            'thumbs_down': '👎',
            'thumbs_up_moving_up': '👍↑',
            'thumbs_up_moving_down': '👍↓',
            'thumbs_down_moving_up': '👎↑',
            'thumbs_down_moving_down': '👎↓',
            'precision_cursor': '🎯',
            'pan': '↔️',
            'paused': '⏸',
            'dry_run': 'DRY',
            '': ''
        }
        
        # Handle exit countdown (e.g., "exit_3")
        if text.startswith('exit_'):
            countdown = text.split('_')[1] if '_' in text else ''
            self.current_text = f"🚪{countdown}"
            self.current_gesture = 'thumbs_down'  # Use thumbs_down sticker
        else:
            self.current_text = emoji_map.get(text, text[:2].upper() if text else '')
            self.current_gesture = text if text in emoji_map else None
        
        self.update()  # Trigger repaint
    
    def reload_config(self):
        """Reload configuration and refresh stickers and colors."""
        self.config.reload()
        self._load_stickers()
        self._load_colors()
        
        # Update visual settings
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
        self.update_position()
        self.update()
    
    def check_config_changed(self) -> bool:
        """Check if config file has been modified."""
        try:
            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime != self.last_config_mtime:
                self.last_config_mtime = current_mtime
                return True
        except Exception:
            pass
        return False


def run_gui(config, status_queue, frame_queue=None):
    """
    Run the status indicator GUI.
    
    Args:
        config: Config instance
        status_queue: Queue for receiving status updates from main app
        frame_queue: Optional queue for receiving camera frames
    """
    app = QApplication(sys.argv)
    
    # Status Indicator
    indicator = StatusIndicator(config)
    indicator.show()
    
    # Camera Window (Optional)
    camera_window = None
    if frame_queue is not None:
        camera_window = CameraWindow(config)
        camera_window.show()
    
    signals = StatusSignals()
    signals.update_state.connect(indicator.update_status)
    if camera_window:
        signals.update_frame.connect(camera_window.update_frame)
    
    # Timer to check queues
    timer = QTimer()
    def check_queues():
        # Check Status Queue
        try:
            while True:
                state, text = status_queue.get_nowait()
                signals.update_state.emit(state, text)
        except queue.Empty:
            pass
            
        # Check Frame Queue
        if frame_queue:
            try:
                # Get the latest frame, discard older ones to reduce lag
                frame = None
                while True:
                    frame = frame_queue.get_nowait()
                
            except queue.Empty:
                # If we got at least one frame, emit it
                if frame is not None:
                    signals.update_frame.emit(frame)
            except Exception:
                pass
            else:
                # If loop finished without exception (rare with get_nowait), emit last frame
                if frame is not None:
                    signals.update_frame.emit(frame)

    timer.timeout.connect(check_queues)
    timer.start(30)  # Check every 30ms (~33 FPS)
    
    sys.exit(app.exec())