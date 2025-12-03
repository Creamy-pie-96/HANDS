import sys
import threading
import queue
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont

class StatusSignals(QObject):
    update_state = pyqtSignal(str, str)  # state (color), text (emoji/message)

class StatusIndicator(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.initUI()
        self.current_color = QColor(255, 0, 0)  # Default Red
        self.current_text = "⏸"
        
    def initUI(self):
        # Window flags for always on top, frameless, and transparent background
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Click-through support
        # Note: WA_TransparentForMouseEvents is the correct attribute in Qt 5/6 for passing clicks
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Size and Position
        size = self.config.get('display', 'status_indicator', 'size', default=64)
        self.setFixedSize(size, size)
        
        self.update_position()
        
        # Opacity
        opacity = self.config.get('display', 'status_indicator', 'opacity', default=0.8)
        self.setWindowOpacity(opacity)
        
    def update_position(self):
        screen = QApplication.primaryScreen().geometry()
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
        
        # Draw circle
        painter.setBrush(QBrush(self.current_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())
        
        # Draw text/emoji
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI Emoji", int(self.width() * 0.5))
        # Fallback fonts if Segoe UI Emoji is not available
        font.setStyleHint(QFont.StyleHint.SansSerif)
        painter.setFont(font)
        
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.current_text)

    def update_status(self, color_name, text):
        color_map = {
            'red': QColor(255, 50, 50),      # Error/Paused
            'yellow': QColor(255, 200, 0),   # Idle/Searching
            'green': QColor(50, 255, 50),    # Active/Tracking (Legacy)
            'blue': QColor(50, 150, 255)     # Hand Detected / Gesture Active
        }
        self.current_color = color_map.get(color_name, QColor(100, 100, 100))
        self.current_text = text
        self.update()  # Trigger repaint

def run_gui(config, status_queue):
    app = QApplication(sys.argv)
    
    indicator = StatusIndicator(config)
    indicator.show()
    
    signals = StatusSignals()
    signals.update_state.connect(indicator.update_status)
    
    # Timer to check queue
    timer = QTimer()
    def check_queue():
        try:
            while True:
                state, text = status_queue.get_nowait()
                signals.update_state.emit(state, text)
        except queue.Empty:
            pass
            
    timer.timeout.connect(check_queue)
    timer.start(50)  # Check every 50ms
    
    sys.exit(app.exec())
