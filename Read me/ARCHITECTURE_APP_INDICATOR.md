# Application & Status Indicator Architecture

This document explains how the HANDS application and its status indicator system work together.

---

## Overview

The HANDS application consists of two main components running in parallel:

1. **Main App Thread** (`hands_app.py`) - Hand detection and gesture processing
2. **GUI Thread** (`status_indicator.py`) - Visual feedback via PyQt6 widgets

These communicate through thread-safe queues, allowing the CPU-intensive detection loop to run independently from the UI rendering.

---

## Application Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Main App Thread                          │
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌────────────────────┐     │
│  │ Camera  │ -> │ MediaPipe │ -> │ Gesture Processing │     │
│  │ Capture │    │ Detection │    │                    │     │
│  └─────────┘    └──────────┘    └────────────────────┘     │
│                                          │                  │
│                                          v                  │
│                              ┌────────────────────┐        │
│                              │ System Controller  │        │
│                              │ (mouse, keyboard)  │        │
│                              └────────────────────┘        │
│                                          │                  │
└──────────────────────────────────────────│──────────────────┘
                                           │
                          status_queue     │  frame_queue
                               │           │       │
                               v           v       v
┌─────────────────────────────────────────────────────────────┐
│                    GUI Thread (PyQt6)                       │
│                                                             │
│  ┌───────────────────┐    ┌─────────────────────────────┐  │
│  │  DualHandIndicator │    │      CameraWindow           │  │
│  │  ┌──────┐ ┌──────┐ │    │  (optional preview)         │  │
│  │  │ Left │ │Right │ │    │                             │  │
│  │  └──────┘ └──────┘ │    │                             │  │
│  └───────────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Classes

### `HANDSApp` (hands_app.py)

The main application class that orchestrates everything:

```python
class HANDSApp:
    def __init__(self, camera_idx, enable_system_control, status_queue):
        # Initialize camera, MediaPipe, gesture manager, system controller
        self.cap = cv2.VideoCapture(camera_idx)
        self.gesture_manager = ComprehensiveGestureManager()
        self.system_ctrl = SystemController()

    def run(self):
        while self.running:
            # 1. Capture frame
            ret, frame = self.cap.read()

            # 2. Detect hands with MediaPipe
            result = self.hand_detector.detect(frame)

            # 3. Process gestures
            all_gestures = self.gesture_manager.process(landmarks)

            # 4. Execute system actions (if gesture enabled)
            if is_gesture_enabled(gesture_name):
                self.process_gestures(all_gestures)

            # 5. Send status to GUI
            self.status_queue.put(hands_data)
```

### `DualHandIndicator` (status_indicator.py)

Manages two `HandIndicator` widgets (one per hand):

```python
class DualHandIndicator:
    def __init__(self, config, status_queue):
        self.right_indicator = HandIndicator('right', ...)
        self.left_indicator = HandIndicator('left', ...)

        # Timer polls the queue and updates indicators
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)

    def update_hands(self, hands_data):
        # hands_data = {
        #     'left': {'detected': bool, 'state': str, 'gesture': str, 'disabled': bool},
        #     'right': {'detected': bool, 'state': str, 'gesture': str, 'disabled': bool}
        # }
        self.right_indicator.update_state(
            state=hands_data['right']['state'],
            gesture=hands_data['right']['gesture'],
            disabled=hands_data['right']['disabled']
        )
```

### `HandIndicator` (status_indicator.py)

Individual floating indicator widget:

```python
class HandIndicator(QWidget):
    def __init__(self, hand_label, config, sticker_cache, color_map):
        self.current_gesture = None
        self.is_disabled = False

    def paintEvent(self, event):
        # Draw gesture sticker (or fallback emoji circle)
        if self.current_gesture in self.sticker_cache:
            painter.drawPixmap(...)  # Draw sticker
        else:
            painter.drawEllipse(...)  # Fallback circle with emoji

        # Draw disabled indicator (red dot)
        if self.is_disabled:
            painter.drawEllipse(dot_x, dot_y, dot_size, dot_size)
```

---

## Thread Communication

### Status Queue

The main app sends gesture status to the GUI:

```python
# In HANDSApp.run()
hands_data = {
    'left': {
        'detected': True,
        'state': 'blue',           # Indicator color
        'gesture': 'pointing',     # Gesture name for sticker
        'disabled': False          # Red dot if True
    },
    'right': {
        'detected': True,
        'state': 'red',
        'gesture': 'exit_3',       # Exit countdown
        'disabled': False
    }
}
self.status_queue.put(hands_data)
```

### Frame Queue (Optional)

For camera preview window:

```python
# In HANDSApp.run()
if show_camera:
    self.frame_queue.put(display_frame)
```

---

## Gesture Enable/Disable Flow

1. **Config Check**: `is_gesture_enabled(gesture_name)` reads from `gestures_enabled` config
2. **Action Gating**: If disabled, `process_gestures()` skips the system action
3. **Visual Feedback**: `hands_data['disabled']` is set True, triggering red dot in indicator

```python
# In process_gestures()
if 'pointing' in right_gestures and is_gesture_enabled('pointing'):
    self.system_ctrl.move_cursor(x, y)

# In building hands_data
hands_data['right']['disabled'] = not is_gesture_enabled(gesture_name)
```

---

## Click-Through Implementation

The status indicator must not capture mouse clicks:

```python
def _setup_click_through(self):
    # Qt attributes (partial on X11)
    self.setAttribute(Qt.WA_TransparentForMouseEvents)
    self.setAttribute(Qt.WA_X11DoNotAcceptFocus)

    # X11/XShape: Set empty input region
    xext.XShapeCombineRectangles(
        display, window_id,
        ShapeInput,     # Input region (not bounding)
        ShapeSet,       # Replace existing
        None, 0         # Empty rectangle list
    )
```

---

## Sticker Loading

Stickers are loaded from config and cached:

```python
def _load_stickers(self):
    base_path = config.get('display', 'status_indicator', 'stickers_base_path')
    stickers_cfg = config.get('display', 'status_indicator', 'stickers')

    for gesture, filename in stickers_cfg.items():
        path = Path(base_path) / filename
        if path.exists():
            self.sticker_cache[gesture] = QPixmap(str(path))
```

Fallback uses EMOJI_MAP when sticker not found:

```python
EMOJI_MAP = {
    'pointing': '👆', 'pinch': '🤏',
    'zoom_in': '🔍+', 'zoom_out': '🔍-',
    ...
}
```

---

## Key Design Decisions

1. **Separate Threads**: Detection (CPU-heavy) doesn't block UI
2. **Queue-Based Communication**: Thread-safe, decoupled components
3. **Per-Hand Indicators**: Each hand has independent state
4. **Sticker + Emoji Fallback**: Works without custom sticker files
5. **Configurable Disable**: Gesture detection continues but actions blocked
6. **Click-Through**: XShape ensures mouse events pass through

---

## Related Files

- `source_code/app/hands_app.py` - Main application
- `source_code/gui/status_indicator.py` - GUI widgets
- `source_code/config/config_manager.py` - Configuration loading
- `source_code/gui/stickers/` - Gesture sticker images
