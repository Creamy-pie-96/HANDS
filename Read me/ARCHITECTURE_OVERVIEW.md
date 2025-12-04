# HANDS Project Architecture

This document provides a complete overview of the HANDS codebase structure and how all components work together.

---

## Project Structure

```
HANDS/
├── app/                          # Launch scripts
│   ├── start_hands.sh            # Run the main application
│   └── run_config.sh             # Run the config editor GUI
│
├── source_code/                  # Main Python package
│   ├── __init__.py
│   │
│   ├── app/                      # Application entry point
│   │   ├── __init__.py
│   │   └── hands_app.py          # Main HANDSApp class
│   │
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── config.json           # All settings [value, description]
│   │   ├── config_manager.py     # Config loading & accessors
│   │   └── config_gui.py         # PyQt6 config editor
│   │
│   ├── detectors/                # Gesture detection logic
│   │   ├── __init__.py
│   │   ├── gesture_detectors.py  # Individual gesture detectors
│   │   └── bimanual_gestures.py  # Two-hand gestures + manager
│   │
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── system_controller.py  # OS interaction (mouse, keyboard)
│   │   ├── visual_feedback.py    # Debug overlays
│   │   └── math_utils.py         # Vector/geometry helpers
│   │
│   ├── gui/                      # GUI components
│   │   ├── status_indicator.py   # Floating gesture indicators
│   │   ├── generate_gesture_icons.py
│   │   └── stickers/             # Gesture sticker images
│   │
│   ├── models/                   # ML models
│   │   └── hand_landmarker.task  # MediaPipe hand model
│   │
│   └── scripts/                  # Utility scripts
│       ├── app_control.py
│       └── compact_config.py
│
├── Read me/                      # Documentation
│   ├── USER_GUIDE.md
│   ├── config_documentation.md
│   ├── ARCHITECTURE_*.md         # These files
│   └── ...
│
├── shareable/                    # Distributable scripts
│   ├── installation/
│   │   ├── install.sh
│   │   └── requirements.txt
│   └── scripts/
│       └── clone.sh
│
├── install.sh                    # Root install script
└── requirements.txt              # Python dependencies
```

---

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HANDS Application                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ Configuration │           │   Detection   │           │    System     │
│    Layer      │           │    Layer      │           │    Layer      │
├───────────────┤           ├───────────────┤           ├───────────────┤
│ config.json   │◄─────────►│ gesture_      │──────────►│ system_       │
│ config_manager│           │ detectors.py  │           │ controller.py │
│ config_gui.py │           │ bimanual_     │           │ pynput        │
│               │           │ gestures.py   │           │               │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│    Display    │           │   Tracking    │           │   Hardware    │
│    Layer      │           │    Layer      │           │    Layer      │
├───────────────┤           ├───────────────┤           ├───────────────┤
│ status_       │           │ MediaPipe     │           │ Webcam (cv2)  │
│ indicator.py  │           │ Hand          │           │ Mouse         │
│ visual_       │           │ Landmarker    │           │ Keyboard      │
│ feedback.py   │           │ Tasks API     │           │ Display       │
│ PyQt6         │           │               │           │               │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## Data Flow

### 1. Frame Capture → Landmark Detection

```python
# In hands_app.py
cap = cv2.VideoCapture(camera_idx)
ret, frame = cap.read()

# MediaPipe Tasks API (GPU-accelerated)
result = hand_landmarker.detect(mp_image)
for hand in result.hand_landmarks:
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
```

### 2. Landmarks → Gesture Detection

```python
# In bimanual_gestures.py
class ComprehensiveGestureManager:
    def process(self, landmarks_left, landmarks_right):
        all_gestures = {
            'left': {},
            'right': {},
            'bimanual': {}
        }

        # Individual hand gestures
        if landmarks_right:
            all_gestures['right'] = self._detect_single_hand(landmarks_right)

        # Two-hand gestures
        if landmarks_left and landmarks_right:
            all_gestures['bimanual'] = self._detect_bimanual(
                landmarks_left, landmarks_right
            )

        return all_gestures
```

### 3. Gestures → System Actions

```python
# In hands_app.py
def process_gestures(self, all_gestures):
    # Check if gesture is enabled
    if 'pointing' in right_gestures and is_gesture_enabled('pointing'):
        cursor_pos = right_gestures['pointing'].metadata['tip_position']
        self.system_ctrl.move_cursor(*cursor_pos)

    if 'zoom_in' in right_gestures and is_gesture_enabled('zoom_in'):
        velocity = right_gestures['zoom_in'].metadata['ewma_velocity']
        self.system_ctrl.zoom(zoom_in=True, velocity_norm=velocity)
```

### 4. State → Visual Feedback

```python
# Status indicator update
hands_data = {
    'left': {
        'detected': True,
        'state': 'blue',
        'gesture': 'open_hand',
        'disabled': False
    },
    'right': {
        'detected': True,
        'state': 'blue',
        'gesture': 'pointing',
        'disabled': not is_gesture_enabled('pointing')
    }
}
status_queue.put(hands_data)
```

---

## Key Design Patterns

### 1. Singleton Configuration

```python
# config_manager.py
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Global access
config = Config()
value = config.get('gesture_thresholds', 'pinch', 'threshold_rel')
```

### 2. Dataclass for Config Structures

```python
# system_controller.py
@dataclass
class VelocitySensitivityConfig:
    base_sensitivity: float = 1.0
    speed_neutral: float = 1.0
    speed_factor: float = 0.2
    base_delay: float = 0.5
```

### 3. Named Tuple for Gesture Results

```python
# gesture_detectors.py
from collections import namedtuple

GestureResult = namedtuple('GestureResult', ['detected', 'confidence', 'metadata'])

# Usage
result = GestureResult(
    detected=True,
    confidence=0.85,
    metadata={'direction': 'up', 'velocity': 1.2}
)
```

### 4. Queue-Based Thread Communication

```python
# Thread-safe communication between detection and GUI
import queue
import threading

status_queue = queue.Queue(maxsize=10)

# Producer (detection thread)
status_queue.put(hands_data)

# Consumer (GUI thread via QTimer)
def process_queue():
    try:
        data = status_queue.get_nowait()
        update_indicators(data)
    except queue.Empty:
        pass
```

### 5. EWMA Smoothing

Used throughout for noise reduction:

```python
class EWMAFilter:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
```

---

## Gesture Detection Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw        │    │  Extended   │    │  Gesture    │    │  Gesture    │
│  Landmarks  │───►│  Fingers    │───►│  Detectors  │───►│  Results    │
│  (21 points)│    │  Analysis   │    │  (per type) │    │  (dict)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Velocity   │
                   │  Tracking   │
                   │  (EWMA)     │
                   └─────────────┘
```

### Detector Types

| Detector         | Input                 | Output                                 |
| ---------------- | --------------------- | -------------------------------------- |
| PinchDetector    | landmarks, thresholds | `{detected, distance, hold_count}`     |
| PointingDetector | landmarks, thresholds | `{detected, tip_position, speed}`      |
| SwipeDetector    | velocity_history      | `{detected, direction, ewma_velocity}` |
| ZoomDetector     | spread_history        | `{detected, direction, ewma_velocity}` |
| ThumbsDetector   | landmarks, velocity   | `{detected, state, movement}`          |

---

## Configuration System

### Value + Description Format

```json
{
  "pinch": {
    "threshold_rel": [0.2, "Maximum thumb-index distance for pinch"],
    "hold_frames": [3, "Frames pinch must be held"]
  }
}
```

### Accessor Functions

```python
# Get just the value
value = config.get('pinch', 'threshold_rel')  # Returns 0.2

# Get value + description
value, desc = config.get_with_description('pinch', 'threshold_rel')
# Returns (0.2, "Maximum thumb-index distance for pinch")

# Convenience functions
threshold = get_gesture_threshold('pinch', 'threshold_rel')
sensitivity = get_system_control('zoom', 'sensitivity')
enabled = is_gesture_enabled('pointing')  # Returns bool
```

---

## Threading Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Main Thread                                    │
│  (PyQt6 Event Loop - GUI)                                               │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ DualHand    │    │ Camera      │    │ Config      │                 │
│  │ Indicator   │    │ Window      │    │ GUI         │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│         ▲                  ▲                                            │
│         │                  │                                            │
│    status_queue       frame_queue                                       │
│         │                  │                                            │
└─────────│──────────────────│────────────────────────────────────────────┘
          │                  │
┌─────────│──────────────────│────────────────────────────────────────────┐
│         ▼                  ▼                                            │
│                    Worker Thread                                         │
│  (Detection Loop)                                                       │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Camera      │───►│ MediaPipe   │───►│ Gesture     │                 │
│  │ Capture     │    │ Detection   │    │ Processing  │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                               │                         │
│                                               ▼                         │
│                                        ┌─────────────┐                 │
│                                        │ System      │                 │
│                                        │ Controller  │                 │
│                                        └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Points

### Adding a New Gesture

1. **Detector** (`gesture_detectors.py`):

```python
class NewGestureDetector:
    def detect(self, landmarks, metadata) -> GestureResult:
        # Detection logic
        return GestureResult(detected, confidence, metadata)
```

2. **Config** (`config.json`):

```json
"gesture_thresholds": {
  "new_gesture": {
    "threshold": [0.5, "Detection threshold"]
  }
},
"gestures_enabled": {
  "new_gesture": [true, "Enable new gesture"]
}
```

3. **Manager** (`bimanual_gestures.py`):

```python
self.new_detector = NewGestureDetector()
# Add to detection loop
```

4. **Action** (`hands_app.py`):

```python
if 'new_gesture' in gestures and is_gesture_enabled('new_gesture'):
    self.system_ctrl.new_action(...)
```

5. **Sticker** (`gui/stickers/`):
   Add `new_gesture.png` and reference in config.

### Adding a New System Action

1. **Controller** (`system_controller.py`):

```python
def new_action(self, param, velocity_norm=1.0):
    if not self.velocity_sensitivity['new_gesture'].should_perform_action(velocity_norm):
        return False
    # Perform action
    return True
```

2. **Config**:

```json
"system_control": {
  "new_gesture": {
    "sensitivity": [1.0, "Action sensitivity"],
    "base_delay": [0.5, "Delay between actions"]
  }
}
```

---

## Performance Optimization

| Component           | Optimization                      |
| ------------------- | --------------------------------- |
| MediaPipe           | GPU delegate via Tasks API        |
| Smoothing           | EWMA (O(1) per update)            |
| Gesture history     | Fixed-size deque (bounded memory) |
| Queue communication | Non-blocking put/get              |
| Overlay rendering   | Conditional (only if enabled)     |
| Config reload       | Every N frames (not every frame)  |

---

## Related Files

- `source_code/app/hands_app.py` - Main application
- `source_code/detectors/bimanual_gestures.py` - Gesture manager
- `source_code/utils/system_controller.py` - OS actions
- `source_code/config/config_manager.py` - Configuration
- `source_code/gui/status_indicator.py` - Visual feedback
