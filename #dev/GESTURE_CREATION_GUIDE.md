# How to Create a New Gesture Class - Complete Guide

This guide teaches you how to create new gesture detector classes in HANDS. By the end, you'll understand the complete flow from detection to system action.

---

## Table of Contents

1. [Overview: How Gestures Work](#overview)
2. [Available Data & APIs](#available-data--apis)
3. [Creating a Gesture Class](#creating-a-gesture-class)
4. [Integrating with GestureManager](#integrating-with-gesturemanager)
5. [Reading from Config](#reading-from-config)
6. [Integrating with hands_app.py](#integrating-with-hands_apppy)
7. [Complete Template Example](#complete-template-example)
8. [Testing Your Gesture](#testing-your-gesture)

---

## Overview

### How Gestures Work in HANDS

```
1. Hand Detection
   ↓
2. Hand Metrics Computed (HandMetrics object with landmarks, finger positions, etc.)
   ↓
3. Gesture Detectors Process Metrics (PinchDetector, PointingDetector, etc.)
   ↓
4. Return GestureResult (detected: bool, gesture_name: str, confidence: float, metadata: dict)
   ↓
5. GestureManager Collects Results
   ↓
6. ActionDispatcher Maps Gesture → Action
   ↓
7. SystemController Executes Action (mouse move, click, etc.)
```

**Key Point:** You create the **Gesture Detectors** (step 3). Everything else is already built!

---

## Available Data & APIs

### 1. HandMetrics Object - What You Get in `detect()`

The `detect()` method receives a `HandMetrics` object. This contains everything about the hand:

```python
# Location & Structure
metrics.centroid              # Tuple (x, y) - hand center (0..1 normalized)
metrics.bbox                  # Tuple (xmin, ymin, xmax, ymax) - bounding box
metrics.landmarks_norm        # np.ndarray shape (21, 2) - all landmark positions

# Finger Information
metrics.tip_positions         # Dict: {'thumb': (x,y), 'index': (x,y), ...}
metrics.fingers_extended      # Dict: {'thumb': bool, 'index': bool, ...}

# Distances Between Fingers (normalized by hand size)
metrics.tip_distances         # Dict: {
                              #   'index_thumb': float,
                              #   'index_middle': float,
                              #   'thumb_middle': float,
                              #   ... (all finger combinations)
                              # }

# Movement Data
metrics.velocity              # Tuple (vx, vy) - hand movement velocity
metrics.timestamp             # float - when this frame was captured

# Hand Size
metrics.diag_rel              # float - hand diagonal relative to image
                              # Useful for normalizing distances
```

### 2. Available Utility Functions

Located in `source_code/utils/math_utils.py`:

```python
from source_code.utils.math_utils import (
    euclidean,           # euclidean(point1, point2) → distance
    EWMA,                # Exponential Weighted Moving Average filter
    landmarks_to_array,  # Convert landmarks to numpy array
    ClickDetector        # For detecting click patterns
)

# Example Usage:
dist = euclidean((0.5, 0.5), (0.6, 0.6))  # Returns float distance

# EWMA for smoothing:
ewma_filter = EWMA(alpha=0.3)  # Higher alpha = more responsive
smoothed_value = ewma_filter.update([raw_value])  # Returns array
```

### 3. Landmark Names Reference

```python
# In gesture_detectors.py, you have access to:
from source_code.detectors.gesture_detectors import LANDMARK_NAMES

LANDMARK_NAMES = {
    'WRIST': 0,
    # Thumb
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    # Index
    'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8,
    # Middle
    'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
    # Ring
    'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
    # Pinky
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20,
}

# Use with landmarks_norm:
# Example: Get thumb tip position
thumb_tip = metrics.landmarks_norm[LANDMARK_NAMES['THUMB_TIP']]  # Returns (x, y)
```

### 4. Data Structures You Work With

```python
from source_code.detectors.gesture_detectors import HandMetrics, GestureResult

# GestureResult - WHAT YOU RETURN
@dataclass
class GestureResult:
    detected: bool                      # True if gesture detected, False otherwise
    gesture_name: str                   # Name of the gesture (e.g., 'my_custom_gesture')
    confidence: float = 1.0             # 0.0 to 1.0 confidence level
    metadata: Dict = field(default_factory=dict)  # Extra data for visual feedback & dispatch
```

---

## Creating a Gesture Class

### Template Structure

```python
class MyGestureDetector:
    """
    Detects my custom gesture.
    Maintains per-hand state to support two-hand detection without interference.
    """

    def __init__(self, param1: float = 0.5, param2: int = 10, ewma_alpha: float = 0.3):
        """
        Initialize the gesture detector.

        Args:
            param1: Description of your parameter
            param2: Another parameter description
            ewma_alpha: Smoothing factor for velocity (optional, for movement-based gestures)
        """
        self.param1 = float(param1)
        self.param2 = int(param2)
        self.ewma_alpha = ewma_alpha

        # Per-hand state dictionary (IMPORTANT: supports left & right hands separately)
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }

    def _create_hand_state(self):
        """
        Create initial state for one hand.
        Return a dict with whatever you need to track over frames.
        """
        return {
            'counter': 0,           # Example: hold frame counter
            'last_detection': -999.0,  # Example: timestamp
            'ewma_filter': EWMA(alpha=self.ewma_alpha),  # Example: velocity filter
        }

    def _get_state(self, hand_label: str):
        """
        Get state for a specific hand. Creates if missing.
        """
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]

    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        """
        Detect the gesture from hand metrics.

        Args:
            metrics: HandMetrics object with all hand data
            hand_label: 'left' or 'right' hand

        Returns:
            GestureResult with detection info
        """
        # Get this hand's state
        state = self._get_state(hand_label)

        # Extract what you need from metrics
        thumb_tip = metrics.tip_positions['thumb']
        index_tip = metrics.tip_positions['index']
        thumb_index_dist = metrics.tip_distances.get('index_thumb', 0.0)
        hand_velocity = metrics.velocity
        timestamp = metrics.timestamp

        # Prepare metadata (always include for visual feedback)
        base_metadata = {
            'hand_label': hand_label,
            'param1_value': self.param1,
            'thumb_index_distance': thumb_index_dist,
            'hand_velocity': hand_velocity,
            'detection_reason': None,  # Will set later
        }

        # YOUR DETECTION LOGIC HERE
        # Example: Check if thumb and index are close
        if thumb_index_dist <= self.param1:
            state['counter'] += 1
            base_metadata['detection_reason'] = 'fingers_close'
            base_metadata['hold_count'] = state['counter']

            # Check if held long enough
            if state['counter'] >= self.param2:
                state['counter'] = 0  # Reset for next detection
                return GestureResult(
                    detected=True,
                    gesture_name='my_gesture',
                    confidence=1.0,
                    metadata=base_metadata
                )
        else:
            state['counter'] = 0
            base_metadata['detection_reason'] = 'fingers_too_far'

        # Not detected (return metadata anyway for visual feedback)
        return GestureResult(
            detected=False,
            gesture_name='my_gesture',
            confidence=0.0,
            metadata=base_metadata
        )
```

### Key Design Principles

1. **Per-Hand State**: Always maintain `_hand_state` dict with 'left' and 'right' keys
2. **Stateless Returns**: Each `detect()` call should be independent (use state for continuity)
3. **Rich Metadata**: Include debugging info in metadata - it's used for visual feedback
4. **Normalized Coordinates**: All positions are (0..1), use `metrics.diag_rel` to normalize distances
5. **Always Return GestureResult**: Even when NOT detected, return with metadata

---

## Integrating with GestureManager

### Step 1: Add to GestureManager.**init**()

Open `source_code/detectors/gesture_detectors.py`, find `class GestureManager`:

```python
class GestureManager:
    def __init__(self):
        # ... existing code ...

        # Add your detector initialization
        my_param1 = get_gesture_threshold('my_gesture', 'param1', default=0.5)
        my_param2 = get_gesture_threshold('my_gesture', 'param2', default=10)
        my_ewma = get_gesture_threshold('my_gesture', 'ewma_alpha', default=0.3)

        self.my_gesture = MyGestureDetector(
            param1=my_param1,
            param2=my_param2,
            ewma_alpha=my_ewma
        )

        # ... rest of code ...
```

### Step 2: Call in process_hand()

Find `GestureManager.process_hand()` method:

```python
def process_hand(self, landmarks, img_shape: Tuple[int, int, int], hand_label: str = 'right') -> Dict[str, GestureResult]:
    """Process a single hand and detect all gestures."""

    # ... existing gesture detections ...

    # Add your gesture detection
    my_gesture_result = self.my_gesture.detect(metrics, hand_label)

    # Collect results
    results = {
        'pinch': pinch_result,
        'pointing': pointing_result,
        # ... other gestures ...
        'my_gesture': my_gesture_result,  # Add yours
    }

    return results
```

---

## Reading from Config

### How to Load Parameters from config.json

```python
# In GestureManager.__init__():
from source_code.config.config_manager import get_gesture_threshold

# Read from config with fallback default
param_value = get_gesture_threshold('gesture_name', 'parameter_name', default=default_value)

# Example:
pinch_threshold = get_gesture_threshold('pinch', 'threshold_rel', default=0.055)
```

### What Goes in config.json

Add to `source_code/config/config.json` under the `"performance"` section:

```json
{
  "performance": {
    "gesture_thresholds": {
      "my_gesture": {
        "param1": [0.5, "Distance threshold"],
        "param2": [10, "Hold frames"],
        "ewma_alpha": [0.3, "Smoothing factor"]
      }
    }
  }
}
```

---

## Integrating with hands_app.py

### How Gestures Flow Through hands_app.py

```
ComprehensiveGestureManager.process_hands()
    ↓
    Returns: {'left': {...}, 'right': {...}, 'bimanual': {...}}
    ↓
HANDSApplication.process_gestures(all_gestures)
    ↓
    Extracts primary gesture names: left_name, right_name
    ↓
ActionDispatcher.dispatch(left_name, right_name, metadata)
```

### Using Your Gesture in hands_app.py

**Option 1: Map to System Control (Recommended)**

Add to `config.json` action_map:

```json
{
  "action_map": [
    {
      "left": "none",
      "right": "my_gesture",
      "type": "system_action",
      "action": "zoom_in"
    }
  ]
}
```

**Option 2: Custom Logic in process_gestures() (Advanced)**

```python
# In HANDSApplication.process_gestures()
def process_gestures(self, all_gestures):
    # ... existing code ...

    # Check for your custom gesture
    right_gestures = all_gestures.get('right', {})
    if 'my_gesture' in right_gestures:
        result = right_gestures['my_gesture']
        if result.detected:
            # Do something custom
            metadata = result.metadata
            print(f"My gesture detected with velocity: {metadata.get('hand_velocity')}")
            # Call system controller directly
            self.system_ctrl.zoom(zoom_in=True, velocity_norm=1.0)
```

---

## Complete Template Example

Here's a complete, ready-to-use template for a "Fist Clap" gesture:

```python
from source_code.utils.math_utils import euclidean, EWMA
from source_code.detectors.gesture_detectors import HandMetrics, GestureResult
import time

class FistClapDetector:
    """
    Detects fast upward hand movement (like clapping hands together).
    Uses velocity to ensure fast movement.
    """

    def __init__(self,
                 velocity_threshold: float = 1.0,  # Min speed to detect clap
                 hold_frames: int = 3,              # Frames to hold detection
                 cooldown_s: float = 0.8,           # Min time between claps
                 ewma_alpha: float = 0.3):          # Smoothing factor
        """Initialize clap detector."""
        self.velocity_threshold = float(velocity_threshold)
        self.hold_frames = int(hold_frames)
        self.cooldown_s = float(cooldown_s)
        self.ewma_alpha = ewma_alpha

        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }

    def _create_hand_state(self):
        """Create state for one hand."""
        return {
            'hold_count': 0,
            'last_clap_time': -999.0,
            'ewma_velocity': EWMA(alpha=self.ewma_alpha),
        }

    def _get_state(self, hand_label: str):
        """Get or create state for hand."""
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]

    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        """Detect clap gesture."""
        state = self._get_state(hand_label)

        # Get velocity
        raw_velocity = float(np.hypot(metrics.velocity[0], metrics.velocity[1]))
        smoothed_vel_arr = state['ewma_velocity'].update([raw_velocity])
        smoothed_velocity = float(smoothed_vel_arr[0])

        # Check cooldown
        now = time.time()
        in_cooldown = now - state['last_clap_time'] < self.cooldown_s

        # Metadata for visual feedback
        base_metadata = {
            'hand_label': hand_label,
            'raw_velocity': raw_velocity,
            'smoothed_velocity': smoothed_velocity,
            'velocity_threshold': self.velocity_threshold,
            'in_cooldown': in_cooldown,
            'cooldown_remaining': max(0.0, self.cooldown_s - (now - state['last_clap_time'])),
            'hold_count': state['hold_count'],
            'reason': None,
        }

        # Cooldown check
        if in_cooldown:
            state['hold_count'] = 0
            base_metadata['reason'] = 'in_cooldown'
            return GestureResult(detected=False, gesture_name='fist_clap', metadata=base_metadata)

        # Check velocity
        if smoothed_velocity >= self.velocity_threshold:
            state['hold_count'] += 1
            base_metadata['hold_count'] = state['hold_count']
            base_metadata['reason'] = 'high_velocity_detected'

            # Check if held long enough
            if state['hold_count'] >= self.hold_frames:
                state['last_clap_time'] = now
                state['hold_count'] = 0
                base_metadata['reason'] = 'clap_confirmed'

                return GestureResult(
                    detected=True,
                    gesture_name='fist_clap',
                    confidence=1.0,
                    metadata=base_metadata
                )
        else:
            state['hold_count'] = 0
            base_metadata['reason'] = 'velocity_too_low'

        return GestureResult(
            detected=False,
            gesture_name='fist_clap',
            confidence=0.0,
            metadata=base_metadata
        )
```

---

## Testing Your Gesture

### Quick Test: Add Debug Print

```python
# In hands_app.py process_gestures():
if 'my_gesture' in all_gestures.get('right', {}):
    result = all_gestures['right']['my_gesture']
    print(f"✓ My gesture: detected={result.detected}, conf={result.confidence}")
    print(f"  Metadata: {result.metadata}")
```

### Full Integration Steps

1. **Create the class** in `gesture_detectors.py`
2. **Add to GestureManager** in `__init__()` and `process_hand()`
3. **Add to config.json** with thresholds
4. **Test with debug prints**
5. **Map to action** in action_map
6. **Run and test!**

---

## Common Patterns

### Pattern 1: Static Gesture (no movement)

```python
# Example: Open Hand (all 5 fingers extended)
if metrics.fingers_extended['thumb'] and \
   metrics.fingers_extended['index'] and \
   metrics.fingers_extended['middle'] and \
   metrics.fingers_extended['ring'] and \
   metrics.fingers_extended['pinky']:
    return GestureResult(detected=True, ...)
```

### Pattern 2: Dynamic Gesture (with movement)

```python
# Example: Swipe gesture
velocity = metrics.velocity
if abs(velocity[0]) > velocity_threshold_x:
    direction = 'right' if velocity[0] > 0 else 'left'
    return GestureResult(detected=True, ...)
```

### Pattern 3: Distance-Based Gesture

```python
# Example: Pinch (fingers close together)
thumb_index_dist = metrics.tip_distances['index_thumb']
if thumb_index_dist < distance_threshold:
    return GestureResult(detected=True, ...)
```

### Pattern 4: Combination Gesture (two hands)

```python
# This would be in bimanual_gestures.py instead
# But same principle - check both hands' metrics
if left_metrics.fingers_extended['pinky'] and \
   right_metrics.velocity[1] > threshold:
    # Left pinky up + right hand moving down
    return GestureResult(detected=True, ...)
```

---

## Debugging Tips

```python
# Print available metrics
print(f"Tip positions: {metrics.tip_positions}")
print(f"Extended fingers: {metrics.fingers_extended}")
print(f"All distances: {metrics.tip_distances}")
print(f"Velocity: {metrics.velocity}")
print(f"Hand size: {metrics.diag_rel}")

# Check state
print(f"Current state: {state}")

# Validate metadata
print(f"Returning metadata: {base_metadata}")
```

---

## Next Steps

1. **Choose your gesture idea** - What should it detect?
2. **Sketch the logic** - What metrics matter?
3. **Use the template** - Fill in your detection logic
4. **Test with debug prints** - See what metrics you get
5. **Refine thresholds** - Tune config values
6. **Map to actions** - Link to system control
7. **Celebrate!** 🎉

Good luck creating your custom gesture!
