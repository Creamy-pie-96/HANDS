# HANDS Gesture Detection Implementation Guide

**Goal:** Build a complete gesture control system step-by-step, learning as you go.

**Your existing tools (already working!):**

- ✅ `math_utils.py`: EWMA smoothing, distance calculation, ClickDetector
- ✅ `openCV_test.py`: Camera capture, MediaPipe integration, visualization
- ✅ Relative distance detection (invariant to hand size/distance)

**What we're building:**

- `gesture_detectors.py`: Modular detectors for pinch, pointing, swipe, zoom, open hand
- Each detector is independent and reusable for left/right hand
- All work in normalized coordinates (0..1) for resolution independence

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (Start Here!)

**Goal:** Get hand metrics working and test with simple detection

#### STEP 1: Implement `compute_hand_metrics()`

**File:** `gesture_detectors.py` (lines ~80-120)

**What you're building:** A function that takes MediaPipe landmarks and computes everything detectors need.

**Code template:**

```python
def compute_hand_metrics(landmarks, img_shape, prev_metrics=None):
    # 1. Convert landmarks to numpy array
    norm = landmarks_to_array(landmarks.landmark)  # ALREADY IMPORTED!

    # 2. Compute centroid (geometric center)
    centroid = (float(norm[:, 0].mean()), float(norm[:, 1].mean()))

    # 3. Compute bounding box
    xmin, xmax = float(norm[:, 0].min()), float(norm[:, 0].max())
    ymin, ymax = float(norm[:, 1].min()), float(norm[:, 1].max())
    bbox = (xmin, ymin, xmax, ymax)

    # 4. Extract tip positions (use LANDMARK_NAMES dict for indices)
    tip_positions = {
        'thumb': tuple(norm[4]),      # index 4 = THUMB_TIP
        'index': tuple(norm[8]),      # index 8 = INDEX_TIP
        'middle': tuple(norm[12]),    # index 12 = MIDDLE_TIP
        'ring': tuple(norm[16]),      # index 16 = RING_TIP
        'pinky': tuple(norm[20]),     # index 20 = PINKY_TIP
    }

    # 5. Compute distances between key tips
    #    Use euclidean() from math_utils - ALREADY IMPORTED!
    tip_distances = {
        'index_thumb': float(euclidean(norm[8], norm[4])),
        'index_middle': float(euclidean(norm[8], norm[12])),
        'thumb_middle': float(euclidean(norm[4], norm[12])),
        # Add more pairs as needed
    }

    # 6. Compute hand diagonal relative to image diagonal
    h, w = img_shape[0], img_shape[1]
    hand_diag_px = np.hypot(xmax - xmin, ymax - ymin) * w  # convert normalized to pixels
    img_diag_px = np.hypot(w, h)
    diag_rel = hand_diag_px / img_diag_px

    # 7. Detect which fingers are extended
    fingers_extended = {
        'thumb': is_finger_extended(norm, 'thumb'),
        'index': is_finger_extended(norm, 'index'),
        'middle': is_finger_extended(norm, 'middle'),
        'ring': is_finger_extended(norm, 'ring'),
        'pinky': is_finger_extended(norm, 'pinky'),
    }

    # 8. Compute velocity if previous metrics exist
    velocity = (0.0, 0.0)
    if prev_metrics is not None:
        dt = time.time() - prev_metrics.timestamp
        if dt > 0:
            vx = (centroid[0] - prev_metrics.centroid[0]) / dt
            vy = (centroid[1] - prev_metrics.centroid[1]) / dt
            velocity = (vx, vy)

    # 9. Create and return HandMetrics object
    return HandMetrics(
        landmarks_norm=norm,
        timestamp=time.time(),
        centroid=centroid,
        bbox=bbox,
        tip_positions=tip_positions,
        tip_distances=tip_distances,
        fingers_extended=fingers_extended,
        diag_rel=diag_rel,
        velocity=velocity
    )
```

**Testing step 1:**

```python
# Add to bottom of gesture_detectors.py temporarily
if __name__ == "__main__":
    # Quick test with dummy landmarks
    print("compute_hand_metrics: TODO - test with real camera")
```

---

#### STEP 2: Implement `is_finger_extended()`

**File:** `gesture_detectors.py` (lines ~123-150)

**What you're building:** Detect if a finger is extended (straightened) or curled.

**The rule:** For most fingers, tip.y < pip.y means extended (y increases downward!)

**Code template:**

```python
def is_finger_extended(landmarks_norm, finger_name, handedness='Right'):
    """
    Finger landmark indices:
    - Thumb: CMC=1, MCP=2, IP=3, TIP=4
    - Index: MCP=5, PIP=6, DIP=7, TIP=8
    - Middle: MCP=9, PIP=10, DIP=11, TIP=12
    - Ring: MCP=13, PIP=14, DIP=15, TIP=16
    - Pinky: MCP=17, PIP=18, DIP=19, TIP=20
    """

    if finger_name == 'index':
        # Simple rule: tip (8) is above PIP (6) in y-coordinate
        # Remember: y increases downward, so "above" means smaller y
        return landmarks_norm[8][1] < landmarks_norm[6][1]

    elif finger_name == 'middle':
        return landmarks_norm[12][1] < landmarks_norm[10][1]

    elif finger_name == 'ring':
        return landmarks_norm[16][1] < landmarks_norm[14][1]

    elif finger_name == 'pinky':
        return landmarks_norm[20][1] < landmarks_norm[18][1]

    elif finger_name == 'thumb':
        # Thumb is trickier - it moves sideways not up/down
        # Use x-coordinate comparison
        # For right hand: thumb extended means tip (4) is left of MCP (2)
        # For left hand: opposite
        if handedness == 'Right':
            return landmarks_norm[4][0] < landmarks_norm[2][0]
        else:
            return landmarks_norm[4][0] > landmarks_norm[2][0]

    return False
```

**Testing step 2:**

```python
# Test by printing finger states in main loop
# Add to openCV_test.py temporarily:
for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
    extended = is_finger_extended(norm, finger)
    print(f"{finger}: {extended}", end="  ")
print()  # newline
```

---

### Phase 2: Easy Detectors (Build Confidence)

#### STEP 3: Implement `OpenHandDetector.detect()`

**File:** `gesture_detectors.py` (lines ~270-290)

**Why start here:** Simplest detector - just count extended fingers!

**Code:**

```python
def detect(self, metrics: HandMetrics) -> GestureResult:
    # Count how many fingers are extended
    count = sum(metrics.fingers_extended.values())

    # Detect if at least min_fingers are extended
    detected = count >= self.min_fingers

    return GestureResult(
        detected=detected,
        gesture_name='open_hand',
        confidence=1.0 if detected else 0.0,
        metadata={'finger_count': count}
    )
```

**Testing:**

- Open your hand in front of camera → should detect
- Close fist → should not detect
- Try 4 fingers vs 5 fingers

---

#### STEP 4: Implement `PinchDetector.detect()`

**File:** `gesture_detectors.py` (lines ~170-190)

**What you're doing:** Adapting your WORKING ClickDetector code!

**Code:**

```python
def detect(self, metrics: HandMetrics) -> GestureResult:
    # Get the distance you already computed in metrics
    dist_rel = metrics.tip_distances['index_thumb']

    # Use same logic as ClickDetector.pinched()
    now = time.time()
    if now - self._last_time < self.cooldown_s:
        return GestureResult(detected=False, gesture_name='pinch')

    if dist_rel <= self.thresh_rel:
        self._count += 1
        if self._count >= self.hold_frames:
            self._last_time = now
            self._count = 0
            return GestureResult(
                detected=True,
                gesture_name='pinch',
                confidence=1.0,
                metadata={'dist_rel': dist_rel}
            )
    else:
        self._count = 0

    return GestureResult(detected=False, gesture_name='pinch')
```

**Testing:**

- Should work exactly like your current pinch detection
- Print `dist_rel` to see values

---

### Phase 3: Movement-Based Detectors

#### STEP 5: Implement `PointingDetector.detect()`

**File:** `gesture_detectors.py` (lines ~194-225)

**What you're building:** Detect when ONLY index finger is extended and stable.

**Code:**

```python
def detect(self, metrics: HandMetrics) -> GestureResult:
    # Condition 1: Index must be extended
    if not metrics.fingers_extended['index']:
        return GestureResult(detected=False, gesture_name='pointing')

    # Condition 2: Other fingers should be curled (allow 1 for tolerance)
    other_fingers = ['middle', 'ring', 'pinky']  # don't count thumb
    extended_count = sum(metrics.fingers_extended[f] for f in other_fingers)
    if extended_count > 1:
        return GestureResult(detected=False, gesture_name='pointing')

    # Condition 3: Index tip should be far enough from palm
    index_tip = metrics.tip_positions['index']
    centroid = metrics.centroid
    distance = euclidean(index_tip, centroid)  # Use your utility!

    if distance < self.min_extension_ratio:
        return GestureResult(detected=False, gesture_name='pointing')

    # Condition 4: Hand should be relatively stable (low velocity)
    speed = np.hypot(metrics.velocity[0], metrics.velocity[1])
    if speed > 0.5:  # adjust this threshold
        return GestureResult(detected=False, gesture_name='pointing')

    # All conditions met!
    direction = (index_tip[0] - centroid[0], index_tip[1] - centroid[1])

    return GestureResult(
        detected=True,
        gesture_name='pointing',
        confidence=1.0,
        metadata={
            'tip_position': index_tip,
            'direction': direction,
            'distance': distance
        }
    )
```

**Testing:**

- Point with index finger → should detect
- Move hand slowly while pointing → should still detect
- Extend multiple fingers → should NOT detect
- Swipe quickly → should NOT detect

---

#### STEP 6: Implement `SwipeDetector.detect()`

**File:** `gesture_detectors.py` (lines ~228-265)

**What you're building:** Detect fast hand movements (left/right/up/down).

**Code:**

```python
def detect(self, metrics: HandMetrics) -> GestureResult:
    # Add current metrics to history
    self.history.append(metrics)

    # Need at least 3 frames to compute velocity reliably
    if len(self.history) < 3:
        return GestureResult(detected=False, gesture_name='swipe')

    # Check cooldown
    now = time.time()
    if now - self.last_swipe_time < self.cooldown_s:
        return GestureResult(detected=False, gesture_name='swipe')

    # Compute velocity from current metrics (already has velocity!)
    vx, vy = metrics.velocity
    speed = np.hypot(vx, vy)

    # Check if speed exceeds threshold
    if speed < self.velocity_threshold:
        return GestureResult(detected=False, gesture_name='swipe')

    # Determine direction based on which component is larger
    if abs(vx) > abs(vy):
        direction = 'right' if vx > 0 else 'left'
    else:
        direction = 'down' if vy > 0 else 'up'

    # Update cooldown timer
    self.last_swipe_time = now

    return GestureResult(
        detected=True,
        gesture_name='swipe',
        confidence=1.0,
        metadata={
            'direction': direction,
            'speed': speed,
            'velocity': (vx, vy)
        }
    )
```

**Testing:**

- Swipe hand quickly left/right → should detect direction
- Swipe up/down → should detect direction
- Move hand slowly → should NOT detect
- Print speed values to tune threshold

---

#### STEP 7: Implement `ZoomDetector.detect()`

**File:** `gesture_detectors.py` (lines ~268-295)

**What you're building:** Detect 3-finger pinch for zoom in/out.

**Code:**

```python
def detect(self, metrics: HandMetrics) -> GestureResult:
    # Check if thumb, index, and middle are all extended
    required = ['thumb', 'index', 'middle']
    if not all(metrics.fingers_extended[f] for f in required):
        self.history.clear()  # Reset if gesture breaks
        return GestureResult(detected=False, gesture_name='zoom')

    # Compute current "spread" - average distance between the 3 tips
    spread = (
        metrics.tip_distances.get('index_thumb', 0) +
        metrics.tip_distances.get('thumb_middle', 0) +
        metrics.tip_distances.get('index_middle', 0)
    ) / 3.0

    # Add to history
    self.history.append(spread)

    # Need at least 2 frames to compare
    if len(self.history) < 2:
        return GestureResult(detected=False, gesture_name='zoom')

    # Compare current spread to average of recent history
    prev_spread = np.mean(list(self.history)[:-1])

    if prev_spread < 0.001:  # Avoid division by zero
        return GestureResult(detected=False, gesture_name='zoom')

    # Compute scale change ratio
    ratio = spread / prev_spread

    # Detect zoom based on change threshold
    if ratio > 1 + self.scale_threshold:
        zoom_type = 'out'
        detected = True
    elif ratio < 1 - self.scale_threshold:
        zoom_type = 'in'
        detected = True
    else:
        detected = False
        zoom_type = None

    return GestureResult(
        detected=detected,
        gesture_name='zoom',
        confidence=1.0 if detected else 0.0,
        metadata={
            'zoom_type': zoom_type,
            'scale_ratio': ratio,
            'spread': spread
        }
    )
```

**Testing:**

- Extend thumb + index + middle
- Move them apart → should detect zoom out
- Bring them together → should detect zoom in
- Print spread and ratio values to tune threshold

---

### Phase 4: Integration

#### STEP 8: Implement `GestureManager.process_hand()`

**File:** `gesture_detectors.py` (lines ~320-360)

**What you're building:** Orchestrate all detectors and handle conflicts.

**Code:**

```python
def process_hand(self, landmarks, img_shape, hand_label='right'):
    # Get previous metrics for velocity
    prev = self.history[hand_label][-1] if self.history[hand_label] else None

    # Compute current metrics
    metrics = compute_hand_metrics(landmarks, img_shape, prev)
    self.history[hand_label].append(metrics)

    # Run all detectors
    results = {}

    # Order matters - check in priority order
    # 1. Pinch (highest priority)
    pinch_result = self.pinch.detect(metrics)
    if pinch_result.detected:
        results['pinch'] = pinch_result
        # When pinching, don't check other gestures
        return results

    # 2. Open hand (mode switch)
    open_result = self.open_hand.detect(metrics)
    if open_result.detected:
        results['open_hand'] = open_result
        # Open hand can coexist with others, continue checking

    # 3. Zoom (requires specific 3-finger config)
    zoom_result = self.zoom.detect(metrics)
    if zoom_result.detected:
        results['zoom'] = zoom_result
        return results  # Zoom is exclusive

    # 4. Pointing (for cursor control)
    pointing_result = self.pointing.detect(metrics)
    if pointing_result.detected:
        results['pointing'] = pointing_result

    # 5. Swipe (check last to avoid conflicts with stable gestures)
    swipe_result = self.swipe.detect(metrics)
    if swipe_result.detected:
        results['swipe'] = swipe_result

    return results
```

**Testing:**

- Each gesture should be detected independently
- Pinch should block other detections
- Multiple non-conflicting gestures can coexist

---

### Phase 5: Visualization & Testing

#### STEP 9: Implement `visualize_hand_metrics()`

**File:** `gesture_detectors.py` (lines ~375-400)

**What you're building:** Debug overlay to see what detectors see.

**Code:**

```python
def visualize_hand_metrics(frame, metrics: HandMetrics, color=(0, 255, 0)):
    h, w = frame.shape[:2]

    # Draw bounding box
    bbox = metrics.bbox
    pt1 = (int(bbox[0] * w), int(bbox[1] * h))
    pt2 = (int(bbox[2] * w), int(bbox[3] * h))
    cv2.rectangle(frame, pt1, pt2, (255, 0, 255), 2)

    # Draw centroid
    cx, cy = metrics.centroid
    center = (int(cx * w), int(cy * h))
    cv2.circle(frame, center, 8, (0, 255, 255), -1)

    # Draw finger tips with different colors based on extended state
    finger_colors = {
        'thumb': (255, 0, 0),
        'index': (0, 255, 0),
        'middle': (0, 0, 255),
        'ring': (255, 255, 0),
        'pinky': (255, 0, 255)
    }

    for finger_name, pos in metrics.tip_positions.items():
        px, py = int(pos[0] * w), int(pos[1] * h)
        color = finger_colors[finger_name] if metrics.fingers_extended[finger_name] else (128, 128, 128)
        cv2.circle(frame, (px, py), 6, color, -1)

    # Draw velocity arrow
    vx, vy = metrics.velocity
    if np.hypot(vx, vy) > 0.1:
        end_x = int((cx + vx * 0.2) * w)
        end_y = int((cy + vy * 0.2) * h)
        cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), 2)

    # Draw text info
    info_text = f"Speed: {np.hypot(vx, vy):.2f}"
    cv2.putText(frame, info_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

---

### Phase 6: Integration with openCV_test.py

#### STEP 10: Update main demo to use GestureManager

**File:** Create new `openCV_gesture_demo.py` or modify `openCV_test.py`

**Code template:**

```python
# At top, add import
from gesture_detectors import GestureManager, visualize_hand_metrics

# In main(), after MediaPipe setup
gesture_manager = GestureManager()

# In the main loop, replace manual detection with:
if results.multi_hand_landmarks:
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # Determine hand label (you can use results.multi_handedness)
        hand_label = 'right'  # or extract from results.multi_handedness[idx]

        # Process hand through gesture manager
        gestures = gesture_manager.process_hand(
            hand_landmarks,
            frame_bgr.shape,
            hand_label
        )

        # Visualize detected gestures
        y_offset = 70
        for gesture_name, result in gestures.items():
            text = f"{gesture_name.upper()}: {result.metadata}"
            cv2.putText(frame_bgr, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30

        # Optionally visualize hand metrics
        if debug:
            metrics = gesture_manager.history[hand_label][-1]
            visualize_hand_metrics(frame_bgr, metrics)
```

---

## 🎯 Testing Checklist

### Unit Testing

Create `test_gesture_detectors.py`:

```python
from gesture_detectors import *
import numpy as np

def create_test_metrics(finger_states, distances):
    """Helper to create HandMetrics for testing."""
    return HandMetrics(
        landmarks_norm=np.zeros((21, 2)),
        timestamp=time.time(),
        centroid=(0.5, 0.5),
        bbox=(0.3, 0.3, 0.7, 0.7),
        tip_positions={'index': (0.5, 0.3), 'thumb': (0.4, 0.4)},
        tip_distances=distances,
        fingers_extended=finger_states,
        diag_rel=0.2,
        velocity=(0.0, 0.0)
    )

# Test OpenHandDetector
def test_open_hand():
    detector = OpenHandDetector(min_fingers=4)

    # Test with all fingers extended
    metrics = create_test_metrics(
        {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
        {}
    )
    result = detector.detect(metrics)
    assert result.detected == True, "Should detect open hand"

    # Test with fist
    metrics = create_test_metrics(
        {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
        {}
    )
    result = detector.detect(metrics)
    assert result.detected == False, "Should not detect with fist"

    print("✓ OpenHandDetector tests passed")

if __name__ == "__main__":
    test_open_hand()
    # Add more tests...
```

### Integration Testing

1. Run demo with each gesture and verify detection
2. Test edge cases (transitions between gestures)
3. Test with different lighting conditions
4. Test at different distances from camera

---

## 🐛 Debugging Tips

### Common Issues & Fixes

**Problem:** Finger detection is inverted

- **Fix:** Check if y-coordinate comparison is correct (y increases downward)
- **Debug:** Print `landmarks_norm[tip_idx][1]` and `landmarks_norm[pip_idx][1]`

**Problem:** Gestures trigger too easily

- **Fix:** Increase thresholds (velocity_threshold, scale_threshold, etc.)
- **Debug:** Print metric values and observe ranges

**Problem:** Gestures don't trigger

- **Fix:** Decrease thresholds or check finger extension logic
- **Debug:** Add print statements in detector conditions

**Problem:** Multiple conflicting gestures detected

- **Fix:** Improve priority logic in GestureManager
- **Debug:** Print all results before filtering

### Visualization for Debugging

Always use `visualize_hand_metrics()` when developing:

- See which fingers are detected as extended (colored vs gray dots)
- See hand velocity (arrow from centroid)
- See bounding box and centroid position

---

## 📊 Tuning Parameters

After implementing, tune these values based on testing:

### Distance Thresholds

```python
# In PinchDetector
thresh_rel = 0.055  # Current value, adjust 0.04-0.08

# In PointingDetector
min_extension_ratio = 0.12  # Adjust 0.10-0.15
```

### Velocity Thresholds

```python
# In SwipeDetector
velocity_threshold = 0.8  # Adjust 0.6-1.2

# In PointingDetector (for stability)
max_speed_for_pointing = 0.5  # Adjust 0.3-0.7
```

### Temporal Parameters

```python
# Hold frames (at 30 FPS)
hold_frames = 5  # 5 frames ≈ 167ms
# Increase for more stability, decrease for faster response

# Cooldown periods
cooldown_s = 0.6  # Adjust 0.3-1.0
```

---

## 🚀 Next Steps After Implementation

Once all detectors work:

1. **Add action mapping**

   - Create `action_mapper.py` to map gestures to OS commands
   - Use `pyautogui` for mouse/keyboard control

2. **Add two-hand gestures**

   - Track both hands in `GestureManager`
   - Implement combined gestures (e.g., two-hand zoom)

3. **Add state machine**

   - Implement mode switching with open hand gesture
   - Different gesture meanings in different modes

4. **Performance optimization**

   - Profile code to find bottlenecks
   - Optimize history buffer sizes
   - Add frame skipping if needed

5. **Configuration file**
   - Save/load threshold settings
   - Allow runtime tuning with UI

---

## 📚 Learning Resources

- **MediaPipe Hands landmarks:** https://google.github.io/mediapipe/solutions/hands
- **NumPy operations:** Your `math_utils.py` has good examples
- **Computer Vision patterns:** Your `openCV_test.py` shows the structure

---

## ✅ Success Criteria

You'll know you're done when:

- [ ] All 5 detectors work independently
- [ ] GestureManager integrates them smoothly
- [ ] No false positives in normal use
- [ ] Gestures feel natural and responsive
- [ ] Debug visualization shows correct metrics
- [ ] You can control cursor with pointing
- [ ] Pinch triggers clicks reliably
- [ ] Swipes are detected with correct direction
- [ ] You understand how each part works!

**Remember:** Implement step-by-step. Test each function before moving to the next. Use print statements and visualization liberally during development. Good luck! 🎉
