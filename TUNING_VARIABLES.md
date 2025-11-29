# Gesture Detection Tuning Variables

This document lists all tunable parameters for the gesture detection system. Watch the on-screen values in `test_gestures.py` (enable with metrics) to see real-time values and adjust thresholds accordingly.

---

## 📍 File: `gesture_detectors.py`

### GestureManager.**init**()

Location: Line ~448

```python
# PinchDetector parameters
self.pinch = PinchDetector(
    thresh_rel=0.055,      # Distance threshold (relative to image diagonal)
    hold_frames=5,         # Frames to hold before triggering
    cooldown_s=0.6         # Cooldown period in seconds
)

# PointingDetector parameters
self.pointing = PointingDetector(
    min_extension_ratio=0.12  # Minimum distance between extended index and other fingers
)

# SwipeDetector parameters
self.swipe = SwipeDetector(
    velocity_threshold=0.8,   # Velocity threshold (normalized units/sec)
    cooldown_s=0.5            # Cooldown period in seconds
)

# ZoomDetector parameters
self.zoom = ZoomDetector(
    scale_threshold=0.15,     # 15% change threshold to trigger zoom
    history_size=5            # Number of frames to track
)

# OpenHandDetector parameters
self.open_hand = OpenHandDetector(
    min_fingers=4             # Minimum fingers extended (4 or 5)
)
```

---

## 🎯 PinchDetector Parameters

### `thresh_rel` (Default: 0.055)

**What it does**: Distance threshold between thumb tip and index tip (relative to image diagonal)
**On-screen display**: "Pinch: X.XXXX" and "Threshold: X.XXXX"
**How to tune**:

- Watch "Pinch: X.XXXX" value when you pinch
- If it shows 0.06 when pinching, set `thresh_rel=0.065` (slightly higher)
- If false positives, decrease value
- If hard to trigger, increase value

**Typical range**: 0.04 - 0.08

### `hold_frames` (Default: 5)

**What it does**: Number of consecutive frames distance must be below threshold
**How to tune**:

- Increase if detecting too many brief accidental pinches
- Decrease if pinch feels laggy/unresponsive
- At 30 FPS, 5 frames = ~167ms delay

**Typical range**: 3 - 10 frames

### `cooldown_s` (Default: 0.6)

**What it does**: Minimum time in seconds between pinch detections
**How to tune**:

- Increase if getting rapid-fire pinch triggers
- Decrease if you want faster repeated pinches
- Prevents double-clicks

**Typical range**: 0.3 - 1.0 seconds

---

## 👉 PointingDetector Parameters

### `min_extension_ratio` (Default: 0.12)

**What it does**: Minimum distance between extended index finger and curled middle finger
**On-screen display**: "Ext: X.XXXX" (when index is extended)
**How to tune**:

- Watch "Ext: X.XXXX" value when you point with index only
- If pointing not detected, decrease value
- If detected when index + middle extended, increase value

**Typical range**: 0.08 - 0.15

### `max_speed` (Default: 0.5)

**What it does**: Maximum velocity (normalized units/sec) allowed for stable pointing
**How to tune**:

- Increase if pointing not detected during slow movement
- Decrease if you want only stationary pointing

**Typical range**: 0.3 - 0.8

---

## 🌊 SwipeDetector Parameters

### `velocity_threshold` (Default: 0.8)

**What it does**: Minimum velocity (normalized units/sec) to trigger swipe
**On-screen display**: "Vel: X.XXX" (velocity magnitude)
**How to tune**:

- Watch "Vel: X.XXX" value when you swipe fast
- If swipe not detected, decrease threshold
- If detecting small movements as swipes, increase threshold

**Typical range**: 0.5 - 1.5

### `cooldown_s` (Default: 0.5)

**What it does**: Minimum time between swipe detections
**How to tune**:

- Increase to prevent rapid-fire swipes
- Decrease for faster repeated swipes

**Typical range**: 0.3 - 1.0 seconds

### `history_size` (Default: 10)

**What it does**: Number of frames used for velocity smoothing
**How to tune**:

- Increase for smoother, less jittery detection
- Decrease for more responsive detection

**Typical range**: 5 - 20 frames

---

## 🔍 ZoomDetector Parameters

### `scale_threshold` (Default: 0.15)

**What it does**: Minimum relative change (15%) in 3-finger spread to trigger zoom
**On-screen display**: "Zoom: X.XXXX" (weighted spread value)
**How to tune**:

- Watch "Zoom: X.XXXX" value when you pinch/spread 3 fingers
- Calculate change: (final - initial) / initial
- If zoom too sensitive, increase threshold
- If hard to trigger, decrease threshold

**Typical range**: 0.10 - 0.25 (10% - 25% change)

### `history_size` (Default: 5)

**What it does**: Number of frames used to detect continuous direction
**How to tune**:

- Increase for more stable detection (requires longer continuous motion)
- Decrease for quicker response (may be jittery)
- Minimum 3 frames required for direction detection

**Typical range**: 3 - 10 frames

---

## 🖐️ OpenHandDetector Parameters

### `min_fingers` (Default: 4)

**What it does**: Minimum number of extended fingers to detect open hand
**On-screen display**: "Fingers: N"
**How to tune**:

- Set to 5 if you want strict 5-finger detection
- Set to 4 for tolerance (detects 4 or 5 fingers)
- Set to 3 if you have trouble extending all fingers

**Typical range**: 3 - 5 fingers

---

## 📊 MediaPipe Hand Tracking Parameters

### File: `test_gestures.py` and `openCV_test.py`

```python
hands = mp_hands.Hands(
    min_detection_confidence=0.7,   # Initial detection confidence
    min_tracking_confidence=0.3,    # Continuous tracking confidence
    max_num_hands=2                 # Number of hands to track
)
```

### `min_detection_confidence` (Default: 0.7)

**What it does**: Confidence threshold for initial hand detection (first frame)
**How to tune**:

- Increase (0.8-0.9) if detecting false hands in background
- Decrease (0.5-0.6) if hand not detected at all

**Typical range**: 0.5 - 0.9

### `min_tracking_confidence` (Default: 0.3)

**What it does**: Confidence threshold for continuous tracking (subsequent frames)
**How to tune**:

- **Lower = better fast motion tracking** (less likely to lose hand)
- Increase if tracking jitters or jumps to other objects
- Current 0.3 is optimized for fast gestures

**Typical range**: 0.3 - 0.7

### `max_num_hands` (Default: 2)

**What it does**: Maximum number of hands to detect
**How to tune**:

- Set to 1 for single-hand applications (faster processing)
- Set to 2 for two-hand gestures

**Typical range**: 1 - 2

---

## 🎥 Camera Parameters

### File: `test_gestures.py` and `openCV_test.py`

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)   # Default: 640 or 320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # Default: 400 or 240
cap.set(cv2.CAP_PROP_FPS, 60)              # Target FPS
```

### Resolution (Default: 640x400)

**What it does**: Camera capture resolution
**How to tune**:

- **Lower resolution = faster processing**
- Use 320x240 for maximum speed (recommended for testing)
- Use 640x480 for better accuracy
- Use 1280x720 only if you have powerful hardware

**Typical values**: 320x240, 640x400, 640x480, 1280x720

### FPS (Default: 60)

**What it does**: Requested frames per second from camera
**Note**: Actual FPS depends on camera hardware
**How to tune**:

- Set to 60 for fast motion tracking
- Check terminal output for actual achieved FPS
- Some cameras max out at 30 FPS

**Typical values**: 30, 60

---

## 🛠️ Smoothing Parameters

### File: `openCV_test.py` (EWMA smoothing)

```python
idx_ewma = EWMA(alpha=0.25)
thumb_ewma = EWMA(alpha=0.25)
```

### `alpha` (Default: 0.25)

**What it does**: Exponential smoothing factor
**How to tune**:

- **Lower (0.1-0.2) = smoother, more lag**
- **Higher (0.3-0.5) = more responsive, more jitter**
- alpha = 0.0 would freeze position
- alpha = 1.0 would be no smoothing (raw input)

**Typical range**: 0.1 - 0.5

---

## 📐 Distance Calculation Method

All distances are **relative to image diagonal** for resolution independence:

```python
dist_rel = pixel_distance / sqrt(width² + height²)
```

This means:

- Same gesture works at 320x240 or 1280x720
- Same gesture works if hand is close or far from camera
- Threshold values are dimensionless (0..1 range)

**Typical relative distances**:

- Pinch (thumb-index): 0.04 - 0.08
- Zoom spread (3 fingers): 0.10 - 0.25
- Finger extension gap: 0.08 - 0.15

---

## 🧪 Testing Workflow

1. **Start test script with metrics enabled**:

   ```bash
   python test_gestures.py --width 320 --height 240
   ```

2. **Press `d` to enable debug mode** (shows metadata)

3. **Perform gesture slowly and watch on-screen values**:

   - Pinch: Watch "Pinch: X.XXXX" value
   - Zoom: Watch "Zoom: X.XXXX" spread value
   - Swipe: Watch "Vel: X.XXX" velocity
   - Pointing: Watch "Ext: X.XXXX" extension ratio

4. **Note the value when gesture feels natural**

5. **Edit `gesture_detectors.py` GestureManager.**init**()**:

   - Adjust threshold to be slightly above/below observed value
   - Save and restart test

6. **Repeat until gestures feel responsive and accurate**

---

## 🎯 Quick Start Tuning Recommendations

### If gestures too hard to trigger:

- **Pinch**: Increase `thresh_rel` (try 0.07)
- **Zoom**: Decrease `scale_threshold` (try 0.12)
- **Swipe**: Decrease `velocity_threshold` (try 0.6)
- **Pointing**: Decrease `min_extension_ratio` (try 0.10)

### If too many false positives:

- **Pinch**: Decrease `thresh_rel` (try 0.045)
- **Zoom**: Increase `scale_threshold` (try 0.20)
- **Swipe**: Increase `velocity_threshold` (try 1.0)
- **Pointing**: Increase `min_extension_ratio` (try 0.15)

### If gestures feel laggy:

- **All**: Decrease `hold_frames` / `history_size`
- **All**: Decrease `cooldown_s`
- **Camera**: Use lower resolution (320x240)

### If gestures too jittery:

- **All**: Increase `hold_frames` / `history_size`
- **Smoothing**: Decrease `alpha` in EWMA (try 0.15)
- **MediaPipe**: Increase `min_tracking_confidence` (try 0.4)

---

## 📝 Parameter Change Log Template

Keep track of your tuning changes:

```
Date: 2025-11-30
Changed: pinch.thresh_rel from 0.055 to 0.065
Reason: Pinch was hard to trigger at normal distance
Result: Better responsiveness, no false positives

Date: 2025-11-30
Changed: zoom.scale_threshold from 0.15 to 0.12
Reason: Zoom in not detecting with small finger movements
Result: More sensitive, works for subtle zooms
```

---

## 🚀 Advanced Tuning

### Adaptive Thresholds

Consider implementing auto-calibration that adjusts thresholds based on:

- User's hand size
- Camera distance
- Lighting conditions
- Recent detection history

### Per-User Profiles

Save tuned parameters per user:

```python
user_config = {
    'pinch_thresh': 0.065,
    'zoom_thresh': 0.12,
    'swipe_vel': 0.7
}
```

### A/B Testing

Test multiple threshold sets and measure:

- Detection accuracy
- False positive rate
- User satisfaction

---

Good luck with your tuning! 🎮
