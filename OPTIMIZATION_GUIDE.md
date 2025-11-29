# Performance Optimization Guide for Hand Tracking

## ✅ Already Implemented

### 1. **Camera FPS Optimization**

- Requesting 60 FPS from camera with `cap.set(cv2.CAP_PROP_FPS, 60)`
- Note: Actual FPS depends on hardware support
- Check terminal output for actual achieved FPS

### 2. **MediaPipe Tracking Confidence**

- Lowered `min_tracking_confidence=0.3` (from 0.5)
- Prevents tracker from falling back to slow detection during fast motion
- Keeps hand tracking continuous during rapid movements

### 3. **Lower Resolution Processing**

- Default: 640x400 (can go even lower with `--width 320 --height 240`)
- Smaller images = faster processing = higher effective FPS
- Trade-off: accuracy vs speed

### 4. **Real-Time Tuning Display**

- Both scripts show live detection values on screen
- Allows you to adjust thresholds based on observed values
- Example: If pinch distance shows 0.06 when you pinch, set threshold to 0.055

---

## 🚀 Additional Performance Optimizations

### 5. **Skip Frame Processing (Temporal Sampling)**

Process every Nth frame instead of every frame, using interpolation between detections.

```python
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame

frame_count = 0
last_landmarks = None

while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = hands.process(frame_rgb)
        last_landmarks = results.multi_hand_landmarks
    else:
        # Use previous frame's landmarks
        results = type('obj', (object,), {'multi_hand_landmarks': last_landmarks})()

    # Rest of processing...
```

**Pros**: 2x speedup with minimal accuracy loss  
**Cons**: Slightly jerkier motion, may miss very brief gestures

---

### 6. **Static Image Mode for Single Frame Processing**

If you're processing individual images (not video), use static mode:

```python
hands = mp_hands.Hands(
    static_image_mode=True,  # Optimize for single images
    max_num_hands=2,
    min_detection_confidence=0.5
)
```

**Use case**: Processing screenshots or still images  
**Don't use**: For video streams (it's slower!)

---

### 7. **Multithreading Camera Capture**

Separate camera reading from processing using threading:

```python
import threading
import queue

class CameraThread(threading.Thread):
    def __init__(self, camera_idx=0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_idx)
        self.queue = queue.Queue(maxsize=2)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.queue.full():
                    self.queue.get()  # Drop oldest frame
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.cap.release()

# Usage
cam_thread = CameraThread(0)
cam_thread.start()

while True:
    frame = cam_thread.read()
    # Process frame...
```

**Benefit**: Camera I/O doesn't block processing  
**Complexity**: Medium (requires threading knowledge)

---

### 8. **GPU Acceleration (TensorFlow Lite GPU)**

Use MediaPipe's GPU backend (requires GPU and proper drivers):

```python
# Install: pip install mediapipe-gpu
import mediapipe as mp

# Use GPU delegate (Linux/Android)
hands = mp.solutions.hands.Hands(
    model_complexity=0,  # Lighter model
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)
```

**Pros**: Significant speedup on GPU-enabled devices  
**Cons**: Setup complexity, platform-dependent

---

### 9. **Reduce Model Complexity**

Use lighter MediaPipe hand model:

```python
hands = mp_hands.Hands(
    model_complexity=0,  # 0=lite, 1=full (default)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3
)
```

**Trade-off**: Faster processing but slightly less accurate landmarks  
**Best for**: Real-time applications where speed > precision

---

### 10. **Optimize NumPy Operations**

Use vectorized operations instead of loops:

```python
# ❌ Slow
distances = []
for i in range(len(points)):
    dist = np.sqrt((points[i][0] - ref[0])**2 + (points[i][1] - ref[1])**2)
    distances.append(dist)

# ✅ Fast
distances = np.linalg.norm(points - ref, axis=1)
```

---

### 11. **Predictive Tracking (Kalman Filter)**

Smooth and predict hand position between frames:

```python
from filterpy.kalman import KalmanFilter
import numpy as np

kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
kf.F = np.array([[1, 0, 1, 0],  # State transition
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                 [0, 1, 0, 0]])
kf.R *= 0.1  # Measurement noise
kf.Q *= 0.01  # Process noise

# In loop:
kf.predict()
measurement = np.array([hand_x, hand_y])
kf.update(measurement)
smoothed_pos = kf.x[:2]  # Use smoothed position
```

**Benefit**: Smoother motion, can predict position when hand temporarily lost  
**Complexity**: High (requires understanding Kalman filtering)

---

### 12. **Adaptive Threshold Tuning**

Auto-adjust detection thresholds based on recent history:

```python
from collections import deque

class AdaptiveThreshold:
    def __init__(self, initial=0.055, history_size=100):
        self.history = deque(maxlen=history_size)
        self.threshold = initial

    def update(self, distance, is_pinch_detected):
        self.history.append(distance)

        # Recalculate threshold as 90th percentile of "non-pinch" distances
        if len(self.history) > 50:
            non_pinch = [d for d in self.history if d > self.threshold * 1.2]
            if non_pinch:
                self.threshold = np.percentile(non_pinch, 10)  # 10th percentile

    def check_pinch(self, distance):
        return distance < self.threshold
```

**Benefit**: Adapts to user's hand size and camera distance  
**Use case**: Multi-user systems or varying lighting conditions

---

### 13. **Lighting Optimization (Auto-Exposure)**

Disable camera auto-exposure for consistent performance:

```python
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)        # Set fixed exposure
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
```

**Benefit**: Consistent tracking across lighting changes  
**Requires**: Good initial lighting setup

---

### 14. **Region of Interest (ROI) Tracking**

Only process hand region after initial detection:

```python
roi_bbox = None  # (x, y, w, h)

while True:
    ret, frame = cap.read()

    if roi_bbox is None:
        # Full frame detection
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # Calculate bounding box from landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x, y = int(min(xs) * w), int(min(ys) * h)
            w_roi, h_roi = int((max(xs) - min(xs)) * w * 1.5), int((max(ys) - min(ys)) * h * 1.5)
            roi_bbox = (x, y, w_roi, h_roi)
    else:
        # Process only ROI
        x, y, w_roi, h_roi = roi_bbox
        roi_frame = frame[y:y+h_roi, x:x+w_roi]
        results = hands.process(roi_frame)
        # ... adjust landmark coords back to full frame coords
```

**Benefit**: 2-3x speedup by processing smaller region  
**Complexity**: High (requires coordinate transformation)

---

### 15. **C++ Migration Path**

When you're ready to move to C++:

```bash
# Install MediaPipe C++
git clone https://github.com/google/mediapipe.git
cd mediapipe

# Build with Bazel
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
```

**Expected speedup**: 3-5x faster than Python  
**Effort**: High (requires C++ knowledge, build system setup)  
**Best for**: Production deployment, embedded systems

---

## 🎯 Recommended Optimization Strategy

### Phase 1: Quick Wins (Implement Now)

1. ✅ Lower resolution (320x240 or 480x360)
2. ✅ Reduce `min_tracking_confidence` to 0.3
3. ✅ Request 60 FPS from camera
4. ✅ Add real-time tuning display

### Phase 2: Medium Effort (Next Week)

5. Skip frame processing (2-3x speedup)
6. Model complexity = 0 (lite model)
7. Optimize NumPy operations in your code

### Phase 3: Advanced (When Needed)

8. Multithreaded camera capture
9. Kalman filter for prediction
10. ROI-based tracking

### Phase 4: Production (Final Stage)

11. GPU acceleration (if available)
12. Migrate to C++ for maximum performance

---

## 📊 Expected Performance Gains

| Optimization               | Speedup | Complexity | Priority |
| -------------------------- | ------- | ---------- | -------- |
| Lower resolution (640→320) | 2x      | Low        | **HIGH** |
| min_tracking_confidence    | 1.5x    | Low        | **HIGH** |
| Skip frames (every 2nd)    | 2x      | Low        | **HIGH** |
| Model complexity=0         | 1.3x    | Low        | Medium   |
| Multithreaded capture      | 1.2x    | Medium     | Medium   |
| ROI tracking               | 2-3x    | High       | Low      |
| C++ migration              | 3-5x    | Very High  | Later    |

---

## 🔧 Tuning Tips

### For Fast Motion Tracking:

- **Priority**: Lower `min_tracking_confidence` (0.2-0.3)
- **Trade-off**: May track background objects as hands occasionally
- **Solution**: Add hand size validation (reject tiny/huge detections)

### For Accuracy:

- **Priority**: Higher resolution (640x480 minimum)
- **Trade-off**: Slower FPS
- **Solution**: Use skip-frame processing to compensate

### For Low-End Hardware:

- **Priority**: 320x240 resolution + model_complexity=0 + skip frames
- **Expected**: 30+ FPS on most laptops

### For High-End Hardware:

- **Priority**: Keep 640x400 + GPU acceleration + multithreading
- **Expected**: 60+ FPS possible

---

## 🧪 Testing Your Optimizations

Run this benchmark to measure improvements:

```python
import time

frame_count = 0
start_time = time.time()

while frame_count < 300:  # Test for 300 frames
    ret, frame = cap.read()
    results = hands.process(frame_rgb)
    frame_count += 1

elapsed = time.time() - start_time
fps = frame_count / elapsed
print(f"Average FPS: {fps:.2f}")
```

**Target**: 30+ FPS for smooth gestures, 60+ FPS for fast motion

---

## 📝 Notes

- Always test optimizations with your actual use case (fast swipes, pinches, etc.)
- Some cameras report 60 FPS but actually deliver 30 FPS
- USB bandwidth can limit multi-camera setups
- Lighting significantly affects MediaPipe accuracy

Good luck with your optimizations! 🚀
