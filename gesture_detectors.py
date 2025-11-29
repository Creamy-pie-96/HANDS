"""
Gesture Detection Module for HANDS (Hand Assisted Navigation and Device System)

This module provides modular gesture detectors that work with MediaPipe hand landmarks.
All detectors operate on normalized coordinates (0..1) to be resolution-independent.

Design principles:
- Reuse existing utilities from math_utils.py (EWMA, euclidean, landmarks_to_array)
- Keep detectors stateless where possible; use small state objects for temporal logic
- Compute hand metrics once per frame and pass to all detectors
- Operate in normalized space; convert to pixels only for visualization
"""

import numpy as np
import time
import cv2
from collections import deque
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from math_utils import landmarks_to_array, euclidean, EWMA, ClickDetector


# Data Structures

@dataclass
class HandMetrics:
    """
    Per-frame computed metrics for a single hand.
    All spatial values are in normalized coordinates (0..1) unless noted.
    """
    # Raw and derived landmark data
    landmarks_norm: np.ndarray  # shape (21, 2) - all landmarks in normalized coords
    timestamp: float            # time.time() when computed
    
    # Spatial metrics (normalized)
    centroid: Tuple[float, float]          # geometric center of hand
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    
    # Tip positions (normalized) - easier access than indexing landmarks_norm
    tip_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # {'thumb': (x,y), 'index': (x,y), 'middle': (x,y), 'ring': (x,y), 'pinky': (x,y)}
    
    # Distances between key points (normalized relative to image diagonal)
    tip_distances: Dict[str, float] = field(default_factory=dict)
    # {'index_thumb': dist, 'index_middle': dist, ...}
    
    # Finger extension state
    fingers_extended: Dict[str, bool] = field(default_factory=dict)
    # {'thumb': True, 'index': False, ...}
    
    # Scale metrics
    diag_rel: float = 0.0  # hand bbox diagonal relative to image diagonal
    
    # Velocity (computed from history)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in normalized units/sec


@dataclass
class GestureResult:
    """Result from a gesture detector."""
    detected: bool
    gesture_name: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

# MediaPipe Hand Landmark indices (for reference)
LANDMARK_NAMES = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8,
    'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
    'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20,
}

def compute_hand_metrics(
    landmarks,  # MediaPipe landmarks object
    img_shape: Tuple[int, int, int],  # (height, width, channels)
    prev_metrics: Optional[HandMetrics] = None
) -> HandMetrics:
    
    # Convert lnadmarks to numpy array
    norm = landmarks_to_array(landmarks.landmark)

    # Compute centroid
    centroid = (float(norm[:, 0].mean()), float(norm[:, 1].mean()))

    # Compute bounding box 
    xmin, xmax = float(norm[:,0].min()), float(norm[:,0].max())
    ymin, ymax = float(norm[:,1].min()), float(norm[:,1].max())

    bbox = (xmin,ymin,xmax,ymax)
    # Extract finget tips positions
    tip_positions = {
        'thumb' : tuple(norm[4]),
        'index' : tuple(norm[8]),
        'middle' : tuple(norm[12]),
        'ring' : tuple(norm[16]),
        'pinky' : tuple(norm[20]),
    }

    # Distence between tips
    tip_distances = {
        'index_thumb' : float(euclidean(norm[4],norm[8])),
        'index_middle' : float(euclidean(norm[8],norm[12])),
        'index_ring' : float(euclidean(norm[8],norm[16])),
        'index_pinky' : float(euclidean(norm[8],norm[20])),
        'thumb_middle' : float(euclidean(norm[4],norm[12])),
        'thumb_ring' : float(euclidean(norm[4],norm[16])),
        'thumb_pinky' : float(euclidean(norm[4],norm[20])),
        'middle_ring' : float(euclidean(norm[12],norm[16])),
        'middle_pinky' : float(euclidean(norm[12],norm[20])),
        'ring_pinky' : float(euclidean(norm[16],norm[20]))
    }

    # diagonal relations

    h, w = img_shape[0], img_shape[1]
    hand_diag_px = np.hypot(xmax-xmin,ymax-ymin)
    img_diag_px = np.hypot(w,h)
    diag_rel = hand_diag_px / img_diag_px

    # detect which finger is extented
    finger_extended = {
        'thumb': is_finger_extended(norm, 'thumb'),
        'index': is_finger_extended(norm, 'index'),
        'middle': is_finger_extended(norm, 'middle'),
        'ring': is_finger_extended(norm, 'ring'),
        'pinky': is_finger_extended(norm, 'pinky'),
    }

    # compute velocity 
    velocity = (0.0,0.0)
    if prev_metrics is not None:
        dt = time.time() - prev_metrics.timestamp
        if dt > 0:
            vx = (centroid[0]-prev_metrics.centroid[0])/dt
            vy = (centroid[1] - prev_metrics.centroid[1])/dt
            velocity = (vx, vy)
    
    return HandMetrics(
        landmarks_norm= norm,
        timestamp= time.time(),
        centroid= centroid,
        bbox=bbox,
        tip_positions=tip_positions,
        tip_distances= tip_distances,
        fingers_extended= finger_extended,
        diag_rel= diag_rel,
        velocity= velocity

    )





def is_finger_extended(
    landmarks_norm: np.ndarray,
    finger_name: str,
    handedness: str = 'Right'
) -> bool:
    
    if finger_name == 'index':
        return landmarks_norm[8][1] < landmarks_norm[6][1]

    elif finger_name == 'middle':
        return landmarks_norm[12][1] < landmarks_norm[10][1]

    elif finger_name == 'ring':
        return landmarks_norm[16][1] < landmarks_norm[14][1]

    elif finger_name == 'pinky':
        return landmarks_norm[20][1] < landmarks_norm[18][1]

    elif finger_name == 'thumb':
        # Thumb is trickier - it moves sideways not up/down
        # Using x-coordinate comparison
        # For right hand: thumb extended means tip (4) is left of MCP (2)
        # For left hand: opposite
        if handedness == 'Right':
            return landmarks_norm[4][0] < landmarks_norm[2][0]
        else:
            return landmarks_norm[4][0] > landmarks_norm[2][0]

    return False


# ============================================================================
# STEP 3: Gesture Detectors (Build on HandMetrics)
# ============================================================================

class PinchDetector:
    """
    Detects thumb-index pinch gesture.
    ALREADY IMPLEMENTED - you can move ClickDetector logic here and adapt it.
    """
    def __init__(self, thresh_rel: float = 0.055, hold_frames: int = 5, cooldown_s: float = 0.6):
        self.thresh_rel = thresh_rel
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self._count = 0
        self._last_time = -999.0
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
       
        dist_rel = metrics.tip_distances['index_thumb']
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


class PointingDetector:
    def __init__(self, min_extension_ratio: float = 0.12):
        self.min_extension_ratio = min_extension_ratio
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        if not metrics.fingers_extended['index']:
            return GestureResult(detected=False,gesture_name='pointing')
    
        other_fingers = ['middle','ring','pinky']
        extended_count = sum(metrics.fingers_extended[f] for f in other_fingers)
        if extended_count > 1: # tolerance of 1(1 extra finger other than index can be opened)
            return GestureResult(detected=False, gesture_name='pointing')
        
        # index finger should be far from palm
        index_tip = metrics.tip_positions['index']
        centroid = metrics.centroid
        distance = euclidean(index_tip, centroid)

        if distance < self.min_extension_ratio:
            return GestureResult(detected=False,gesture_name='pointing')
        
        # Condition 4: Hand should be relatively stable (low velocity)
        speed = np.hypot(metrics.velocity[0], metrics.velocity[1])
        if speed > 0.5:  # TODO: needs tweak threshold
            return GestureResult(detected=False, gesture_name='pointing')

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

class SwipeDetector:
    """
    Detects quick directional hand movements.
    Uses velocity thresholds to determine swipe direction.
    """
    def __init__(
        self,
        velocity_threshold: float = 0.8,  # relative to img_diag per second
        cooldown_s: float = 0.5,
        history_size: int = 8
    ):
        self.velocity_threshold = velocity_threshold
        self.cooldown_s = cooldown_s
        self.history = deque(maxlen=history_size)
        self.last_swipe_time = -999.0
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
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


class ZoomDetector:
    """
    Detects pinch-to-zoom gesture using 3 fingers (thumb, index, middle).
    Tracks spread change to determine zoom in/out.
    """
    def __init__(
        self,
        scale_threshold: float = 0.15,  # 15% change to trigger
        history_size: int = 5
    ):
        self.scale_threshold = scale_threshold
        self.history = deque(maxlen=history_size)
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        """
        Improved zoom detection:
        - Uses weighted distances (thumb distances weighted higher)
        - Tracks continuous direction (zoom in vs out)
        - Works for both starting positions (spread or pinched)
        """
        # Check if thumb, index, and middle are all extended
        required = ['thumb', 'index', 'middle']
        if not all(metrics.fingers_extended[f] for f in required):
            self.history.clear()  # Reset if gesture breaks
            return GestureResult(detected=False, gesture_name='zoom')
        
        # Compute weighted spread (thumb distances matter more!)
        # User insight: thumb-index and thumb-middle change more than index-middle
        thumb_index = metrics.tip_distances.get('index_thumb', 0)
        thumb_middle = metrics.tip_distances.get('thumb_middle', 0)
        index_middle = metrics.tip_distances.get('index_middle', 0)
        
        # Weighted average: thumb distances get 2x weight
        spread = (2 * thumb_index + 2 * thumb_middle + index_middle) / 5.0
        
        # Add to history
        self.history.append(spread)
        
        # Need at least 3 frames to detect continuous direction
        if len(self.history) < 3:
            return GestureResult(detected=False, gesture_name='zoom')
        
        # Check for continuous increasing or decreasing trend
        recent = list(self.history)[-3:]  # Last 3 frames
        
        # Compute differences between consecutive frames
        diff1 = recent[1] - recent[0]
        diff2 = recent[2] - recent[1]
        
        # Check if all differences have same sign (continuous direction)
        increasing = diff1 > 0 and diff2 > 0
        decreasing = diff1 < 0 and diff2 < 0
        
        if not (increasing or decreasing):
            return GestureResult(detected=False, gesture_name='zoom')
        
        # Compute total change magnitude
        total_change = recent[-1] - recent[0]
        relative_change = abs(total_change) / (recent[0] + 1e-6)  # Avoid div by zero
        
        # Detect zoom based on continuous direction and threshold
        if increasing and relative_change > self.scale_threshold:
            zoom_type = 'out'
            detected = True
        elif decreasing and relative_change > self.scale_threshold:
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
                'relative_change': relative_change,
                'spread': spread,
                'trend': 'increasing' if increasing else 'decreasing' if decreasing else 'unstable'
            }
        )


class OpenHandDetector:
    """
    Detects all 5 fingers extended (open palm).
    Used for mode switching or special commands.
    """
    def __init__(self, min_fingers: int = 4):
        """Allow 4 or 5 fingers for tolerance."""
        self.min_fingers = min_fingers
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        """
        Count extended fingers - simplest detector!
        """
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


# ============================================================================
# STEP 8: Gesture Manager (Orchestrates all detectors)
# ============================================================================

class GestureManager:
    """
    Manages multiple gesture detectors and resolves conflicts.
    Maintains per-hand state and history.
    """
    def __init__(self):
        # Initialize all detectors
        self.pinch = PinchDetector(thresh_rel=0.055, hold_frames=5, cooldown_s=0.6)
        self.pointing = PointingDetector(min_extension_ratio=0.12)
        self.swipe = SwipeDetector(velocity_threshold=0.8, cooldown_s=0.5)
        self.zoom = ZoomDetector(scale_threshold=0.15)
        self.open_hand = OpenHandDetector(min_fingers=4)
        
        # State tracking
        self.history = {'left': deque(maxlen=16), 'right': deque(maxlen=16)}
        self.current_gesture = {'left': None, 'right': None}
    
    def process_hand(
        self,
        landmarks,
        img_shape: Tuple[int, int, int],
        hand_label: str = 'right'
    ) -> Dict[str, GestureResult]:
        """
        Process one hand and return all detected gestures.
        
        TODO (STEP 9 - INTEGRATION):
        1. Compute hand metrics
        2. Run all detectors
        3. Apply priority rules (e.g., pinch overrides pointing)
        4. Return dict of detected gestures
        
        Args:
            landmarks: MediaPipe hand landmarks
            img_shape: frame shape
            hand_label: 'left' or 'right'
            
        Returns:
            Dict mapping gesture names to GestureResult objects
        """
        # Get previous metrics for velocity computation
        prev = self.history[hand_label][-1] if self.history[hand_label] else None
        
        # Compute current metrics
        metrics = compute_hand_metrics(landmarks, img_shape, prev)
        self.history[hand_label].append(metrics)
        
        # Run all detectors
        results = {}
        
        # YOUR CODE HERE:
        # Call each detector and collect results
        # Example:
        # pinch_result = self.pinch.detect(metrics)
        # if pinch_result.detected:
        #     results['pinch'] = pinch_result
        
        # Apply conflict resolution (implement priority logic)
        # Example: if pinch is detected, don't report pointing
        
        return results


# ============================================================================
# STEP 10: Testing and Validation Helpers
# ============================================================================

def visualize_hand_metrics(frame, metrics: HandMetrics, color=(0, 255, 0)):
    """
    Draw hand metrics overlays on frame for debugging.
    Shows bounding box, centroid, finger tips, and velocity.
    """
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
        is_extended = metrics.fingers_extended[finger_name]
        finger_color = finger_colors[finger_name] if is_extended else (128, 128, 128)
        cv2.circle(frame, (px, py), 6, finger_color, -1)
        
        # Label the finger
        cv2.putText(frame, finger_name[0].upper(), (px+8, py), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger_color, 1)
    
    # Draw velocity arrow if significant
    vx, vy = metrics.velocity
    speed = np.hypot(vx, vy)
    if speed > 0.1:
        end_x = int((cx + vx * 0.2) * w)
        end_y = int((cy + vy * 0.2) * h)
        cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
    
    # Draw text info at bottom
    info_lines = [
        f"Speed: {speed:.2f}",
        f"Fingers: {sum(metrics.fingers_extended.values())}",
        f"Diag: {metrics.diag_rel:.3f}"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, h - 60 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def test_detector(detector, test_cases: List[Dict]) -> None:
    """
    Run unit tests on a detector with synthetic hand metrics.
    
    Example test case:
    {
        'name': 'pinch_close',
        'metrics': HandMetrics(...),  # with index_thumb distance = 0.04
        'expected': True
    }
    """
    for case in test_cases:
        result = detector.detect(case['metrics'])
        passed = result.detected == case['expected']
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {case['name']}")


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'HandMetrics',
    'GestureResult',
    'compute_hand_metrics',
    'is_finger_extended',
    'PinchDetector',
    'PointingDetector',
    'SwipeDetector',
    'ZoomDetector',
    'OpenHandDetector',
    'GestureManager',
    'visualize_hand_metrics',
    'test_detector',
]
