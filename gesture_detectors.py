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
from collections import deque
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from math_utils import landmarks_to_array, euclidean, EWMA, ClickDetector


# ============================================================================
# Data Structures
# ============================================================================

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


# ============================================================================
# STEP 1: Compute Hand Metrics (Foundation for all detectors)
# ============================================================================

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
    """
    Compute comprehensive hand metrics from MediaPipe landmarks.
    
    TODO (STEP 1 - YOU IMPLEMENT THIS):
    1. Use landmarks_to_array() from math_utils to convert landmarks to np.ndarray
    2. Compute centroid as mean of all landmarks
    3. Compute bbox as (min_x, min_y, max_x, max_y)
    4. Extract tip positions for each finger (indices: thumb=4, index=8, middle=12, ring=16, pinky=20)
    5. Compute distances between key tips (use euclidean from math_utils)
    6. Compute hand diagonal relative to image diagonal
    7. Detect which fingers are extended (call is_finger_extended for each)
    8. Compute velocity if prev_metrics exists
    
    Args:
        landmarks: MediaPipe hand landmarks object
        img_shape: frame shape (height, width, channels)
        prev_metrics: previous frame's metrics for velocity computation
        
    Returns:
        HandMetrics object with all computed values
        
    Example skeleton:
        norm = landmarks_to_array(landmarks.landmark)
        centroid = (float(norm[:, 0].mean()), float(norm[:, 1].mean()))
        # ... continue implementation
    """
    # YOUR CODE HERE - follow the TODO steps above
    # Hint: Start by converting landmarks and computing simple metrics first
    # Then add finger detection and distances
    
    pass  # Remove this and implement


def is_finger_extended(
    landmarks_norm: np.ndarray,
    finger_name: str,
    handedness: str = 'Right'
) -> bool:
    """
    Detect if a specific finger is extended.
    
    TODO (STEP 2 - YOU IMPLEMENT THIS):
    Rule for most fingers: tip.y < pip.y (remember y increases downward in image coords)
    Rule for thumb: more complex - compare x-positions and angle
    
    Args:
        landmarks_norm: (21, 2) array of normalized landmarks
        finger_name: 'thumb', 'index', 'middle', 'ring', or 'pinky'
        handedness: 'Right' or 'Left' (affects thumb detection)
        
    Returns:
        True if finger is extended, False otherwise
        
    Hints:
    - For index: compare landmarks_norm[8][1] < landmarks_norm[6][1]
    - For thumb: check if tip is far from palm center in x-direction
    - Use LANDMARK_NAMES dict to get indices
    """
    # YOUR CODE HERE
    # Start with index finger (simplest case)
    # Then add other fingers
    # Thumb is the trickiest - implement last
    
    pass  # Remove this and implement


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
        """
        TODO (STEP 3A - ADAPT EXISTING CODE):
        Take the logic from ClickDetector.pinched() and adapt it here.
        Use metrics.tip_distances['index_thumb'] instead of passing dist_rel directly.
        
        Returns:
            GestureResult with detected=True when pinch occurs
        """
        # YOUR CODE HERE - copy and adapt from ClickDetector
        pass


class PointingDetector:
    """
    Detects single index finger pointing gesture.
    Used for cursor control.
    """
    def __init__(self, min_extension_ratio: float = 0.12):
        """
        Args:
            min_extension_ratio: minimum distance from palm to index tip (relative to img diag)
        """
        self.min_extension_ratio = min_extension_ratio
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        """
        TODO (STEP 4 - YOU IMPLEMENT THIS):
        
        Conditions for pointing:
        1. Index finger is extended (metrics.fingers_extended['index'] == True)
        2. All other fingers are NOT extended (or at most 1 other for tolerance)
        3. Index tip is sufficiently far from palm (use centroid)
        4. Hand is relatively stable (low velocity)
        
        Returns:
            GestureResult with metadata containing:
            - 'direction': normalized vector from palm to index tip
            - 'tip_position': (x, y) normalized coords of index tip
        """
        # YOUR CODE HERE
        # Start with condition 1 and 2
        # Then add distance check
        # Finally add velocity check for stability
        
        pass


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
        """
        TODO (STEP 5 - YOU IMPLEMENT THIS):
        
        1. Add current metrics to history deque
        2. Check if cooldown has passed
        3. Compute velocity from history (use EWMA or simple average)
        4. If |vx| or |vy| > threshold, determine direction
        5. Return result with direction metadata
        
        Velocity computation hint:
        - Get last 3-5 frames from history
        - Compute centroid displacement
        - Divide by time difference
        - Compare to threshold
        """
        # YOUR CODE HERE
        pass


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
        TODO (STEP 6 - YOU IMPLEMENT THIS):
        
        1. Check if thumb, index, and middle are all extended
        2. Compute current "spread" = average distance between the 3 tips
        3. Compare to previous spread (from history)
        4. If ratio > 1 + threshold -> zoom out
        5. If ratio < 1 - threshold -> zoom in
        
        Hint: Use metrics.tip_distances to get pairwise distances
        Compute spread as mean of (thumb-index, thumb-middle, index-middle)
        """
        # YOUR CODE HERE
        pass


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
        TODO (STEP 7 - EASY ONE TO START WITH):
        
        Simply count how many fingers are extended.
        If count >= self.min_fingers, return detected=True
        
        Hint: sum(metrics.fingers_extended.values())
        """
        # YOUR CODE HERE - this is the simplest detector, start here!
        pass


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
    
    TODO (STEP 10 - VISUALIZATION):
    1. Draw bounding box
    2. Draw centroid as circle
    3. Draw finger tips with different colors based on extended state
    4. Draw text showing distances and velocity
    
    This helps you see what the detectors are "seeing"
    """
    # YOUR CODE HERE
    pass


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
