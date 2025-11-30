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
    prev_metrics: Optional[HandMetrics] = None,
    handedness: str = 'right',
    open_ratio: float = 1.20,
    close_ratio: float = 1.10,
    motion_speed_threshold: float = 0.15,
    motion_sigmoid_k: float = 20.0,
) -> HandMetrics:
    
    # Convert lnadmarks to numpy array
    norm = landmarks_to_array(landmarks.landmark)

    # Compute a more stable palm-centered centroid using wrist + MCP joints
    # (wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp). This is more
    # resilient to finger pose and gives a better reference for extension
    # and pointing calculations. If something goes wrong, fall back to the
    # mean of all landmarks so we never raise an exception here.
    try:
        mcp_indices = [LANDMARK_NAMES['WRIST'], LANDMARK_NAMES['INDEX_MCP'],
                       LANDMARK_NAMES['MIDDLE_MCP'], LANDMARK_NAMES['RING_MCP'],
                       LANDMARK_NAMES['PINKY_MCP']]
        palm_points = norm[mcp_indices, :]
        centroid = (float(palm_points[:, 0].mean()), float(palm_points[:, 1].mean()))
    except Exception:
        # Fallback to global mean to avoid crashing in unusual cases
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
    # bbox is in normalized coords; convert to pixel sizes for diagonal
    bbox_w_px = (xmax - xmin) * w
    bbox_h_px = (ymax - ymin) * h
    hand_diag_px = float(np.hypot(bbox_w_px, bbox_h_px))
    img_diag_px = float(np.hypot(w, h))
    # diag_rel: hand diagonal relative to image diagonal (unitless)
    eps = 1e-6
    diag_rel = hand_diag_px / (img_diag_px + eps)

    # compute velocity early so we can apply motion-aware adjustments
    # Normalize velocity by hand size (hand diagonal) so motion thresholds
    # are independent of how close/far the hand is from the camera.
    velocity = (0.0, 0.0)
    speed = 0.0
    if prev_metrics is not None:
        dt = time.time() - prev_metrics.timestamp
        if dt > 0:
            # centroid is normalized; convert to pixel coords
            cx_px, cy_px = centroid[0] * w, centroid[1] * h
            prev_cx_px, prev_cy_px = prev_metrics.centroid[0] * w, prev_metrics.centroid[1] * h

            dx_px = cx_px - prev_cx_px
            dy_px = cy_px - prev_cy_px

            # velocity in pixels/sec
            vx_px = dx_px / dt
            vy_px = dy_px / dt

            # normalize by hand diagonal (unitless per second)
            norm_factor = hand_diag_px + eps
            vx = vx_px / norm_factor
            vy = vy_px / norm_factor

            velocity = (float(vx), float(vy))
            speed = float(np.hypot(vx, vy))

    # Motion-aware scaling: compute sigmoid of speed and use it to increase
    # the open_ratio when the hand is moving. Sigmoid in (0,1), factor = 1+sigmoid -> (1,2).
    try:
        # Numerical stable sigmoid using numpy
        sig = 1.0 / (1.0 + float(np.exp(-motion_sigmoid_k * (speed - motion_speed_threshold))))
    except Exception:
        sig = 0.0

    motion_factor = 1.0 + sig
    effective_open_ratio = open_ratio * motion_factor

    # detect which finger is extented (use provided hysteresis thresholds)
    finger_extended = {
        'thumb': is_finger_extended(norm, 'thumb', centroid, prev_metrics, diag_rel, handedness, effective_open_ratio, close_ratio),
        'index': is_finger_extended(norm, 'index',  centroid, prev_metrics, diag_rel, handedness, effective_open_ratio, close_ratio),
        'middle': is_finger_extended(norm, 'middle',  centroid, prev_metrics, diag_rel, handedness, effective_open_ratio, close_ratio),
        'ring': is_finger_extended(norm, 'ring',  centroid, prev_metrics, diag_rel, handedness, effective_open_ratio, close_ratio),
        'pinky': is_finger_extended(norm, 'pinky',  centroid, prev_metrics, diag_rel, handedness, effective_open_ratio, close_ratio),
    }
    
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
    palm_centroid: Tuple[float,float],
    prev_metrics: Optional[HandMetrics] = None,
    diag_rel: float = 0.0,
    handedness: str = 'Right',
    open_ratio: float = 1.20,
    close_ratio: float = 1.10,
) -> bool:
    # Map tip/pip indices correctly per finger
    tip_idx_map = {
        'thumb': LANDMARK_NAMES['THUMB_TIP'],
        'index': LANDMARK_NAMES['INDEX_TIP'],
        'middle': LANDMARK_NAMES['MIDDLE_TIP'],
        'ring': LANDMARK_NAMES['RING_TIP'],
        'pinky': LANDMARK_NAMES['PINKY_TIP'],
    }

    pip_idx_map = {
        # for thumb we'll use THUMB_MCP (2) as the proximal joint reference
        'thumb': LANDMARK_NAMES['THUMB_MCP'],
        'index': LANDMARK_NAMES['INDEX_PIP'],
        'middle': LANDMARK_NAMES['MIDDLE_PIP'],
        'ring': LANDMARK_NAMES['RING_PIP'],
        'pinky': LANDMARK_NAMES['PINKY_PIP'],
    }

    # Defensive: ensure requested finger is known
    if finger_name not in tip_idx_map or finger_name not in pip_idx_map:
        return False

    tip_idx = tip_idx_map[finger_name]
    pip_idx = pip_idx_map[finger_name]

    tip_pt = landmarks_norm[tip_idx]
    pip_pt = landmarks_norm[pip_idx]

    # distances to palm centroid (use points, not indices)
    d_tip = euclidean(tip_pt, palm_centroid)
    d_pip = euclidean(pip_pt, palm_centroid)

    # eps to avoid dividing by zero
    eps = 1e-6
    ratio = d_tip / (d_pip + eps)

    # hysteresis thresholds (tune via config)

    prev_state = False
    if prev_metrics is not None and getattr(prev_metrics, 'fingers_extended', None):
        prev_state = bool(prev_metrics.fingers_extended.get(finger_name, False))

    threshold = close_ratio if prev_state else open_ratio

    return (ratio > threshold)


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
        history_size: int = 5,
        finger_gap_threshold: float = 0.19  # Max distance between index and middle
    ):
        self.scale_threshold = scale_threshold
        self.history = deque(maxlen=history_size)
        self.finger_gap_threshold = finger_gap_threshold
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        """
        Improved zoom detection with finger pairing:
        - Index and middle MUST be close together (like a pair)
        - Measures spread between (index+middle pair) and thumb
        - Tracks continuous direction (zoom in vs out)
        - Prevents confusion with pinch (which has thumb+index, not index+middle pair)
        """
        # Check if thumb, index, and middle are all extended
        required = ['thumb', 'index', 'middle']
        if not all(metrics.fingers_extended[f] for f in required):
            self.history.clear()  # Reset if gesture breaks
            return GestureResult(detected=False, gesture_name='zoom')
        
        # NEW: Check if index and middle are close together (paired)
        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        if index_middle_dist > self.finger_gap_threshold:
            # Fingers too far apart - not a valid zoom gesture
            self.history.clear()
            return GestureResult(detected=False, gesture_name='zoom')
        
        # Compute spread: distance between the paired fingers and thumb
        # Use thumb to index as primary measure (middle is close to index anyway)
        thumb_to_pair = metrics.tip_distances.get('index_thumb', 0.0)
        
        # Spread is the distance from thumb to the paired fingers
        spread = thumb_to_pair
        
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
                'finger_gap': index_middle_dist,
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
        Count extended fingers and check they're not pinching.
        Prevents false detection when doing pinch gesture.
        """
        # Count how many fingers are extended
        count = sum(metrics.fingers_extended.values())
        
        # Additional check: if thumb and index are too close, it's a pinch, not open hand
        thumb_index_dist = metrics.tip_distances.get('index_thumb', 0.0)
        is_pinching = thumb_index_dist < 0.08  # If closer than this, it's a pinch
        
        # Detect if at least min_fingers are extended AND not pinching
        detected = count >= self.min_fingers and not is_pinching
        
        return GestureResult(
            detected=detected,
            gesture_name='open_hand',
            confidence=1.0 if detected else 0.0,
            metadata={'finger_count': count}
        )



class ThumbsDetector:
    """
    Detects Thumbs Up/Down and their movements.
    """
    def __init__(self, velocity_threshold: float = 0.2):
        self.velocity_threshold = velocity_threshold

    def detect(self, metrics: HandMetrics) -> GestureResult:
        extended = metrics.fingers_extended
        # Check if only thumb is extended
        only_thumb = extended['thumb'] and not any([extended['index'], extended['middle'], extended['ring'], extended['pinky']])
        
        if not only_thumb:
            return GestureResult(detected=False, gesture_name='none')
            
        thumb_tip_y = metrics.landmarks_norm[4][1]
        pinky_mcp_y = metrics.landmarks_norm[17][1]
        
        is_thumbs_up = thumb_tip_y < pinky_mcp_y
        is_thumbs_down = thumb_tip_y > pinky_mcp_y
        
        vx, vy = metrics.velocity
        
        velocity_up = vy < -self.velocity_threshold
        velocity_down = vy > self.velocity_threshold
        
        gesture_name = 'none'
        detected = False
        
        if is_thumbs_up:
            detected = True
            if velocity_up:
                gesture_name = 'thumbs_up_moving_up'
            elif velocity_down:
                gesture_name = 'thumbs_up_moving_down'
            else:
                gesture_name = 'thumbs_up'
        elif is_thumbs_down:
            detected = True
            if velocity_up:
                gesture_name = 'thumbs_down_moving_up'
            elif velocity_down:
                gesture_name = 'thumbs_down_moving_down'
            else:
                gesture_name = 'thumbs_down'
                
        return GestureResult(
            detected=detected,
            gesture_name=gesture_name,
            confidence=1.0 if detected else 0.0,
            metadata={'velocity': (vx, vy)}
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
        # Initialize all detectors using values from config.json when available
        try:
            print("loading configs")
            from config_manager import get_gesture_threshold

            pinch_thresh = get_gesture_threshold('pinch', 'threshold_rel', default=0.055)
            pinch_hold = get_gesture_threshold('pinch', 'hold_frames', default=5)
            pinch_cd = get_gesture_threshold('pinch', 'cooldown_seconds', default=0.6)

            pointing_min_ext = get_gesture_threshold('pointing', 'min_extension_ratio', default=0.12)

            swipe_thresh = get_gesture_threshold('swipe', 'velocity_threshold', default=0.8)
            swipe_cd = get_gesture_threshold('swipe', 'cooldown_seconds', default=0.5)
            swipe_hist = get_gesture_threshold('swipe', 'history_size', default=8)

            zoom_scale = get_gesture_threshold('zoom', 'scale_threshold', default=0.15)
            zoom_gap = get_gesture_threshold('zoom', 'finger_gap_threshold', default=0.06)
            zoom_hist = get_gesture_threshold('zoom', 'history_size', default=5)

            # Finger extension hysteresis thresholds
            open_ratio = get_gesture_threshold('finger_extension', 'open_ratio', default=1.20)
            close_ratio = get_gesture_threshold('finger_extension', 'close_ratio', default=1.10)
            # Motion parameters for velocity-based scaling
            finger_motion_threshold = get_gesture_threshold('finger_extension', 'motion_speed_threshold', default=0.15)
            finger_motion_sigmoid_k = get_gesture_threshold('finger_extension', 'motion_sigmoid_k', default=20.0)

            open_min = get_gesture_threshold('open_hand', 'min_fingers', default=4)
            
            thumbs_velocity_thresh = get_gesture_threshold('thumbs', 'velocity_threshold', default=0.2)
        except Exception:
            print("Failed to load config file and so falling back to hadcoded value")
            # Fallback to hardcoded defaults if config access fails
            pinch_thresh, pinch_hold, pinch_cd = 0.055, 5, 0.6
            pointing_min_ext = 0.12
            swipe_thresh, swipe_cd, swipe_hist = 0.8, 0.5, 8
            zoom_scale, zoom_gap, zoom_hist = 0.15, 0.06, 5
            open_min = 4
            open_ratio, close_ratio = 1.20, 1.10
            thumbs_velocity_thresh = 0.2

        # Initialize detectors with resolved parameters
        self.pinch = PinchDetector(thresh_rel=pinch_thresh, hold_frames=pinch_hold, cooldown_s=pinch_cd)
        self.pointing = PointingDetector(min_extension_ratio=pointing_min_ext)
        self.swipe = SwipeDetector(velocity_threshold=swipe_thresh, cooldown_s=swipe_cd, history_size=swipe_hist)
        self.zoom = ZoomDetector(scale_threshold=zoom_scale, history_size=zoom_hist, finger_gap_threshold=zoom_gap)
        self.open_hand = OpenHandDetector(min_fingers=open_min)
        self.thumbs = ThumbsDetector(velocity_threshold=thumbs_velocity_thresh)
        # Store finger-extension hysteresis for use during metric computation
        self.finger_open_ratio = open_ratio
        self.finger_close_ratio = close_ratio
        self.finger_motion_threshold = finger_motion_threshold
        self.finger_motion_sigmoid_k = finger_motion_sigmoid_k
        
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
        
        # Compute current metrics with handedness
        metrics = compute_hand_metrics(
            landmarks,
            img_shape,
            prev_metrics=prev,
            handedness=hand_label,
            open_ratio=self.finger_open_ratio,
            close_ratio=self.finger_close_ratio,
            motion_speed_threshold=getattr(self, 'finger_motion_threshold', 0.15),
            motion_sigmoid_k=getattr(self, 'finger_motion_sigmoid_k', 20.0),
        )
        self.history[hand_label].append(metrics)
        
        # Run all detectors
        results = {}
        
        # Call each detector
        pinch_result = self.pinch.detect(metrics)
        pointing_result = self.pointing.detect(metrics)
        swipe_result = self.swipe.detect(metrics)
        zoom_result = self.zoom.detect(metrics)
        thumbs_result = self.thumbs.detect(metrics)
        open_hand_result = self.open_hand.detect(metrics)
        
        # Apply priority rules and conflict resolution:
        # PRIORITY ORDER (highest to lowest):
        # 1. Pinch
        # 2. Zoom
        # 3. Pointing
        # 4. Swipe
        # 5. Open hand
        # Notes:
        # - Pinch/Zoom/Pointing are treated as mutually exclusive (they short-circuit
        #   and return immediately). They may include `swipe` alongside them when
        #   appropriate (swipe coexists in those branches).
        # - Swipe is now considered before Open hand so quick directional motions
        #   won't be shadowed by an open-palm fallback.
        
        # Check high-priority gestures first
        if pinch_result.detected:
            results['pinch'] = pinch_result
            # Pinch blocks pointing and open_hand
            if swipe_result.detected:
                results['swipe'] = swipe_result
            return results
        
        if zoom_result.detected:
            results['zoom'] = zoom_result
            # Zoom blocks other gestures (needs 3 fingers)
            if swipe_result.detected:
                results['swipe'] = swipe_result
            return results
        
        if pointing_result.detected:
            results['pointing'] = pointing_result
            # Pointing blocks open_hand (only 1 finger vs 5)
            if swipe_result.detected:
                results['swipe'] = swipe_result
            return results
            
        if thumbs_result.detected:
            results['thumbs'] = thumbs_result
            if swipe_result.detected:
                results['swipe'] = swipe_result
            return results

        # Require all five fingers to be extended for a swipe to be valid.
        # This prevents accidental swipes when the user has a closed or partially
        # open hand. Use strict all-open requirement per user request.
        
        # Only check open_hand if no other higher-priority gesture detected
        if open_hand_result.detected:

            if swipe_result.detected:
                finger_count = sum(metrics.fingers_extended.values())
                all_open = (finger_count == 5)
                if all_open:
                    results['swipe'] = swipe_result
            else:
                results['open_hand'] = open_hand_result

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
