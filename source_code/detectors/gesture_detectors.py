import numpy as np
import time
import cv2
from collections import deque
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from source_code.utils.math_utils import landmarks_to_array, euclidean, EWMA, ClickDetector


# Data Structures

@dataclass
class HandMetrics:
    """
    Per-frame computed metrics for a single hand.
    All spatial values are in normalized coordinates (0..1) unless noted.
    """
    # Raw and derived landmark data
    landmarks_norm: np.ndarray  # shape (21, 2), all landmarks in normalized coords
    timestamp: float            
    
    # Spatial metrics (normalized)
    centroid: Tuple[float, float]          # geometric center of hand
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    
    # Tip positions (normalized). easier to access than indexing landmarks_norm
    # {'thumb': (x,y), 'index': (x,y), 'middle': (x,y), 'ring': (x,y), 'pinky': (x,y)}
    tip_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Distances between key points (normalized)
    tip_distances: Dict[str, float] = field(default_factory=dict)
    
    # Finger extension state
    fingers_extended: Dict[str, bool] = field(default_factory=dict)
    
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

    # Compute stable palm-centered centroid using wrist + MCP joints
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
    # diag_rel: hand diagonal relative to image diagonal
    eps = 1e-6
    diag_rel = hand_diag_px / (img_diag_px + eps)

    velocity = (0.0, 0.0)
    speed = 0.0
    if prev_metrics is not None:
        dt = time.time() - prev_metrics.timestamp
        if dt > 0:
            # convert to pixel coords
            cx_px, cy_px = centroid[0] * w, centroid[1] * h
            prev_cx_px, prev_cy_px = prev_metrics.centroid[0] * w, prev_metrics.centroid[1] * h

            dx_px = cx_px - prev_cx_px
            dy_px = cy_px - prev_cy_px

            # velocity in pixels/sec
            vx_px = dx_px / dt
            vy_px = dy_px / dt

            # normalize by hand diagonal
            norm_factor = hand_diag_px + eps
            vx = vx_px / norm_factor
            vy = vy_px / norm_factor

            velocity = (float(vx), float(vy))
            speed = float(np.hypot(vx, vy))

    try:
        sig = 1.0 / (1.0 + float(np.exp(-motion_sigmoid_k * (speed - motion_speed_threshold))))
    except Exception:
        sig = 0.0

    motion_factor = 1.0 + sig
    effective_open_ratio = open_ratio * motion_factor

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

    # ensures requested finger is known
    if finger_name not in tip_idx_map or finger_name not in pip_idx_map:
        return False

    tip_idx = tip_idx_map[finger_name]
    pip_idx = pip_idx_map[finger_name]

    tip_pt = landmarks_norm[tip_idx]
    pip_pt = landmarks_norm[pip_idx]

    # distances to palm centroid
    d_tip = euclidean(tip_pt, palm_centroid)
    d_pip = euclidean(pip_pt, palm_centroid)

    # eps to avoid dividing by zero
    eps = 1e-6
    ratio = d_tip / (d_pip + eps)

    # hysteresis thresholds

    prev_state = False
    if prev_metrics is not None and getattr(prev_metrics, 'fingers_extended', None):
        prev_state = bool(prev_metrics.fingers_extended.get(finger_name, False))

    threshold = close_ratio if prev_state else open_ratio

    return (ratio > threshold)


class PinchDetector:

    def __init__(self, thresh_rel: float = 0.055, hold_frames: int = 5, cooldown_s: float = 0.6):
        self.thresh_rel = thresh_rel
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self._count = 0
        self._last_time = -999.0
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
       
        dist_rel = metrics.tip_distances['index_thumb']
        now = time.time()
        in_cooldown = now - self._last_time < self.cooldown_s
        
        # Always return full metadata for visual feedback
        base_metadata = {
            'dist_rel': dist_rel,
            'threshold': self.thresh_rel,
            'hold_count': self._count,
            'hold_frames_needed': self.hold_frames,
            'in_cooldown': in_cooldown,
            'cooldown_remaining': max(0.0, self.cooldown_s - (now - self._last_time))
        }
        
        if in_cooldown:
            return GestureResult(detected=False, gesture_name='pinch', metadata=base_metadata)

        if dist_rel <= self.thresh_rel:
            self._count += 1
            base_metadata['hold_count'] = self._count
            if self._count >= self.hold_frames:
                self._last_time = now
                self._count = 0
                base_metadata['hold_count'] = 0
                base_metadata['reason'] = 'pinch_completed'
                return GestureResult(
                    detected=True,
                    gesture_name='pinch',
                    confidence=1.0,
                    metadata=base_metadata
                )
            else:
                base_metadata['reason'] = 'holding'
        else:
            self._count = 0
            base_metadata['hold_count'] = 0
            base_metadata['reason'] = 'fingers_too_far'

        return GestureResult(detected=False, gesture_name='pinch', metadata=base_metadata)


class PointingDetector:
    def __init__(
            self, 
            min_extension_ratio: float = 0.12, 
            max_extra_fingers: int = 1, 
            max_speed: float = 0.5
        ):
        self.min_extension_ratio = float(min_extension_ratio)
        self.max_extra_fingers = int(max_extra_fingers)
        self.max_speed = float(max_speed)

    def detect(self, metrics: HandMetrics) -> GestureResult:
        index_tip = metrics.tip_positions['index']
        centroid = metrics.centroid
        distance = euclidean(index_tip, centroid)
        speed = float(np.hypot(metrics.velocity[0], metrics.velocity[1]))
        direction = (index_tip[0] - centroid[0], index_tip[1] - centroid[1])
        
        index_extended = metrics.fingers_extended.get('index', False)
        other_fingers = ['middle', 'ring', 'pinky']
        extended_count = sum(1 for f in other_fingers if metrics.fingers_extended.get(f, False))
        
        # Always return full metadata for visual feedback
        base_metadata = {
            'tip_position': index_tip,
            'direction': direction,
            'distance': distance,
            'min_extension_ratio': self.min_extension_ratio,
            'speed': speed,
            'max_speed': self.max_speed,
            'index_extended': index_extended,
            'extra_fingers_count': extended_count,
            'max_extra_fingers': self.max_extra_fingers,
            'reason': None
        }
        
        if not index_extended:
            base_metadata['reason'] = 'index_not_extended'
            return GestureResult(detected=False, gesture_name='pointing', metadata=base_metadata)

        if extended_count > self.max_extra_fingers:
            base_metadata['reason'] = 'too_many_fingers'
            return GestureResult(detected=False, gesture_name='pointing', metadata=base_metadata)

        if distance < self.min_extension_ratio:
            base_metadata['reason'] = 'too_close_to_palm'
            return GestureResult(detected=False, gesture_name='pointing', metadata=base_metadata)

        if speed > self.max_speed:
            base_metadata['reason'] = 'moving_too_fast'
            return GestureResult(detected=False, gesture_name='pointing', metadata=base_metadata)

        base_metadata['reason'] = 'pointing_detected'
        return GestureResult(
            detected=True,
            gesture_name='pointing',
            confidence=1.0,
            metadata=base_metadata
        )

class SwipeDetector:
    """
    Detects continuous directional hand movements with axis-specific tuning.
    """
    def __init__(
        self,
        ewma_alpha: float = 0.3,
        velocity_threshold_x: float = 0.4,
        velocity_threshold_y: float = 0.4,
        confidence_ramp_up: float = 0.25,
        confidence_decay: float = 0.15,
        confidence_threshold: float = 0.6,
        max_velocity_x: float = 2.0,
        max_velocity_y: float = 2.0
    ):
        self.ewma_alpha = ewma_alpha
        self.velocity_threshold_x = velocity_threshold_x
        self.velocity_threshold_y = velocity_threshold_y
        self.confidence_ramp_up = confidence_ramp_up
        self.confidence_decay = confidence_decay
        self.confidence_threshold = confidence_threshold
        self.max_velocity_x = max_velocity_x
        self.max_velocity_y = max_velocity_y
        
        self.confidence = 0.0
        self.ewma_velocity = EWMA(alpha=ewma_alpha)
        self.current_direction: Optional[str] = None
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        vx, vy = metrics.velocity
        
        smoothed = self.ewma_velocity.update([vx, vy])
        ewma_vx = float(smoothed[0])
        ewma_vy = float(smoothed[1])
        
        if abs(ewma_vx) > abs(ewma_vy):
            detected_direction = 'right' if ewma_vx > 0 else 'left'
            primary_velocity = abs(ewma_vx)
            velocity_threshold = self.velocity_threshold_x
            max_velocity = self.max_velocity_x
        else:
            detected_direction = 'down' if ewma_vy > 0 else 'up'
            primary_velocity = abs(ewma_vy)
            velocity_threshold = self.velocity_threshold_y
            max_velocity = self.max_velocity_y
        
        base_metadata = {
            'direction': detected_direction,
            'current_direction': self.current_direction,
            'raw_velocity': (vx, vy),
            'ewma_velocity': (ewma_vx, ewma_vy),
            'velocity_threshold_x': self.velocity_threshold_x,
            'velocity_threshold_y': self.velocity_threshold_y,
            'confidence': self.confidence,
            'confidence_threshold': self.confidence_threshold,
            'reason': None
        }
        
        if primary_velocity > max_velocity:
            self.confidence = max(0.0, self.confidence - self.confidence_decay)
            base_metadata['reason'] = 'velocity_too_high'
            base_metadata['confidence'] = self.confidence
            
            if self.confidence == 0.0:
                self.current_direction = None
            
            return GestureResult(
                detected=(self.confidence >= self.confidence_threshold),
                gesture_name='swipe',
                confidence=self.confidence,
                metadata=base_metadata
            )
        
        if primary_velocity > velocity_threshold:
            if self.current_direction is None:
                self.current_direction = detected_direction
                self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
                base_metadata['reason'] = 'swipe_started'
            elif self.current_direction == detected_direction:
                self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
                base_metadata['reason'] = 'swipe_sustained'
            else:
                self.confidence = max(0.0, self.confidence - self.confidence_decay)
                base_metadata['reason'] = 'direction_changed'
                
                if self.confidence == 0.0:
                    self.current_direction = detected_direction
                    self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
                    base_metadata['reason'] = 'swipe_direction_switched'
        else:
            self.confidence = max(0.0, self.confidence - self.confidence_decay)
            base_metadata['reason'] = 'movement_stopped'
            
            if self.confidence == 0.0:
                self.current_direction = None
        
        base_metadata['current_direction'] = self.current_direction
        base_metadata['confidence'] = self.confidence
        
        is_detected = self.confidence >= self.confidence_threshold
        
        return GestureResult(
            detected=is_detected,
            gesture_name='swipe',
            confidence=self.confidence,
            metadata=base_metadata
        )


class ZoomDetector:
    """
    Detects pinch-to-zoom using three fingers (thumb, index, middle).
    Tracks spread change between fingers with smoothing and confidence decay.
    """
    def __init__(
        self,
        finger_gap_threshold: float = 0.10,  # Max distance between index and middle
        ewma_alpha: float = 0.3,  # EWMA smoothing factor (higher=more responsive)
        velocity_threshold: float = 0.08,  # Min EWMA velocity to gain confidence
        confidence_ramp_up: float = 0.25,  # Confidence increase per valid frame
        confidence_decay: float = 0.15,  # Confidence decrease per invalid frame
        confidence_threshold: float = 0.6,  # Min confidence to report detected
        max_velocity: float = 1.5,  # Maximum spread change velocity (filters jumps)
        require_fingers_extended: bool = False  # Whether to require finger extension
    ):
        self.finger_gap_threshold = finger_gap_threshold
        self.ewma_alpha = ewma_alpha
        self.velocity_threshold = velocity_threshold
        self.confidence_ramp_up = confidence_ramp_up
        self.confidence_decay = confidence_decay
        self.confidence_threshold = confidence_threshold
        self.max_velocity = max_velocity
        self.require_fingers_extended = require_fingers_extended
        
        self.confidence = 0.0
        self.prev_spread: Optional[float] = None
        self.ewma_velocity = EWMA(alpha=ewma_alpha)
        self.current_direction: Optional[str] = None
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        
        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        spread = metrics.tip_distances.get('index_thumb', 0.0)
        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        spread = metrics.tip_distances.get('index_thumb', 0.0)

        base_metadata = {
            'spread': spread,
            'finger_gap': index_middle_dist,
            'finger_gap_threshold': self.finger_gap_threshold,
            'ewma_velocity': float(self.ewma_velocity.value) if self.ewma_velocity.value is not None else 0.0,
            'velocity_threshold': self.velocity_threshold,
            'confidence': self.confidence,
            'confidence_threshold': self.confidence_threshold,
            'direction': self.current_direction,
            'reason': None
        }

        if index_middle_dist > self.finger_gap_threshold:
            self.confidence = max(0.0, self.confidence - self.confidence_decay)
            base_metadata['reason'] = 'fingers_not_paired'
            
            if self.confidence == 0.0:
                self.prev_spread = None
                self.ewma_velocity = EWMA(alpha=self.ewma_alpha)
                self.current_direction = None
            
            return GestureResult(
                detected=(self.confidence >= self.confidence_threshold),
                gesture_name='zoom',
                confidence=self.confidence,
                metadata=base_metadata
            )
        
        if self.prev_spread is None:
            self.prev_spread = spread
            base_metadata['reason'] = 'initializing'
            return GestureResult(
                detected=False,
                gesture_name='zoom',
                confidence=self.confidence,
                metadata=base_metadata
            )
        
        delta_spread = spread - self.prev_spread
        self.prev_spread = spread
        
        smoothed = self.ewma_velocity.update([delta_spread])
        ewma_vel = float(smoothed[0])
        
        base_metadata['ewma_velocity'] = ewma_vel
        base_metadata['delta_spread'] = delta_spread
        
        abs_velocity = abs(ewma_vel)
        if abs_velocity > self.max_velocity:
            self.confidence = max(0.0, self.confidence - self.confidence_decay)
            base_metadata['reason'] = 'velocity_too_high'
            
            if self.confidence == 0.0:
                self.current_direction = None
            
            return GestureResult(
                detected=(self.confidence >= self.confidence_threshold),
                gesture_name='zoom',
                confidence=self.confidence,
                metadata=base_metadata
            )
        
        eps = 1e-6
        if ewma_vel < -self.velocity_threshold:
            detected_direction = 'out'
        elif ewma_vel > self.velocity_threshold:
            detected_direction = 'in'
        else:
            detected_direction = None
        
        if detected_direction is not None:
            if self.current_direction is None:
                self.current_direction = detected_direction
                self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
                base_metadata['reason'] = 'direction_started'
            elif self.current_direction == detected_direction:
                self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
                base_metadata['reason'] = 'direction_sustained'
            else:
                self.confidence = max(0.0, self.confidence - self.confidence_decay)
                base_metadata['reason'] = 'direction_reversed'
                
                if self.confidence == 0.0:
                    self.current_direction = detected_direction
                    self.confidence = min(1.0, self.confidence + self.confidence_ramp_up)
        else:
            self.confidence = max(0.0, self.confidence - self.confidence_decay)
            base_metadata['reason'] = 'movement_stopped'
            
            if self.confidence == 0.0:
                self.current_direction = None
        
        base_metadata['direction'] = self.current_direction
        base_metadata['confidence'] = self.confidence
        
        is_detected = self.confidence >= self.confidence_threshold
        
        # Update reason if detected
        if is_detected and base_metadata['reason'] not in ['direction_started', 'direction_sustained']:
            base_metadata['reason'] = 'zoom_active'
        
        return GestureResult(
            detected=is_detected,
            gesture_name='zoom',
            confidence=self.confidence,
            metadata=base_metadata
        )


class OpenHandDetector:
    """
    Detects all 5 fingers extended (open palm).
    """
    def __init__(self, min_fingers: int = 4, pinch_threshold: float = 0.08):
        self.min_fingers = int(min_fingers)
        self.pinch_threshold = float(pinch_threshold)
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        # Count how many fingers are extended
        count = sum(metrics.fingers_extended.values())
        
        # if thumb and index are too close, it's a pinch
        thumb_index_dist = metrics.tip_distances.get('index_thumb', 0.0)
        is_pinching = thumb_index_dist < float(self.pinch_threshold)
        
        # Detect if at least min_fingers are extended AND not pinching
        detected = count >= self.min_fingers and not is_pinching
        
        # Always return full metadata for visual feedback
        if detected:
            reason = 'open_hand_detected'
        elif is_pinching:
            reason = 'pinching'
        else:
            reason = 'not_enough_fingers'
        
        base_metadata = {
            'finger_count': count,
            'min_fingers': self.min_fingers,
            'thumb_index_dist': thumb_index_dist,
            'pinch_threshold': self.pinch_threshold,
            'is_pinching': is_pinching,
            'fingers_extended': dict(metrics.fingers_extended),
            'reason': reason
        }
        
        return GestureResult(
            detected=detected,
            gesture_name='open_hand',
            confidence=1.0 if detected else 0.0,
            metadata=base_metadata
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

class GestureManager:
    """
    Manages multiple gesture detectors and resolves conflicts.
    Maintains per-hand state and history.
    """
    def __init__(self):
        # Initialize all detectors using values from config.json when available
        try:
            print("loading configs")
            from source_code.config.config_manager import get_gesture_threshold

            pinch_thresh = get_gesture_threshold('pinch', 'threshold_rel', default=0.055)
            pinch_hold = get_gesture_threshold('pinch', 'hold_frames', default=5)
            pinch_cd = get_gesture_threshold('pinch', 'cooldown_seconds', default=0.6)

            pointing_min_ext = get_gesture_threshold('pointing', 'min_extension_ratio', default=0.12)
            pointing_max_extra = get_gesture_threshold('pointing', 'max_extra_fingers', default=1)
            pointing_max_speed = get_gesture_threshold('pointing', 'max_speed', default=0.5)

            swipe_ewma_alpha = get_gesture_threshold('swipe', 'ewma_alpha', default=0.3)
            swipe_vel_thresh_x = get_gesture_threshold('swipe', 'velocity_threshold_x', default=0.4)
            swipe_vel_thresh_y = get_gesture_threshold('swipe', 'velocity_threshold_y', default=0.4)
            swipe_conf_ramp = get_gesture_threshold('swipe', 'confidence_ramp_up', default=0.25)
            swipe_conf_decay = get_gesture_threshold('swipe', 'confidence_decay', default=0.15)
            swipe_conf_thresh = get_gesture_threshold('swipe', 'confidence_threshold', default=0.6)
            swipe_max_vel_x = get_gesture_threshold('swipe', 'max_velocity_x', default=2.0)
            swipe_max_vel_y = get_gesture_threshold('swipe', 'max_velocity_y', default=2.0)

            # New EWMA-based zoom parameters
            zoom_gap = get_gesture_threshold('zoom', 'finger_gap_threshold', default=0.10)
            zoom_ewma_alpha = get_gesture_threshold('zoom', 'ewma_alpha', default=0.3)
            zoom_vel_thresh = get_gesture_threshold('zoom', 'velocity_threshold', default=0.08)
            zoom_conf_ramp = get_gesture_threshold('zoom', 'confidence_ramp_up', default=0.25)
            zoom_conf_decay = get_gesture_threshold('zoom', 'confidence_decay', default=0.15)
            zoom_conf_thresh = get_gesture_threshold('zoom', 'confidence_threshold', default=0.6)
            zoom_max_vel = get_gesture_threshold('zoom', 'max_velocity', default=1.5)
            zoom_require_ext = get_gesture_threshold('zoom', 'require_fingers_extended', default=False)

            # Finger extension hysteresis thresholds
            open_ratio = get_gesture_threshold('finger_extension', 'open_ratio', default=1.20)
            close_ratio = get_gesture_threshold('finger_extension', 'close_ratio', default=1.10)
            # Motion parameters for velocity-based scaling
            finger_motion_threshold = get_gesture_threshold('finger_extension', 'motion_speed_threshold', default=0.15)
            finger_motion_sigmoid_k = get_gesture_threshold('finger_extension', 'motion_sigmoid_k', default=20.0)

            open_min = get_gesture_threshold('open_hand', 'min_fingers', default=4)
            open_pinch_exclusion = get_gesture_threshold('open_hand', 'pinch_exclusion_distance', default=0.08)
            
            thumbs_velocity_thresh = get_gesture_threshold('thumbs', 'velocity_threshold', default=0.2)
        except Exception:
            print("Failed to load config file and so falling back to hadcoded value")
            # Fallback to hardcoded defaults if config access fails
            pinch_thresh, pinch_hold, pinch_cd = 0.055, 5, 0.6
            pointing_min_ext = 0.12
            pointing_max_extra = 1
            pointing_max_speed = 0.5
            swipe_ewma_alpha = 0.3
            swipe_vel_thresh_x = 0.4
            swipe_vel_thresh_y = 0.4
            swipe_conf_ramp = 0.25
            swipe_conf_decay = 0.15
            swipe_conf_thresh = 0.6
            swipe_max_vel_x = 2.0
            swipe_max_vel_y = 2.0
            # New EWMA-based zoom fallbacks
            zoom_gap = 0.10
            zoom_ewma_alpha = 0.3
            zoom_vel_thresh = 0.08
            zoom_conf_ramp = 0.25
            zoom_conf_decay = 0.15
            zoom_conf_thresh = 0.6
            zoom_max_vel = 1.5
            zoom_require_ext = False
            open_min = 4
            open_pinch_exclusion = 0.08
            open_ratio, close_ratio = 1.20, 1.10
            finger_motion_threshold = 0.15
            finger_motion_sigmoid_k = 20.0
            thumbs_velocity_thresh = 0.2

        # Initialize detectors with resolved parameters
        self.pinch = PinchDetector(thresh_rel=pinch_thresh, hold_frames=pinch_hold, cooldown_s=pinch_cd)
        self.pointing = PointingDetector(min_extension_ratio=pointing_min_ext, max_extra_fingers=pointing_max_extra, max_speed=pointing_max_speed)
        self.swipe = SwipeDetector(
            ewma_alpha=swipe_ewma_alpha,
            velocity_threshold_x=swipe_vel_thresh_x,
            velocity_threshold_y=swipe_vel_thresh_y,
            confidence_ramp_up=swipe_conf_ramp,
            confidence_decay=swipe_conf_decay,
            confidence_threshold=swipe_conf_thresh,
            max_velocity_x=swipe_max_vel_x,
            max_velocity_y=swipe_max_vel_y
        )
        self.zoom = ZoomDetector(
            finger_gap_threshold=zoom_gap,
            ewma_alpha=zoom_ewma_alpha,
            velocity_threshold=zoom_vel_thresh,
            confidence_ramp_up=zoom_conf_ramp,
            confidence_decay=zoom_conf_decay,
            confidence_threshold=zoom_conf_thresh,
            max_velocity=zoom_max_vel,
            require_fingers_extended=zoom_require_ext
        )
        self.open_hand = OpenHandDetector(min_fingers=open_min, pinch_threshold=open_pinch_exclusion)
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
        
        # Always expose all detector results as metadata previews for visual feedback
        # under reserved keys so visualizers can display tuning info even when gestures
        # are not reported as active.
        results['__pinch_meta'] = pinch_result
        results['__pointing_meta'] = pointing_result
        results['__swipe_meta'] = swipe_result
        results['__zoom_meta'] = zoom_result
        results['__thumbs_meta'] = thumbs_result
        results['__open_hand_meta'] = open_hand_result

        # Priority rules and conflict resolution:
        # PRIORITY ORDER (highest to lowest):
        # 1. Thumbs
        # 2. Zoom
        # 3. Pinch
        # 4. Pointing
        # 5. Swipe
        # 6. Open hand
        
        # Check high-priority gestures first
        if thumbs_result.detected:
            results['thumbs'] = thumbs_result
            # if swipe_result.detected:
            #     results['swipe'] = swipe_result
            return results
        
        if zoom_result.detected:
            results['zoom'] = zoom_result
            # if swipe_result.detected:
            #     results['swipe'] = swipe_result
            return results
        
        if pinch_result.detected:
            results['pinch'] = pinch_result
            # if swipe_result.detected:
            #     results['swipe'] = swipe_result
            return results
        
        if pointing_result.detected:
            results['pointing'] = pointing_result
            # Pointing blocks open_hand (only 1 finger vs 5)
            # if swipe_result.detected:
            #     results['swipe'] = swipe_result
            return results
            
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
