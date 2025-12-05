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
    
    # Calculate hand bounding box dimensions for normalization
    h, w = img_shape[0], img_shape[1]
    bbox_w_px = (xmax - xmin) * w
    bbox_h_px = (ymax - ymin) * h
    hand_diag_px = float(np.hypot(bbox_w_px, bbox_h_px))
    img_diag_px = float(np.hypot(w, h))
    eps = 1e-6
    diag_rel = hand_diag_px / (img_diag_px + eps)
    
    # Extract finger tips positions
    tip_positions = {
        'thumb' : tuple(norm[4]),
        'index' : tuple(norm[8]),
        'middle' : tuple(norm[12]),
        'ring' : tuple(norm[16]),
        'pinky' : tuple(norm[20]),
    }

    # Distance between tips - normalized by hand bounding box diagonal
    # Since euclidean gives distance in normalized image coords (0-1),
    # and diag_rel = hand_diag / img_diag, dividing by diag_rel gives distance relative to hand size
    tip_distances = {
        'index_thumb' : float(euclidean(norm[4],norm[8])) / (diag_rel + eps),
        'index_middle' : float(euclidean(norm[8],norm[12])) / (diag_rel + eps),
        'index_ring' : float(euclidean(norm[8],norm[16])) / (diag_rel + eps),
        'index_pinky' : float(euclidean(norm[8],norm[20])) / (diag_rel + eps),
        'thumb_middle' : float(euclidean(norm[4],norm[12])) / (diag_rel + eps),
        'thumb_ring' : float(euclidean(norm[4],norm[16])) / (diag_rel + eps),
        'thumb_pinky' : float(euclidean(norm[4],norm[20])) / (diag_rel + eps),
        'middle_ring' : float(euclidean(norm[12],norm[16])) / (diag_rel + eps),
        'middle_pinky' : float(euclidean(norm[12],norm[20])) / (diag_rel + eps),
        'ring_pinky' : float(euclidean(norm[16],norm[20])) / (diag_rel + eps)
    }

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
    MAX_PENALTY_ABS = (float)(1.5 - open_ratio) # As human finger can have max ratio of 1.5
    motion_penalty_abs =  sig * MAX_PENALTY_ABS
    effective_open_ratio = open_ratio + motion_penalty_abs

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
    wrist_idx = LANDMARK_NAMES['WRIST']
    # ensures requested finger is known
    if finger_name not in tip_idx_map or finger_name not in pip_idx_map:
        return False

    tip_idx = tip_idx_map[finger_name]
    pip_idx = pip_idx_map[finger_name]

    tip_pt = landmarks_norm[tip_idx]
    pip_pt = landmarks_norm[pip_idx]
    
    wrist_pt = landmarks_norm[wrist_idx]

    # distances to palm centroid - already in normalized coords, no further normalization needed
    # since we're computing a ratio, the normalization cancels out
    d_tip = euclidean(tip_pt, palm_centroid)
    d_pip = euclidean(pip_pt, palm_centroid)
    d_tip_wrist = euclidean(tip_pt, wrist_pt)
    d_pip_wrist = euclidean(pip_pt,wrist_pt)
    # eps to avoid dividing by zero
    eps = 1e-6
    ratio1 = d_tip / (d_pip + eps)
    ratio2 = d_tip_wrist / (d_pip_wrist + eps)
    # hysteresis thresholds

    prev_state = False
    if prev_metrics is not None and getattr(prev_metrics, 'fingers_extended', None):
        prev_state = bool(prev_metrics.fingers_extended.get(finger_name, False))

    
    threshold = close_ratio if prev_state else open_ratio
    
    return (ratio1 > threshold and ratio2 > threshold)


class PinchDetector:
    """
    Detects pinch gesture (thumb and index finger close together).
    Maintains per-hand state to support two-hand detection without interference.
    """

    def __init__(self, thresh_rel: float = 0.055, hold_frames: int = 5, cooldown_s: float = 0.6,index_middle_gap_threshold = 0.2):
        self.thresh_rel = thresh_rel
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self.index_middle_gap_threshold = index_middle_gap_threshold
        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }
    
    def _create_hand_state(self):
        return {'count': 0, 'last_time': -999.0}
    
    def _get_state(self, hand_label: str):
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]
    
    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        state = self._get_state(hand_label)
       
        dist_rel = metrics.tip_distances['index_thumb']
        now = time.time()
        in_cooldown = now - state['last_time'] < self.cooldown_s

        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        # Always return full metadata for visual feedback
        base_metadata = {
            'dist_rel': dist_rel,
            'threshold': self.thresh_rel,
            'hold_count': state['count'],
            'hold_frames_needed': self.hold_frames,
            'in_cooldown': in_cooldown,
            'cooldown_remaining': max(0.0, self.cooldown_s - (now - state['last_time'])),
            'hand_label': hand_label,
            'indx_mdl_dist' : index_middle_dist
        }
        
        if in_cooldown:
            return GestureResult(detected=False, gesture_name='pinch', metadata=base_metadata)
        if index_middle_dist <= self.index_middle_gap_threshold:
            base_metadata['reason'] = "index is too close to middle finger"
            return GestureResult(detected=False,gesture_name='pinch',metadata=base_metadata)
        if dist_rel <= self.thresh_rel:
            state['count'] += 1
            base_metadata['hold_count'] = state['count']
            if state['count'] >= self.hold_frames:
                state['last_time'] = now
                state['count'] = 0
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
            state['count'] = 0
            base_metadata['hold_count'] = 0
            base_metadata['reason'] = 'fingers_too_far'

        return GestureResult(detected=False, gesture_name='pinch', metadata=base_metadata)


class PointingDetector:
    """
    Detects pointing gesture (index finger extended).
    Maintains per-hand state to support two-hand detection without interference.
    """
    def __init__(
            self, 
            min_extension_ratio: float = 0.12, 
            max_extra_fingers: int = 1, 
            max_speed: float = 0.5,
            ewma_alpha: float = 0.4
        ):
        self.min_extension_ratio = float(min_extension_ratio)
        self.max_extra_fingers = int(max_extra_fingers)
        self.max_speed = float(max_speed)
        self.ewma_alpha = ewma_alpha
        
        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }
    
    def _create_hand_state(self):
        return {'ewma_speed': EWMA(alpha=self.ewma_alpha)}
    
    def _get_state(self, hand_label: str):
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]
    
    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        state = self._get_state(hand_label)
        
        index_tip = metrics.tip_positions['index']
        centroid = metrics.centroid
        # Normalize distance by hand size (diag_rel = hand_diag / img_diag)
        # Distance is in normalized image coords, so divide by diag_rel to get relative to hand
        eps = 1e-6
        distance = euclidean(index_tip, centroid) / (metrics.diag_rel + eps)
        
        raw_speed = float(np.hypot(metrics.velocity[0], metrics.velocity[1]))
        smoothed_speed_arr = state['ewma_speed'].update([raw_speed])
        speed = float(smoothed_speed_arr[0])
        
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
            'raw_speed': raw_speed,
            'ewma_speed': speed,
            'max_speed': self.max_speed,
            'ewma_alpha': self.ewma_alpha,
            'index_extended': index_extended,
            'extra_fingers_count': extended_count,
            'max_extra_fingers': self.max_extra_fingers,
            'hand_label': hand_label,
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
    Maintains per-hand state to support two-hand detection without interference.
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
        
        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }
    
    def _create_hand_state(self):
        return {
            'confidence': 0.0,
            'ewma_velocity': EWMA(alpha=self.ewma_alpha),
            'current_direction': None
        }
    
    def _get_state(self, hand_label: str):
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]
    
    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        state = self._get_state(hand_label)
        
        vx, vy = metrics.velocity
        
        smoothed = state['ewma_velocity'].update([vx, vy])
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
            'current_direction': state['current_direction'],
            'raw_velocity': (vx, vy),
            'ewma_velocity': (ewma_vx, ewma_vy),
            'velocity_threshold_x': self.velocity_threshold_x,
            'velocity_threshold_y': self.velocity_threshold_y,
            'confidence': state['confidence'],
            'confidence_threshold': self.confidence_threshold,
            'hand_label': hand_label,
            'reason': None
        }
        
        if primary_velocity > max_velocity:
            state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
            base_metadata['reason'] = 'velocity_too_high'
            base_metadata['confidence'] = state['confidence']
            
            if state['confidence'] == 0.0:
                state['current_direction'] = None
            
            return GestureResult(
                detected=(state['confidence'] >= self.confidence_threshold),
                gesture_name='swipe',
                confidence=state['confidence'],
                metadata=base_metadata
            )
        
        if primary_velocity > velocity_threshold:
            if state['current_direction'] is None:
                state['current_direction'] = detected_direction
                state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
                base_metadata['reason'] = 'swipe_started'
            elif state['current_direction'] == detected_direction:
                state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
                base_metadata['reason'] = 'swipe_sustained'
            else:
                state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
                base_metadata['reason'] = 'direction_changed'
                
                if state['confidence'] == 0.0:
                    state['current_direction'] = detected_direction
                    state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
                    base_metadata['reason'] = 'swipe_direction_switched'
        else:
            state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
            base_metadata['reason'] = 'movement_stopped'
            
            if state['confidence'] == 0.0:
                state['current_direction'] = None
        
        base_metadata['current_direction'] = state['current_direction']
        base_metadata['confidence'] = state['confidence']
        
        is_detected = state['confidence'] >= self.confidence_threshold
        
        # Include direction in gesture name for consistency with other directional gestures
        gesture_name = 'swipe'
        if is_detected and state['current_direction']:
            gesture_name = f'swipe_{state["current_direction"]}'
        
        return GestureResult(
            detected=is_detected,
            gesture_name=gesture_name,
            confidence=state['confidence'],
            metadata=base_metadata
        )


class ZoomDetector:
    """
    Detects pinch-to-zoom using three fingers (thumb, index, middle).
    Tracks spread change between fingers with smoothing and confidence decay.
    Maintains per-hand state to support two-hand detection without interference.
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
        
        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }
    
    def _create_hand_state(self):
        return {
            'confidence': 0.0,
            'prev_spread': None,
            'ewma_velocity': EWMA(alpha=self.ewma_alpha),
            'current_direction': None
        }
    
    def _get_state(self, hand_label: str):
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]
    
    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        state = self._get_state(hand_label)
        
        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        spread = metrics.tip_distances.get('index_thumb', 0.0)

        base_metadata = {
            'spread': spread,
            'finger_gap': index_middle_dist,
            'finger_gap_threshold': self.finger_gap_threshold,
            'ewma_velocity': float(state['ewma_velocity'].value) if state['ewma_velocity'].value is not None else 0.0,
            'velocity_threshold': self.velocity_threshold,
            'confidence': state['confidence'],
            'confidence_threshold': self.confidence_threshold,
            'direction': state['current_direction'],
            'hand_label': hand_label,
            'reason': None
        }

        if index_middle_dist > self.finger_gap_threshold:
            state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
            base_metadata['reason'] = 'fingers_not_paired'
            
            if state['confidence'] == 0.0:
                state['prev_spread'] = None
                state['ewma_velocity'] = EWMA(alpha=self.ewma_alpha)
                state['current_direction'] = None
            
            return GestureResult(
                detected=(state['confidence'] >= self.confidence_threshold),
                gesture_name='zoom',
                confidence=state['confidence'],
                metadata=base_metadata
            )
        
        if state['prev_spread'] is None:
            state['prev_spread'] = spread
            base_metadata['reason'] = 'initializing'
            return GestureResult(
                detected=False,
                gesture_name='zoom',
                confidence=state['confidence'],
                metadata=base_metadata
            )
        
        delta_spread = spread - state['prev_spread']
        state['prev_spread'] = spread
        
        smoothed = state['ewma_velocity'].update([delta_spread])
        ewma_vel = float(smoothed[0])
        
        base_metadata['ewma_velocity'] = ewma_vel
        base_metadata['delta_spread'] = delta_spread
        
        abs_velocity = abs(ewma_vel)
        if abs_velocity > self.max_velocity:
            state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
            base_metadata['reason'] = 'velocity_too_high'
            
            if state['confidence'] == 0.0:
                state['current_direction'] = None
            
            return GestureResult(
                detected=(state['confidence'] >= self.confidence_threshold),
                gesture_name='zoom',
                confidence=state['confidence'],
                metadata=base_metadata
            )
        
        if ewma_vel < -self.velocity_threshold:
            detected_direction = 'out'
        elif ewma_vel > self.velocity_threshold:
            detected_direction = 'in'
        else:
            detected_direction = None
        
        if detected_direction is not None:
            if state['current_direction'] is None:
                state['current_direction'] = detected_direction
                state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
                base_metadata['reason'] = 'direction_started'
            elif state['current_direction'] == detected_direction:
                state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
                base_metadata['reason'] = 'direction_sustained'
            else:
                state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
                base_metadata['reason'] = 'direction_reversed'
                
                if state['confidence'] == 0.0:
                    state['current_direction'] = detected_direction
                    state['confidence'] = min(1.0, state['confidence'] + self.confidence_ramp_up)
        else:
            state['confidence'] = max(0.0, state['confidence'] - self.confidence_decay)
            base_metadata['reason'] = 'movement_stopped'
            
            if state['confidence'] == 0.0:
                state['current_direction'] = None
        
        base_metadata['direction'] = state['current_direction']
        base_metadata['confidence'] = state['confidence']
        
        is_detected = state['confidence'] >= self.confidence_threshold
        
        # Update reason if detected
        if is_detected and base_metadata['reason'] not in ['direction_started', 'direction_sustained']:
            base_metadata['reason'] = 'zoom_active'
        
        # Include direction in gesture name for consistency with other directional gestures
        gesture_name = 'zoom'
        if is_detected and state['current_direction']:
            gesture_name = f'zoom_{state["current_direction"]}'
        
        return GestureResult(
            detected=is_detected,
            gesture_name=gesture_name,
            confidence=state['confidence'],
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



class UniversalGestureDetector:
    """
    Generic detector for static and directional gestures based on finger states.
    Supports:
    - Custom finger state configuration (extended/curled)
    - 4-way directional movement (up/down/left/right) using EWMA velocity
    - Static hold validation
    - Optional finger gap constraints (e.g. for victory sign)
    - Optional orientation check (e.g. thumb up vs thumb down)
    """
    def __init__(
        self,
        finger_config: Dict[str, bool], # {finger_name: is_extended}
        gesture_name: str,
        velocity_threshold_x: float = 0.2,
        velocity_threshold_y: float = 0.2,
        ewma_alpha: float = 0.3,
        hold_frames: int = 5,
        confidence_ramp_up: float = 0.3,
        confidence_decay: float = 0.2,
        confidence_threshold: float = 0.6,
        max_finger_gap_pair: Optional[Tuple[str, str]] = None,
        max_finger_gap_threshold: float = 1.0, # default loose
        orientation_tip_index: Optional[int] = None,
        orientation_ref_index: Optional[int] = None # usually MCP
    ):
        self.finger_config = finger_config
        self.gesture_name = gesture_name
        self.velocity_threshold_x = velocity_threshold_x
        self.velocity_threshold_y = velocity_threshold_y
        self.ewma_alpha = ewma_alpha
        self.hold_frames = hold_frames
        self.confidence_ramp_up = confidence_ramp_up
        self.confidence_decay = confidence_decay
        self.confidence_threshold = confidence_threshold
        self.max_finger_gap_pair = max_finger_gap_pair
        self.max_finger_gap_threshold = max_finger_gap_threshold
        self.orientation_tip_index = orientation_tip_index
        self.orientation_ref_index = orientation_ref_index
        
        # Per-hand state
        self._hand_state = {
            'left': self._create_hand_state(),
            'right': self._create_hand_state()
        }

    def _create_hand_state(self):
        return {
            'ewma_velocity': EWMA(alpha=self.ewma_alpha),
            'move_confidence': 0.0,
            'current_move_direction': None,
            'static_hold_count': 0,
            'last_static_gesture': None
        }

    def _get_state(self, hand_label: str):
        if hand_label not in self._hand_state:
            self._hand_state[hand_label] = self._create_hand_state()
        return self._hand_state[hand_label]

    def detect(self, metrics: HandMetrics, hand_label: str = 'right') -> GestureResult:
        state = self._get_state(hand_label)
        
        # 1. Validate Finger States
        extended = metrics.fingers_extended
        fingers_match = True
        
        # Check explicitly configured fingers AND implicit closed fingers
        all_fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger in all_fingers:
            required = self.finger_config.get(finger, False) # Default to CLOSED of not specified
            if extended[finger] != required:
                fingers_match = False
                break
        
        # 2. Validate Gap (if configured)
        if fingers_match and self.max_finger_gap_pair:
            f1, f2 = self.max_finger_gap_pair
            dist = metrics.tip_distances.get(f'{f1}_{f2}', metrics.tip_distances.get(f'{f2}_{f1}', 1.0))
            if dist > self.max_finger_gap_threshold:
                fingers_match = False

        # 3. Check Orientation (Up/Down) if configured
        gesture_suffix = ""
        if fingers_match and self.orientation_tip_index is not None and self.orientation_ref_index is not None:
            # Check Y-coord of tip vs reference (MCP)
            # In image coords, y increases downwards.
            # Up means Tip Y < Ref Y
            # Down means Tip Y > Ref Y
            try:
                tip_y = metrics.landmarks_norm[self.orientation_tip_index][1]
                ref_y = metrics.landmarks_norm[self.orientation_ref_index][1]
                
                if tip_y < ref_y:
                    gesture_suffix = "_up"
                else:
                    gesture_suffix = "_down"
            except IndexError:
                fingers_match = False # Safety fallback

        base_metadata = {
            'fingers_match': fingers_match,
            'velocity': metrics.velocity,
            'ewma_velocity_y': 0.0,
            'move_confidence': state['move_confidence'],
            'static_hold_count': state['static_hold_count'],
            'hold_frames_needed': self.hold_frames,
            'hand_label': hand_label,
            'gesture_base': self.gesture_name,
            'reason': None
        }

        if not fingers_match:
            state['static_hold_count'] = 0
            state['last_static_gesture'] = None
            state['move_confidence'] = max(0.0, state['move_confidence'] - self.confidence_decay)
            if state['move_confidence'] == 0.0:
                state['current_move_direction'] = None
            base_metadata['reason'] = 'pose_mismatch'
            return GestureResult(detected=False, gesture_name='none', metadata=base_metadata)

        # 4. Process Velocity
        vx, vy = metrics.velocity
        smoothed = state['ewma_velocity'].update([vx, vy])
        ewma_vx = float(smoothed[0])
        ewma_vy = float(smoothed[1])
        base_metadata['ewma_velocity'] = (ewma_vx, ewma_vy)

        detected_move_direction = None
        
        # Determine direction based on dominant velocity
        if abs(ewma_vx) > abs(ewma_vy):
             # X movement
            if abs(ewma_vx) > self.velocity_threshold_x:
                detected_move_direction = 'right' if ewma_vx > 0 else 'left'
        else:
            # Y movement
            if abs(ewma_vy) > self.velocity_threshold_y:
                 detected_move_direction = 'down' if ewma_vy > 0 else 'up' 

        # 5. Update Confidence
        if detected_move_direction is not None:
            if state['current_move_direction'] is None:
                state['current_move_direction'] = detected_move_direction
                state['move_confidence'] = min(1.0, state['move_confidence'] + self.confidence_ramp_up)
            elif state['current_move_direction'] == detected_move_direction:
                state['move_confidence'] = min(1.0, state['move_confidence'] + self.confidence_ramp_up)
            else:
                # Direction changed
                state['move_confidence'] = max(0.0, state['move_confidence'] - self.confidence_decay)
                if state['move_confidence'] == 0.0:
                    state['current_move_direction'] = detected_move_direction
                    state['move_confidence'] = min(1.0, self.confidence_ramp_up)
        else:
             state['move_confidence'] = max(0.0, state['move_confidence'] - self.confidence_decay)
             if state['move_confidence'] == 0.0:
                state['current_move_direction'] = None

        movement_confirmed = state['move_confidence'] > self.confidence_threshold
        
        detected = False
        final_gesture_name = 'none'
        
        # Combine base name + suffix (e.g. thumbs_up)
        # Only add suffix if orientation is configured AND matched
        # If not orientation configured, suffix is empty, so name is just base
        current_static_name = f"{self.gesture_name}{gesture_suffix}"

        # 6. Determine Result (Moving vs Static)
        if movement_confirmed and state['current_move_direction']:
             final_gesture_name = f'{current_static_name}_moving_{state["current_move_direction"]}'
             detected = True
             state['static_hold_count'] = 0
             state['last_static_gesture'] = None
             base_metadata['reason'] = 'movement_confirmed'
        else:
            # Check static hold
            if state['last_static_gesture'] == current_static_name:
                state['static_hold_count'] += 1
            else:
                state['last_static_gesture'] = current_static_name
                state['static_hold_count'] = 1
            
            base_metadata['static_hold_count'] = state['static_hold_count']
            
            if state['static_hold_count'] >= self.hold_frames:
                final_gesture_name = current_static_name
                detected = True
                base_metadata['reason'] = 'static_confirmed'
            else:
                base_metadata['reason'] = 'waiting_for_hold'

        return GestureResult(
            detected=detected,
            gesture_name=final_gesture_name,
            confidence=state['move_confidence'] if movement_confirmed else (1.0 if detected else 0.0),
            metadata=base_metadata
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
            pinch_indx_midl_gap_thresh = get_gesture_threshold('pinch','index_middle_gap_threshold', default=0.2)

            pointing_min_ext = get_gesture_threshold('pointing', 'min_extension_ratio', default=0.12)
            pointing_max_extra = get_gesture_threshold('pointing', 'max_extra_fingers', default=1)
            pointing_max_speed = get_gesture_threshold('pointing', 'max_speed', default=0.5)
            pointing_ewma_alpha = get_gesture_threshold('pointing', 'ewma_alpha', default=0.4)

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
            
            thumbs_velocity_thresh_x = get_gesture_threshold('thumbs', 'velocity_threshold_x', default=0.2)
            thumbs_velocity_thresh_y = get_gesture_threshold('thumbs', 'velocity_threshold_y', default=0.2)
            thumbs_ewma_alpha = get_gesture_threshold('thumbs', 'ewma_alpha', default=0.3)
            thumbs_hold_frames = get_gesture_threshold('thumbs', 'hold_frames', default=5)
            thumbs_conf_ramp = get_gesture_threshold('thumbs', 'confidence_ramp_up', default=0.3)
            thumbs_conf_decay = get_gesture_threshold('thumbs', 'confidence_decay', default=0.2)
            thumbs_conf_thresh = get_gesture_threshold('thumbs', 'confidence_threshold', default=0.6)

            # For closed hands
            closed_velocity_thresh_x = get_gesture_threshold('closed', 'velocity_threshold_x', default=0.3)
            closed_velocity_thresh_y = get_gesture_threshold('closed', 'velocity_threshold_y', default=0.3)
            closed_ewma_alpha = get_gesture_threshold('closed', 'ewma_alpha', default=0.3)
            closed_hold_frames = get_gesture_threshold('closed', 'hold_frames', default=5)
            closed_conf_ramp = get_gesture_threshold('closed', 'confidence_ramp_up', default=0.3)
            closed_conf_decay = get_gesture_threshold('closed', 'confidence_decay', default=0.2)
            closed_conf_thresh = get_gesture_threshold('closed', 'confidence_threshold', default=0.6)

            # For victory hands
            victory_velocity_thresh_x = get_gesture_threshold('victory', 'velocity_threshold_x', default=0.3)
            victory_velocity_thresh_y = get_gesture_threshold('victory', 'velocity_threshold_y', default=0.3)
            victory_ewma_alpha = get_gesture_threshold('victory', 'ewma_alpha', default=0.3)
            victory_hold_frames = get_gesture_threshold('victory', 'hold_frames', default=5)
            victory_conf_ramp = get_gesture_threshold('victory', 'confidence_ramp_up', default=0.3)
            victory_conf_decay = get_gesture_threshold('victory', 'confidence_decay', default=0.2)
            victory_conf_thresh = get_gesture_threshold('victory', 'confidence_threshold', default=0.6)
            victory_index_middle_gap_threshold = get_gesture_threshold('victory', 'index_middle_gap_threshold', default=0.175)

            # For middle finger
            middle_velocity_thresh = get_gesture_threshold('middle_finger', 'velocity_threshold', default=0.2)
            middle_ewma_alpha = get_gesture_threshold('middle_finger', 'ewma_alpha', default=0.3)
            middle_hold_frames = get_gesture_threshold('middle_finger', 'hold_frames', default=5)
            middle_conf_ramp = get_gesture_threshold('middle_finger', 'confidence_ramp_up', default=0.3)
            middle_conf_decay = get_gesture_threshold('middle_finger', 'confidence_decay', default=0.2)
            middle_conf_thresh = get_gesture_threshold('middle_finger', 'confidence_threshold', default=0.6)

            # For pinky
            pinky_velocity_thresh = get_gesture_threshold('pinky', 'velocity_threshold', default=0.2)
            pinky_ewma_alpha = get_gesture_threshold('pinky', 'ewma_alpha', default=0.3)
            pinky_hold_frames = get_gesture_threshold('pinky', 'hold_frames', default=5)
            pinky_conf_ramp = get_gesture_threshold('pinky', 'confidence_ramp_up', default=0.3)
            pinky_conf_decay = get_gesture_threshold('pinky', 'confidence_decay', default=0.2)
            pinky_conf_thresh = get_gesture_threshold('pinky', 'confidence_threshold', default=0.6)
        except Exception:
            print("Failed to load config file and so falling back to hadcoded value")
            # Fallback to hardcoded defaults if config access fails
            pinch_thresh, pinch_hold, pinch_cd,index_middle_gap_threshold = 0.055, 5, 0.6,0.2
            pointing_min_ext = 0.12
            pointing_max_extra = 1
            pointing_max_speed = 0.5
            pointing_ewma_alpha = 0.4
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
            thumbs_ewma_alpha = 0.3
            thumbs_hold_frames = 5
            thumbs_conf_ramp = 0.3
            thumbs_conf_decay = 0.2
            thumbs_conf_thresh = 0.6

        # Initialize detectors with resolved parameters
        self.pinch = PinchDetector(
            thresh_rel=pinch_thresh, 
            hold_frames=pinch_hold, 
            cooldown_s=pinch_cd,
            index_middle_gap_threshold=pinch_indx_midl_gap_thresh
        )
        self.pointing = PointingDetector(
            min_extension_ratio=pointing_min_ext,
            max_extra_fingers=pointing_max_extra,
            max_speed=pointing_max_speed,
            ewma_alpha=pointing_ewma_alpha
        )
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
        self.thumbs = UniversalGestureDetector(
            finger_config={'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            gesture_name='thumbs',
            velocity_threshold_x=thumbs_velocity_thresh_x,
            velocity_threshold_y=thumbs_velocity_thresh_y,
            ewma_alpha=thumbs_ewma_alpha,
            hold_frames=thumbs_hold_frames,
            confidence_ramp_up=thumbs_conf_ramp,
            confidence_decay=thumbs_conf_decay,
            confidence_threshold=thumbs_conf_thresh,
            orientation_tip_index=LANDMARK_NAMES['THUMB_TIP'],
            orientation_ref_index=LANDMARK_NAMES['PINKY_MCP']
        )
        self.closed_hand = UniversalGestureDetector(
            finger_config={'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            gesture_name='closed_hand',
            velocity_threshold_x=closed_velocity_thresh_x,
            velocity_threshold_y=closed_velocity_thresh_y,
            ewma_alpha=closed_ewma_alpha,
            hold_frames=closed_hold_frames,
            confidence_ramp_up=closed_conf_ramp,
            confidence_decay=closed_conf_decay,
            confidence_threshold=closed_conf_thresh
        )
        self.victory_hand = UniversalGestureDetector(
            finger_config={'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            gesture_name='victory',
            velocity_threshold_x=victory_velocity_thresh_x,
            velocity_threshold_y=victory_velocity_thresh_y,
            ewma_alpha=victory_ewma_alpha,
            hold_frames=victory_hold_frames,
            confidence_ramp_up=victory_conf_ramp,
            confidence_decay=victory_conf_decay,
            confidence_threshold=victory_conf_thresh,
            max_finger_gap_pair=('index', 'middle'),
            max_finger_gap_threshold=victory_index_middle_gap_threshold
        )
        self.middlefinger = UniversalGestureDetector(
            finger_config={'thumb': False, 'index': False, 'middle': True, 'ring': False, 'pinky': False},
            gesture_name='middle_finger',
            velocity_threshold_x=middle_velocity_thresh,
            velocity_threshold_y=middle_velocity_thresh,
            ewma_alpha=middle_ewma_alpha,
            hold_frames=middle_hold_frames,
            confidence_ramp_up=middle_conf_ramp,
            confidence_decay=middle_conf_decay,
            confidence_threshold=middle_conf_thresh,
            orientation_tip_index=LANDMARK_NAMES['MIDDLE_TIP'],
            orientation_ref_index=LANDMARK_NAMES['PINKY_MCP']
        )
        self.pinky = UniversalGestureDetector(
            finger_config={'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': True},
            gesture_name='pinky',
            velocity_threshold_x=pinky_velocity_thresh,
            velocity_threshold_y=pinky_velocity_thresh,
            ewma_alpha=pinky_ewma_alpha,
            hold_frames=pinky_hold_frames,
            confidence_ramp_up=pinky_conf_ramp,
            confidence_decay=pinky_conf_decay,
            confidence_threshold=pinky_conf_thresh,
            orientation_tip_index=LANDMARK_NAMES['PINKY_TIP'],
            orientation_ref_index=LANDMARK_NAMES['PINKY_MCP']
        )
        # Store finger-extension hysteresis for use during metric computation
        self.finger_open_ratio = open_ratio
        self.finger_close_ratio = close_ratio
        self.finger_motion_threshold = finger_motion_threshold
        self.finger_motion_sigmoid_k = finger_motion_sigmoid_k
        
        # State tracking
        history_maxlen = get_gesture_threshold('performance', 'gesture_history_maxlen', default=16) if 'get_gesture_threshold' in dir() else 16
        self.history = {'left': deque(maxlen=history_maxlen), 'right': deque(maxlen=history_maxlen)}
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
        
        results = {}
        
        # Pass hand_label to all stateful detectors for per-hand state tracking
        pinch_result = self.pinch.detect(metrics, hand_label)
        pointing_result = self.pointing.detect(metrics, hand_label)
        swipe_result = self.swipe.detect(metrics, hand_label)
        zoom_result = self.zoom.detect(metrics, hand_label)
        thumbs_result = self.thumbs.detect(metrics, hand_label)
        open_hand_result = self.open_hand.detect(metrics)  # Stateless, no hand_label needed


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
            results[thumbs_result.gesture_name] = thumbs_result
            # if swipe_result.detected:
            #     results['swipe'] = swipe_result
            return results
        
        if zoom_result.detected:
            results[zoom_result.gesture_name] = zoom_result
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
        if swipe_result.detected:
            finger_count = sum(metrics.fingers_extended.values())
            if finger_count >= 4:
                results[swipe_result.gesture_name] = swipe_result
                return results
                        
        if open_hand_result.detected:
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
