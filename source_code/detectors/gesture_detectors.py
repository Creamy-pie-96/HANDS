
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

    # Computed velocity is normalized so distance of hand from camera does not create issues
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

    # Motion-aware scaling: compute sigmoid of speed and use it to increase
    # the open_ratio when the hand is moving. Sigmoid in (0,1), factor = 1+sigmoid -> (1,2).
    # Using sigmoid cz velocity of hand can be any arbitary value and so we need to wrap all input 
    # value of velocity between a range or value for predictable behaviour(1 to 2). So sigmoid(vel)+1 takes any arbitary
    # value of velocity and wrap it between 0 and 1 and then add 1 to wrap between (1 to 2)
    try:
        sig = 1.0 / (1.0 + float(np.exp(-motion_sigmoid_k * (speed - motion_speed_threshold))))
    except Exception:
        sig = 0.0

    motion_factor = 1.0 + sig
    effective_open_ratio = open_ratio * motion_factor

    # tracks which finger is extented
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
                return GestureResult(
                    detected=True,
                    gesture_name='pinch',
                    confidence=1.0,
                    metadata=base_metadata
                )
        else:
            self._count = 0
            base_metadata['hold_count'] = 0

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

        return GestureResult(
            detected=True,
            gesture_name='pointing',
            confidence=1.0,
            metadata=base_metadata
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
        history_size: int = 8,
        min_history: int = 3
    ):
        self.velocity_threshold = velocity_threshold
        self.cooldown_s = cooldown_s
        self.history = deque(maxlen=history_size)
        # minimum number of frames required before attempting detection
        self.min_history = int(min_history)
        self.last_swipe_time = -999.0
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        self.history.append(metrics)
        
        vx, vy = metrics.velocity
        speed = float(np.hypot(vx, vy))
        
        # Determine direction based on which component is larger
        if abs(vx) > abs(vy):
            direction = 'right' if vx > 0 else 'left'
        else:
            direction = 'down' if vy > 0 else 'up'
        
        now = time.time()
        in_cooldown = now - self.last_swipe_time < self.cooldown_s
        
        # Always return full metadata for visual feedback
        base_metadata = {
            'direction': direction,
            'speed': speed,
            'velocity': (vx, vy),
            'velocity_threshold': self.velocity_threshold,
            'history_size': len(self.history),
            'min_history': self.min_history,
            'in_cooldown': in_cooldown,
            'cooldown_remaining': max(0.0, self.cooldown_s - (now - self.last_swipe_time)),
            'reason': None
        }

        if len(self.history) < self.min_history:
            base_metadata['reason'] = 'insufficient_history'
            return GestureResult(detected=False, gesture_name='swipe', metadata=base_metadata)

        if in_cooldown:
            base_metadata['reason'] = 'in_cooldown'
            return GestureResult(detected=False, gesture_name='swipe', metadata=base_metadata)

        if speed < self.velocity_threshold:
            base_metadata['reason'] = 'speed_too_low'
            return GestureResult(detected=False, gesture_name='swipe', metadata=base_metadata)

        # Update cooldown timer for next time
        self.last_swipe_time = now

        return GestureResult(
            detected=True,
            gesture_name='swipe',
            confidence=1.0,
            metadata=base_metadata
        )


class ZoomDetector:
    """
    Detects pinch-to-zoom gesture using 3 fingers (thumb, index, middle).
    Tracks spread change to determine zoom in/out.
    Uses inertia for stable detection (reduces flicker).
    Uses velocity-based detection to distinguish intentional zoom from drift.
    """
    def __init__(
        self,
        scale_threshold: float = 0.15,  # 15% change to trigger
        history_size: int = 5,
        finger_gap_threshold: float = 0.19,  # Max distance between index and middle
        inertia_increase: float = 0.3,  # How much to increase inertia per valid frame
        inertia_decrease: float = 0.15,  # How much to decrease inertia per invalid frame
        inertia_threshold: float = 0.5,  # Threshold above which zoom is detected
        min_velocity: float = 0.05,  # Minimum spread change velocity (prevents drift)
        max_velocity: float = 2.0,  # Maximum spread change velocity (prevents jumps)
        velocity_consistency_threshold: float = 0.7,  # How consistent velocity direction must be (0-1)
        require_fingers_extended: bool = False  # Whether to require finger extension detection
    ):
        self.scale_threshold = scale_threshold
        self.history = deque(maxlen=history_size)
        self.finger_gap_threshold = finger_gap_threshold
        self.inertia_increase = inertia_increase
        self.inertia_decrease = inertia_decrease
        self.inertia_threshold = inertia_threshold
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.velocity_consistency_threshold = velocity_consistency_threshold
        self.require_fingers_extended = require_fingers_extended
        self.inertia = 0.0  # Current inertia level (0.0 to 1.0)
        self.velocity_history = deque(maxlen=history_size)  # Track velocity for consistency check
    
    def detect(self, metrics: HandMetrics) -> GestureResult:
        """
        Finger pairing:
        - Index and middle MUST be close together (like a pair)
        - Measures spread between (index+middle pair) and thumb
        - Tracks continuous direction (zoom in vs out)
        - Prevents confusion with pinch (which has thumb+index, not index+middle pair)
        """
        
        # Compute basic distances up-front so we can always expose them
        index_middle_dist = metrics.tip_distances.get('index_middle', 0.0)
        spread = metrics.tip_distances.get('index_thumb', 0.0)

        # Check if index and middle are close together (paired)
        if index_middle_dist > self.finger_gap_threshold:
            # Clear history to reset trend analysis, but still return useful
            # metadata so the UI can display current measurements while not
            # actively detecting a zoom.
            self.history.clear()
            return GestureResult(
                detected=False,
                gesture_name='zoom',
                confidence=self.inertia,
                metadata={
                    'finger_gap': index_middle_dist,
                    'spread': spread,
                    'inertia': self.inertia,
                    'reason': 'pair_separation'
                }
            )
        
        # Add to history
        self.history.append(spread)
        
        # Need at least the configured history window frames to detect continuous direction
        history_needed = int(self.history.maxlen) if self.history.maxlen is not None else 3
        if len(self.history) < history_needed:
            # Decrease inertia when not enough history
            self.inertia = max(0.0, self.inertia - self.inertia_decrease)
            detected = self.inertia >= self.inertia_threshold
            # Provide consistent metadata so UI can display live measurements
            return GestureResult(
                detected=detected,
                gesture_name='zoom',
                confidence=self.inertia,
                metadata={
                    'inertia': self.inertia,
                    'reason': 'insufficient_history',
                    'finger_gap': index_middle_dist,
                    'spread': spread,
                    'relative_change': 0.0,
                    'avg_velocity': 0.0,
                    'velocity_consistency': 1.0
                }
            )

        # Determine trend window: use configured history but ensure at least 3
        # frames for a minimal trend calculation.
        window = max(3, history_needed)
        recent = list(self.history)[-window:]
        
        # Compute differences across the recent window and check sign consistency.
        # Using a small epsilon to ignore numerical noise.
        eps = 1e-6
        diffs: List[float] = []
        for i in range(len(recent) - 1):
            diffs.append(float(recent[i + 1] - recent[i]))

        # Determine the trend and compute velocities
        increasing = all(d > eps for d in diffs)
        decreasing = all(d < -eps for d in diffs)
        
        # Compute average velocity (spread change per frame)
        avg_velocity = float(np.mean([abs(d) for d in diffs])) if diffs else 0.0
        self.velocity_history.append(avg_velocity)

        # Check velocity consistency: velocities should be within reasonable range
        velocity_valid = self.min_velocity <= avg_velocity <= self.max_velocity
        
        # Check velocity consistency over history (should be relatively stable)
        if len(self.velocity_history) >= 3:
            vel_list = list(self.velocity_history)
            vel_std = float(np.std(vel_list))
            vel_mean = float(np.mean(vel_list))
            # Coefficient of variation (std/mean) should be reasonable
            velocity_consistency = 1.0 - min(1.0, vel_std / (vel_mean + eps)) if vel_mean > eps else 0.0
        else:
            velocity_consistency = 1.0  # Assume consistent if not enough data
        
        velocity_consistent = velocity_consistency >= self.velocity_consistency_threshold

        # If the diffs do not all share the same sign it was not consistent to count as zoom
        if not (increasing or decreasing):
            # Decrease inertia when trend is unstable
            self.inertia = max(0.0, self.inertia - self.inertia_decrease)
            detected = self.inertia >= self.inertia_threshold
            return GestureResult(
                detected=detected,
                gesture_name='zoom',
                confidence=self.inertia,
                metadata={
                    'inertia': self.inertia,
                    'reason': 'unstable_trend',
                    'finger_gap': index_middle_dist,
                    'spread': spread,
                    'relative_change': 0.0,
                    'avg_velocity': avg_velocity,
                    'velocity_consistency': velocity_consistency
                }
            )
        
        # Check velocity-based conditions
        if not velocity_valid:
            # Velocity too low (drift) or too high (jump/noise)
            self.inertia = max(0.0, self.inertia - self.inertia_decrease)
            detected = self.inertia >= self.inertia_threshold
            return GestureResult(
                detected=detected,
                gesture_name='zoom',
                confidence=self.inertia,
                metadata={
                    'inertia': self.inertia,
                    'reason': 'velocity_out_of_range',
                    'finger_gap': index_middle_dist,
                    'spread': spread,
                    'relative_change': 0.0,
                    'avg_velocity': avg_velocity,
                    'velocity_consistency': velocity_consistency
                }
            )
        
        if not velocity_consistent:
            # Velocity fluctuating too much (not smooth zoom)
            self.inertia = max(0.0, self.inertia - self.inertia_decrease)
            detected = self.inertia >= self.inertia_threshold
            return GestureResult(
                detected=detected,
                gesture_name='zoom',
                confidence=self.inertia,
                metadata={
                    'inertia': self.inertia,
                    'reason': 'velocity_inconsistent',
                    'finger_gap': index_middle_dist,
                    'spread': spread,
                    'relative_change': 0.0,
                    'avg_velocity': avg_velocity,
                    'velocity_consistency': velocity_consistency
                }
            )
        
        # Compute total change magnitude using an EWMA average across the
        try:
            ewma_alpha = 1.0 / float(len(recent)) if len(recent) > 0 else 1.0
            ew = EWMA(alpha=ewma_alpha)
            for v in recent:
                ew.update([float(v)])
            avg_spread = float(np.asarray(ew.value).flatten()[0])
        except Exception:
            # Fallback to first-element baseline if EWMA fails
            avg_spread = float(recent[0]) if len(recent) > 0 else 0.0

        # Compare last sample to the EWMA baseline
        total_change = float(recent[-1]) - avg_spread
        relative_change = abs(total_change) / (avg_spread + 1e-6)
        
        # Check if zoom conditions are met based on continuous direction, threshold, and velocity
        zoom_conditions_met = False
        zoom_type = None
        
        if increasing and relative_change > self.scale_threshold:
            zoom_type = 'out'
            zoom_conditions_met = True
        elif decreasing and relative_change > self.scale_threshold:
            zoom_type = 'in'
            zoom_conditions_met = True
        
        # Update inertia based on whether zoom conditions are met
        if zoom_conditions_met:
            # Increase inertia when zoom is detected
            self.inertia = min(1.0, self.inertia + self.inertia_increase)
        else:
            # Decrease inertia when zoom is not detected
            self.inertia = max(0.0, self.inertia - self.inertia_decrease)
        
        # Only report as detected if inertia is above threshold
        detected = self.inertia >= self.inertia_threshold
        
        return GestureResult(
            detected=detected,
            gesture_name='zoom',
            confidence=self.inertia,
            metadata={
                'zoom_type': zoom_type,
                'relative_change': relative_change,
                'spread': spread,
                'finger_gap': index_middle_dist,
                'trend': 'increasing' if increasing else 'decreasing' if decreasing else 'unstable',
                'inertia': self.inertia,
                'zoom_conditions_met': zoom_conditions_met,
                'avg_velocity': avg_velocity,
                'velocity_consistency': velocity_consistency,
                'velocity_valid': velocity_valid,
                'velocity_consistent': velocity_consistent
            }
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
        
        # Additional check: if thumb and index are too close, it's a pinch, not open hand
        thumb_index_dist = metrics.tip_distances.get('index_thumb', 0.0)
        is_pinching = thumb_index_dist < float(self.pinch_threshold)
        
        # Detect if at least min_fingers are extended AND not pinching
        detected = count >= self.min_fingers and not is_pinching
        
        # Always return full metadata for visual feedback
        base_metadata = {
            'finger_count': count,
            'min_fingers': self.min_fingers,
            'thumb_index_dist': thumb_index_dist,
            'pinch_threshold': self.pinch_threshold,
            'is_pinching': is_pinching,
            'fingers_extended': dict(metrics.fingers_extended),
            'reason': None if detected else ('pinching' if is_pinching else 'not_enough_fingers')
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

            swipe_thresh = get_gesture_threshold('swipe', 'velocity_threshold', default=0.8)
            swipe_cd = get_gesture_threshold('swipe', 'cooldown_seconds', default=0.5)
            swipe_hist = get_gesture_threshold('swipe', 'history_size', default=8)
            swipe_min_history = get_gesture_threshold('swipe', 'min_history', default=3)

            zoom_scale = get_gesture_threshold('zoom', 'scale_threshold', default=0.15)
            zoom_gap = get_gesture_threshold('zoom', 'finger_gap_threshold', default=0.06)
            zoom_hist = get_gesture_threshold('zoom', 'history_size', default=5)
            zoom_inertia_inc = get_gesture_threshold('zoom', 'inertia_increase', default=0.3)
            zoom_inertia_dec = get_gesture_threshold('zoom', 'inertia_decrease', default=0.15)
            zoom_inertia_thresh = get_gesture_threshold('zoom', 'inertia_threshold', default=0.5)
            zoom_min_vel = get_gesture_threshold('zoom', 'min_velocity', default=0.05)
            zoom_max_vel = get_gesture_threshold('zoom', 'max_velocity', default=2.0)
            zoom_vel_consistency = get_gesture_threshold('zoom', 'velocity_consistency_threshold', default=0.7)
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
            swipe_thresh, swipe_cd, swipe_hist = 0.8, 0.5, 8
            swipe_min_history = 3
            zoom_scale, zoom_gap, zoom_hist = 0.15, 0.06, 5
            zoom_inertia_inc, zoom_inertia_dec, zoom_inertia_thresh = 0.3, 0.15, 0.5
            zoom_min_vel, zoom_max_vel, zoom_vel_consistency = 0.05, 2.0, 0.7
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
        self.swipe = SwipeDetector(velocity_threshold=swipe_thresh, cooldown_s=swipe_cd, history_size=swipe_hist, min_history=swipe_min_history)
        self.zoom = ZoomDetector(
            scale_threshold=zoom_scale, 
            history_size=zoom_hist, 
            finger_gap_threshold=zoom_gap, 
            inertia_increase=zoom_inertia_inc, 
            inertia_decrease=zoom_inertia_dec, 
            inertia_threshold=zoom_inertia_thresh,
            min_velocity=zoom_min_vel,
            max_velocity=zoom_max_vel,
            velocity_consistency_threshold=zoom_vel_consistency,
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
        # 1. Zoom
        # 2. Pinch
        # 3. Pointing
        # 4. Thumbs
        # 5. Swipe
        # 6. Open hand
        
        # Check high-priority gestures first
        
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
            
        if thumbs_result.detected:
            results['thumbs'] = thumbs_result
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
