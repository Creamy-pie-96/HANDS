"""
Bimanual (Two-Hand) Gesture Detection System

Implements complex two-hand gestures where one hand acts as a modifier
and the other performs the primary action.

Design principles:
- One hand sets the mode/modifier state
- Other hand performs the action
- Single-hand gestures still work independently
- Two-hand gestures have higher priority for specific actions
"""

import numpy as np
import time
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from gesture_detectors import HandMetrics, GestureResult, compute_hand_metrics
from math_utils import euclidean


@dataclass
class BimanualState:
    """
    Tracks the state of both hands for bimanual gesture detection.
    """
    left_metrics: Optional[HandMetrics] = None
    right_metrics: Optional[HandMetrics] = None
    left_gesture: str = 'none'  # Current left hand gesture
    right_gesture: str = 'none'  # Current right hand gesture
    inter_hand_distance: float = 0.0  # Distance between hand centroids
    timestamp: float = 0.0


class BimanualGestureDetector:
    """
    Detects complex two-hand gestures based on single-hand gesture combinations.
    """
    
    def __init__(self):
        self.history = deque(maxlen=10)
        self.current_mode = None  # Active bimanual mode
        self.anchor_point = None  # Anchor for pan/rotate
        
    def detect(
        self,
        left_metrics: Optional[HandMetrics],
        right_metrics: Optional[HandMetrics],
        left_gestures: Dict[str, GestureResult],
        right_gestures: Dict[str, GestureResult]
    ) -> Dict[str, GestureResult]:
        """
        Detect bimanual gestures based on current hand states.
        
        Args:
            left_metrics: Metrics for left hand (None if not detected)
            right_metrics: Metrics for right hand (None if not detected)
            left_gestures: Detected gestures for left hand
            right_gestures: Detected gestures for right hand
            
        Returns:
            Dict of detected bimanual gestures
        """
        results = {}
        
        # Early exit if either hand missing
        if left_metrics is None or right_metrics is None:
            self.current_mode = None
            self.anchor_point = None
            return results
        
        # Compute inter-hand metrics
        inter_distance = euclidean(left_metrics.centroid, right_metrics.centroid)
        
        # Create state snapshot
        state = BimanualState(
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            left_gesture=self._primary_gesture(left_gestures),
            right_gesture=self._primary_gesture(right_gestures),
            inter_hand_distance=inter_distance,
            timestamp=time.time()
        )
        self.history.append(state)
        
        # Detect bimanual gesture patterns
        
        # 1. Pan/Scroll: Left hand hold still + right hand move
        pan_result = self._detect_pan_scroll(state)
        if pan_result.detected:
            results['pan'] = pan_result
        
        # 2. Rotate: Left hand hold still + right hand zoom
        rotate_result = self._detect_rotate(state)
        if rotate_result.detected:
            results['rotate'] = rotate_result
        
        # 3. Two-hand resize: Both hands pinch
        resize_result = self._detect_two_hand_resize(state)
        if resize_result.detected:
            results['two_hand_resize'] = resize_result
        
        # 4. Precision mode: Left hold still + right pointing
        precision_result = self._detect_precision_mode(state)
        if precision_result.detected:
            results['precision_cursor'] = precision_result
        
        # 5. Draw mode: Left pinch lock + right pointing
        draw_result = self._detect_draw_mode(state)
        if draw_result.detected:
            results['draw_mode'] = draw_result
        
        # 6. Undo: Left hold still + right swipe left
        undo_result = self._detect_undo(state)
        if undo_result.detected:
            results['undo'] = undo_result
        
        # 7. Quick menu: Left zoom in + right pinch
        menu_result = self._detect_quick_menu(state)
        if menu_result.detected:
            results['quick_menu'] = menu_result
        
        # 8. Warp/Teleport: Both hands pointing
        warp_result = self._detect_warp(state)
        if warp_result.detected:
            results['warp'] = warp_result
        
        return results
    
    def _primary_gesture(self, gestures: Dict[str, GestureResult]) -> str:
        """Get the primary detected gesture name."""
        if not gestures:
            return 'none'
        # Return highest priority gesture
        priority = ['pinch', 'zoom', 'pointing', 'swipe', 'open_hand']
        for g in priority:
            if g in gestures:
                return g
        return 'none'
    
    def _is_hand_still(self, metrics: HandMetrics, threshold: float = 0.3) -> bool:
        """Check if hand is relatively still (low velocity)."""
        vel_mag = np.hypot(*metrics.velocity)
        return vel_mag < threshold
    
    def _detect_pan_scroll(self, state: BimanualState) -> GestureResult:
        """
        Pan/Scroll: Left hand hold still, right hand moves.
        The still hand acts as an anchor point.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='pan')
        
        if self._is_hand_still(state.left_metrics):
            right_vel = np.hypot(*state.right_metrics.velocity)
            if right_vel > 0.4:  # Right hand moving
                # Set anchor if not set
                if self.current_mode != 'pan' or self.anchor_point is None:
                    self.anchor_point = state.left_metrics.centroid
                    self.current_mode = 'pan'
                
                # Calculate pan delta relative to anchor
                delta = (
                    state.right_metrics.centroid[0] - self.anchor_point[0],
                    state.right_metrics.centroid[1] - self.anchor_point[1]
                )
                
                return GestureResult(
                    detected=True,
                    gesture_name='pan',
                    confidence=1.0,
                    metadata={
                        'anchor': self.anchor_point,
                        'delta': delta,
                        'velocity': state.right_metrics.velocity
                    }
                )
        
        if self.current_mode == 'pan':
            self.current_mode = None
            self.anchor_point = None
        
        return GestureResult(detected=False, gesture_name='pan')
    
    def _detect_rotate(self, state: BimanualState) -> GestureResult:
        """
        Rotate: Left hand hold still (anchor), right hand zoom gesture.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='rotate')
        
        if self._is_hand_still(state.left_metrics) and state.right_gesture == 'zoom':
            if self.anchor_point is None:
                self.anchor_point = state.left_metrics.centroid
            
            # Calculate rotation angle using vector math
            angle_delta = 0.0
            if len(self.history) >= 2:
                # Get previous state
                prev_state = self.history[-2]
                if prev_state.right_metrics is not None:
                    # Vector from anchor to previous right hand position
                    prev_vec = np.array([
                        prev_state.right_metrics.centroid[0] - self.anchor_point[0],
                        prev_state.right_metrics.centroid[1] - self.anchor_point[1]
                    ])
                    # Vector from anchor to current right hand position
                    curr_vec = np.array([
                        state.right_metrics.centroid[0] - self.anchor_point[0],
                        state.right_metrics.centroid[1] - self.anchor_point[1]
                    ])
                    
                    # Calculate angle between vectors using atan2
                    prev_angle = np.arctan2(prev_vec[1], prev_vec[0])
                    curr_angle = np.arctan2(curr_vec[1], curr_vec[0])
                    angle_delta = np.degrees(curr_angle - prev_angle)
                    
                    # Normalize to [-180, 180]
                    if angle_delta > 180:
                        angle_delta -= 360
                    elif angle_delta < -180:
                        angle_delta += 360
            
            # Rotation based on distance change around anchor
            return GestureResult(
                detected=True,
                gesture_name='rotate',
                confidence=1.0,
                metadata={
                    'anchor': self.anchor_point,
                    'angle_delta': angle_delta
                }
            )
        
        return GestureResult(detected=False, gesture_name='rotate')
    
    def _detect_two_hand_resize(self, state: BimanualState) -> GestureResult:
        """
        Two-hand resize: Both hands pinch + change distance between hands.
        Like pinch-to-zoom on touchscreen but with two hands.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='two_hand_resize')
        
        if state.left_gesture == 'pinch' and state.right_gesture == 'pinch':
            # Need history to detect distance change
            if len(self.history) >= 3:
                recent = list(self.history)[-3:]
                dist_start = recent[0].inter_hand_distance
                dist_end = recent[-1].inter_hand_distance
                change = (dist_end - dist_start) / (dist_start + 1e-6)
                
                if abs(change) > 0.1:  # 10% change threshold
                    resize_type = 'expand' if change > 0 else 'contract'
                    return GestureResult(
                        detected=True,
                        gesture_name='two_hand_resize',
                        confidence=1.0,
                        metadata={
                            'resize_type': resize_type,
                            'scale_factor': 1.0 + change,
                            'distance': state.inter_hand_distance
                        }
                    )
        
        return GestureResult(detected=False, gesture_name='two_hand_resize')
    
    def _detect_precision_mode(self, state: BimanualState) -> GestureResult:
        """
        Precision cursor: Left hand hold still (stabilize), right hand pointing.
        Reduces cursor speed for fine control.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='precision_cursor')
        
        if self._is_hand_still(state.left_metrics) and state.right_gesture == 'pointing':
            return GestureResult(
                detected=True,
                gesture_name='precision_cursor',
                confidence=1.0,
                metadata={
                    'cursor_pos': state.right_metrics.tip_positions['index'],
                    'damping_factor': 0.3  # Reduce speed by 70%
                }
            )
        
        return GestureResult(detected=False, gesture_name='precision_cursor')
    
    def _detect_draw_mode(self, state: BimanualState) -> GestureResult:
        """
        Draw mode: Left hand pinch (mode lock), right hand pointing + moving.
        Continuous action until left hand releases.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='draw_mode')
        
        if state.left_gesture == 'pinch' and state.right_gesture == 'pointing':
            right_vel = np.hypot(*state.right_metrics.velocity)
            if right_vel > 0.2:  # Moving
                return GestureResult(
                    detected=True,
                    gesture_name='draw_mode',
                    confidence=1.0,
                    metadata={
                        'cursor_pos': state.right_metrics.tip_positions['index'],
                        'velocity': state.right_metrics.velocity
                    }
                )
        
        return GestureResult(detected=False, gesture_name='draw_mode')
    
    def _detect_undo(self, state: BimanualState) -> GestureResult:
        """
        Undo: Left hand hold still + right hand swipe left.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='undo')
        
        if self._is_hand_still(state.left_metrics) and state.right_gesture == 'swipe':
            # Get swipe direction from right hand's gesture metadata
            # Note: This assumes swipe detector includes direction in metadata
            # which is already done - SwipeDetector returns direction in metadata
            is_leftward = state.right_metrics.velocity[0] < 0  # Negative vx = leftward
            
            if is_leftward:
                return GestureResult(
                    detected=True,
                    gesture_name='undo',
                    confidence=1.0,
                    metadata={'direction': 'left'}
                )
        
        return GestureResult(detected=False, gesture_name='undo')
    
    def _detect_quick_menu(self, state: BimanualState) -> GestureResult:
        """
        Quick menu: Left hand zoom in + right hand pinch.
        Complex sequence unlikely to happen accidentally.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='quick_menu')
        
        if state.left_gesture == 'zoom' and state.right_gesture == 'pinch':
            return GestureResult(
                detected=True,
                gesture_name='quick_menu',
                confidence=1.0,
                metadata={
                    'menu_pos': state.right_metrics.centroid
                }
            )
        
        return GestureResult(detected=False, gesture_name='quick_menu')
    
    def _detect_warp(self, state: BimanualState) -> GestureResult:
        """
        Warp/Teleport: Both hands pointing at different locations.
        """
        # Safety check
        if state.left_metrics is None or state.right_metrics is None:
            return GestureResult(detected=False, gesture_name='warp')
        
        if state.left_gesture == 'pointing' and state.right_gesture == 'pointing':
            # Check hands are far apart (pointing at different locations)
            if state.inter_hand_distance > 0.3:  # Hands far apart
                return GestureResult(
                    detected=True,
                    gesture_name='warp',
                    confidence=1.0,
                    metadata={
                        'source': state.left_metrics.tip_positions['index'],
                        'target': state.right_metrics.tip_positions['index']
                    }
                )
        
        return GestureResult(detected=False, gesture_name='warp')


class ComprehensiveGestureManager:
    """
    Extended gesture manager that handles both single-hand and two-hand gestures.
    """
    
    def __init__(self):
        # Import here to avoid circular dependency
        from gesture_detectors import GestureManager
        
        self.single_hand_mgr = GestureManager()
        self.bimanual_detector = BimanualGestureDetector()
        
    def process_hands(
        self,
        left_landmarks,
        right_landmarks,
        img_shape: Tuple[int, int, int]
    ) -> Dict[str, Dict[str, GestureResult]]:
        """
        Process both hands and return all detected gestures.
        
        Returns:
            {
                'left': {...},    # Left hand single gestures
                'right': {...},   # Right hand single gestures
                'bimanual': {...} # Two-hand gestures
            }
        """
        results = {
            'left': {},
            'right': {},
            'bimanual': {}
        }
        
        # Process each hand individually
        if left_landmarks is not None:
            results['left'] = self.single_hand_mgr.process_hand(
                left_landmarks, img_shape, 'left'
            )
        
        if right_landmarks is not None:
            results['right'] = self.single_hand_mgr.process_hand(
                right_landmarks, img_shape, 'right'
            )
        
        # Get hand metrics for bimanual detection
        left_metrics = None
        right_metrics = None
        
        if left_landmarks is not None:
            left_history = self.single_hand_mgr.history['left']
            if left_history:
                left_metrics = left_history[-1]
        
        if right_landmarks is not None:
            right_history = self.single_hand_mgr.history['right']
            if right_history:
                right_metrics = right_history[-1]
        
        # Detect bimanual gestures
        if left_metrics is not None and right_metrics is not None:
            results['bimanual'] = self.bimanual_detector.detect(
                left_metrics,
                right_metrics,
                results['left'],
                results['right']
            )
        
        return results
