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

# I have decided to keep it for later extensions !

import numpy as np
import time
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from source_code.detectors.gesture_detectors import HandMetrics, GestureResult, compute_hand_metrics
from source_code.utils.math_utils import euclidean


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
    
    def __init__(self, config=None):
        pass

    def detect(
        self,
        left_metrics: Optional[HandMetrics],
        right_metrics: Optional[HandMetrics],
        left_gestures: Dict[str, GestureResult],
        right_gestures: Dict[str, GestureResult]
    ) -> Dict[str, GestureResult]:
        """Placeholder for bimanual gesture detection. Returns empty dict."""
        return {}

    def _primary_gesture(self, gestures: Dict[str, GestureResult]): # -> str
        """Placeholder for primary gesture selection."""
        pass
    
    def _is_hand_still(self, metrics: HandMetrics, threshold: Optional[float] = None): # -> bool
        """Placeholder for hand stillness check."""
        pass

    def _detect_pan_scroll(self, state: BimanualState): # -> GestureResult
        """Placeholder for pan/scroll gesture."""
        pass
    
    def _detect_rotate(self, state: BimanualState): # -> GestureResult
        """Placeholder for rotate gesture."""
        pass
    
    def _detect_two_hand_resize(self, state: BimanualState): # -> GestureResult
        """Placeholder for two-hand resize gesture."""
        pass
    
    def _detect_precision_mode(self, state: BimanualState): # -> GestureResult
        """Placeholder for precision mode gesture."""
        pass
    
    def _detect_draw_mode(self, state: BimanualState): # -> GestureResult
        """Placeholder for draw mode gesture."""
        pass
    
    def _detect_undo(self, state: BimanualState): # -> GestureResult
        """Placeholder for undo gesture."""
        pass
    
    def _detect_quick_menu(self, state: BimanualState): # -> GestureResult
        """Placeholder for quick menu gesture."""
        pass
    
    def _detect_warp(self, state: BimanualState): # -> GestureResult
        """Placeholder for warp gesture."""
        pass


class ComprehensiveGestureManager:
    """
    Extended gesture manager that handles both single-hand and two-hand gestures.
    """
    
    def __init__(self, config=None):
        # Import here to avoid circular dependency
        from source_code.detectors.gesture_detectors import GestureManager
        
        self.single_hand_mgr = GestureManager()
        self.bimanual_detector = BimanualGestureDetector(config)

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
        
        # Detect bimanual gestures (placeholder - returns empty dict)
        if left_metrics is not None and right_metrics is not None:
            bimanual_results = self.bimanual_detector.detect(
                left_metrics, right_metrics,
                results['left'], results['right']
            )
            if bimanual_results:
                results['bimanual'] = bimanual_results
        
        return results
