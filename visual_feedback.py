"""
Visual Feedback Overlay for HANDS

Provides beautiful, informative visual feedback for gesture control.
Inspired by modern UI/UX principles for gesture-based interfaces.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class UIColors:
    """Modern color palette for UI elements."""
    # Hand colors
    left_hand = (0, 255, 255)  # Cyan
    right_hand = (255, 128, 0)   # Orange
    
    # Gesture colors
    pinch = (255, 0, 255)        # Magenta
    zoom = (0, 255, 100)         # Green
    pointing = (100, 200, 255)   # Light blue
    swipe = (255, 255, 0)        # Yellow
    open_hand = (200, 150, 255)  # Purple
    
    # UI elements
    cursor_preview = (0, 255, 0)      # Green
    cursor_trail = (0, 200, 0)        # Dark green
    active_gesture = (255, 0, 255)    # Magenta
    background = (20, 20, 30)         # Dark blue-grey
    text_primary = (255, 255, 255)    # White
    text_secondary = (180, 180, 200)  # Light grey
    accent = (0, 200, 255)            # Bright cyan
    
    # State indicators
    active = (0, 255, 0)         # Green
    inactive = (100, 100, 120)   # Grey
    warning = (255, 165, 0)      # Orange
    error = (255, 50, 50)        # Red


class VisualFeedback:
    """
    Creates beautiful visual overlays for gesture feedback.
    """
    
    def __init__(self, config=None):
        """Initialize visual feedback system."""
        self.colors = UIColors()
        
        # Animation states
        self.cursor_trail = []
        self.gesture_pulses = {}  # gesture_name -> (timestamp, intensity)
        self.particle_effects = []  # For click/action feedback
        
        # Load config
        if config:
            from config_manager import get_visual_setting
            self.enabled = get_visual_setting('enabled', True)
            self.show_skeleton = get_visual_setting('show_hand_skeleton', True)
            self.show_fingertips = get_visual_setting('show_fingertips', True)
            self.show_cursor = get_visual_setting('show_cursor_preview', True)
            self.show_gesture_name = get_visual_setting('show_gesture_name', True)
            self.opacity = get_visual_setting('overlay_opacity', 0.7)
        else:
            self.enabled = True
            self.show_skeleton = True
            self.show_fingertips = True
            self.show_cursor = True
            self.show_gesture_name = True
            self.opacity = 0.7
    
    def draw_hand_overlay(self, frame, metrics, hand_label='right', gestures=None):
        """
        Draw beautiful hand tracking overlay.
        
        Args:
            frame: Image to draw on
            metrics: HandMetrics object
            hand_label: 'left' or 'right'
            gestures: Dict of active gestures
        """
        if not self.enabled or metrics is None:
            return
        
        h, w = frame.shape[:2]
        color = self.colors.left_hand if hand_label == 'left' else self.colors.right_hand
        
        # Draw hand skeleton with glow effect
        if self.show_skeleton:
            self._draw_hand_skeleton(frame, metrics, color)
        
        # Draw fingertips with status indicators
        if self.show_fingertips:
            self._draw_fingertips(frame, metrics, color)
        
        # Draw gesture-specific overlays
        if gestures:
            self._draw_gesture_overlays(frame, metrics, gestures, hand_label)
    
    def _draw_hand_skeleton(self, frame, metrics, color):
        """Draw hand skeleton with glow effect."""
        h, w = frame.shape[:2]
        landmarks = metrics.landmarks_norm
        
        # MediaPipe connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Draw connections with glow
        for start_idx, end_idx in connections:
            start = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
            end = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
            
            # Glow effect (thicker, semi-transparent line)
            cv2.line(frame, start, end, color, 6, cv2.LINE_AA)
            # Main line
            cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
    
    def _draw_fingertips(self, frame, metrics, color):
        """Draw fingertips with extension status."""
        h, w = frame.shape[:2]
        
        for finger_name, pos in metrics.tip_positions.items():
            px, py = int(pos[0] * w), int(pos[1] * h)
            is_extended = metrics.fingers_extended[finger_name]
            
            # Draw ring around fingertip
            if is_extended:
                # Extended: bright color with glow
                cv2.circle(frame, (px, py), 12, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)
            else:
                # Curled: dimmed
                dim_color = tuple(int(c * 0.4) for c in color)
                cv2.circle(frame, (px, py), 8, dim_color, 2, cv2.LINE_AA)
    
    def _draw_gesture_overlays(self, frame, metrics, gestures, hand_label):
        """Draw gesture-specific visual feedback."""
        h, w = frame.shape[:2]
        
        # Pinch: Draw line between thumb and index
        if 'pinch' in gestures:
            thumb_pos = metrics.tip_positions['thumb']
            index_pos = metrics.tip_positions['index']
            p1 = (int(thumb_pos[0] * w), int(thumb_pos[1] * h))
            p2 = (int(index_pos[0] * w), int(index_pos[1] * h))
            
            # Draw pulsing line
            intensity = self._get_pulse_intensity('pinch')
            line_color = self._blend_colors(self.colors.pinch, (255, 255, 255), intensity)
            cv2.line(frame, p1, p2, line_color, 3, cv2.LINE_AA)
            
            # Draw circles at pinch points
            cv2.circle(frame, p1, 8, line_color, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 8, line_color, -1, cv2.LINE_AA)
        
        # Zoom: Draw triangle between thumb, index, middle
        if 'zoom' in gestures:
            gesture_data = gestures['zoom']
            zoom_type = gesture_data.metadata.get('zoom_type', 'in')
            
            thumb_pos = metrics.tip_positions['thumb']
            index_pos = metrics.tip_positions['index']
            middle_pos = metrics.tip_positions['middle']
            
            p1 = np.array([int(thumb_pos[0] * w), int(thumb_pos[1] * h)])
            p2 = np.array([int(index_pos[0] * w), int(index_pos[1] * h)])
            p3 = np.array([int(middle_pos[0] * w), int(middle_pos[1] * h)])
            
            pts = np.array([p1, p2, p3], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Pulsing triangle
            intensity = self._get_pulse_intensity('zoom')
            color = self.colors.zoom if zoom_type == 'in' else self.colors.warning
            fill_color = tuple(int(c * 0.3) for c in color)
            
            cv2.fillPoly(frame, [pts], fill_color, cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            
            # Draw zoom direction arrow
            center = (p1 + p2 + p3) // 3
            arrow_scale = 40 if zoom_type == 'in' else 30
            if zoom_type == 'in':
                cv2.arrowedLine(frame, tuple(center), tuple(center - [0, arrow_scale]), 
                              color, 3, cv2.LINE_AA, tipLength=0.4)
            else:
                cv2.arrowedLine(frame, tuple(center), tuple(center + [0, arrow_scale]), 
                              color, 3, cv2.LINE_AA, tipLength=0.4)
        
        # Pointing: Draw pointer ray
        if 'pointing' in gestures:
            index_pos = metrics.tip_positions['index']
            centroid = metrics.centroid
            
            p1 = (int(centroid[0] * w), int(centroid[1] * h))
            p2 = (int(index_pos[0] * w), int(index_pos[1] * h))
            
            # Extend ray
            direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            p3 = (int(p2[0] + direction[0] * 100), int(p2[1] + direction[1] * 100))
            
            # Draw ray with gradient effect
            intensity = self._get_pulse_intensity('pointing')
            color = self._blend_colors(self.colors.pointing, (255, 255, 255), intensity)
            
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
            cv2.line(frame, p2, p3, tuple(int(c * 0.5) for c in color), 2, cv2.LINE_AA)
            cv2.circle(frame, p2, 10, color, -1, cv2.LINE_AA)
    
    def draw_cursor_preview(self, frame, norm_x, norm_y, active=True):
        """Draw cursor preview with trail effect."""
        if not self.enabled or not self.show_cursor:
            return
        
        h, w = frame.shape[:2]
        px, py = int(norm_x * w), int(norm_y * h)
        
        # Add to trail
        self.cursor_trail.append((px, py, time.time()))
        # Keep trail length limited
        self.cursor_trail = [(x, y, t) for x, y, t in self.cursor_trail 
                            if time.time() - t < 0.5]
        
        # Draw trail
        for i, (tx, ty, t) in enumerate(self.cursor_trail):
            age = time.time() - t
            alpha = 1.0 - (age / 0.5)
            radius = int(3 + 2 * alpha)
            color = tuple(int(c * alpha * 0.5) for c in self.colors.cursor_trail)
            cv2.circle(frame, (tx, ty), radius, color, -1, cv2.LINE_AA)
        
        # Draw current cursor
        color = self.colors.cursor_preview if active else self.colors.inactive
        cv2.circle(frame, (px, py), 15, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        
        # Draw crosshair
        cv2.line(frame, (px - 20, py), (px - 8, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px + 8, py), (px + 20, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py - 20), (px, py - 8), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py + 8), (px, py + 20), color, 2, cv2.LINE_AA)
    
    def draw_gesture_panel(self, frame, all_gestures, status_text=""):
        """Draw modern gesture status panel."""
        if not self.enabled:
            return
        
        h, w = frame.shape[:2]
        panel_height = min(200, h // 3)
        panel_y = 10
        
        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (300, panel_y + panel_height), 
                     self.colors.background, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, panel_y), (300, panel_y + panel_height), 
                     self.colors.accent, 2, cv2.LINE_AA)
        
        # Title
        y = panel_y + 30
        cv2.putText(frame, "HANDS Control", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors.accent, 2, cv2.LINE_AA)
        
        y += 30
        cv2.line(frame, (20, y), (290, y), self.colors.accent, 1, cv2.LINE_AA)
        y += 20
        
        # Active gestures
        active_count = 0
        for category, gestures in all_gestures.items():
            if not gestures:
                continue
            
            for gesture_name, result in gestures.items():
                if active_count >= 4:  # Limit display
                    break
                
                # Gesture icon and name
                color = self._get_gesture_color(gesture_name)
                cv2.circle(frame, (25, y), 5, color, -1, cv2.LINE_AA)
                
                text = f"{gesture_name.upper()}"
                cv2.putText(frame, text, (40, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.text_primary, 1, cv2.LINE_AA)
                
                # Metadata hint
                if 'zoom_type' in result.metadata:
                    hint = f"({result.metadata['zoom_type']})"
                    cv2.putText(frame, hint, (180, y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.text_secondary, 1, cv2.LINE_AA)
                
                y += 25
                active_count += 1
        
        # Show "No gestures" if nothing active
        if active_count == 0:
            cv2.putText(frame, "No active gestures", (40, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.inactive, 1, cv2.LINE_AA)
        
        # Status text
        if status_text:
            y = panel_y + panel_height - 10
            cv2.putText(frame, status_text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.text_secondary, 1, cv2.LINE_AA)
    
    def _get_pulse_intensity(self, gesture_name, frequency=2.0):
        """Get pulsing intensity for gesture (0..1)."""
        t = time.time()
        return 0.5 + 0.5 * np.sin(t * frequency * 2 * np.pi)
    
    def _blend_colors(self, color1, color2, t):
        """Blend two colors by factor t (0..1)."""
        return tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
    
    def _get_gesture_color(self, gesture_name):
        """Get color for gesture type."""
        color_map = {
            'pinch': self.colors.pinch,
            'zoom': self.colors.zoom,
            'pointing': self.colors.pointing,
            'swipe': self.colors.swipe,
            'open_hand': self.colors.open_hand,
        }
        return color_map.get(gesture_name, self.colors.accent)
