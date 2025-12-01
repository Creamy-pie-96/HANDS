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

        # Draw velocity arrow and speed value for debugging (always visible)
        self._draw_velocity_info(frame, metrics)
        
        # Draw gesture-specific overlays
        if gestures:
            self._draw_gesture_overlays(frame, metrics, gestures, hand_label)
    
    def _draw_hand_skeleton(self, frame, metrics, color):
        """Draw hand skeleton - OPTIMIZED for performance."""
        h, w = frame.shape[:2]
        landmarks = metrics.landmarks_norm
        
        # MediaPipe connections - simplified  
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
        
        # Draw connections - SIMPLE, no glow effect for performance
        for start_idx, end_idx in connections:
            start = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
            end = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
            # Just one line, no glow
            cv2.line(frame, start, end, color, 2)
    
    def _draw_fingertips(self, frame, metrics, color):
        """Draw fingertips - OPTIMIZED."""
        h, w = frame.shape[:2]
        
        for finger_name, pos in metrics.tip_positions.items():
            px, py = int(pos[0] * w), int(pos[1] * h)
            is_extended = metrics.fingers_extended[finger_name]
            
            # Simple circles - no fancy rendering for performance
            if is_extended:
                cv2.circle(frame, (px, py), 6, color, -1)
            else:
                dim_color = tuple(int(c * 0.4) for c in color)
                cv2.circle(frame, (px, py), 4, dim_color, -1)

    def _draw_velocity_info(self, frame, metrics, swipe_threshold: float = 0.8):
        """Draw a velocity arrow from centroid and show numeric speed.

        - Arrow: from centroid in direction of velocity scaled for visibility.
        - Numeric speed: shown next to arrow (normalized units/sec).
        - Color: `swipe` color when speed >= `swipe_threshold`, dim otherwise.
        """
        h, w = frame.shape[:2]

        vx, vy = metrics.velocity
        speed = float(np.hypot(vx, vy))

        # Pixel coordinates of centroid
        cx, cy = metrics.centroid
        start = (int(cx * w), int(cy * h))

        # Scale velocity to pixels for arrow length. Use diag to keep consistent
        diag = np.hypot(w, h)
        # scale factor chosen so that 1.0 normalized/sec maps to ~0.25*diag pixels
        scale = 0.25 * diag
        end_x = int(start[0] + vx * scale)
        end_y = int(start[1] + vy * scale)
        end = (end_x, end_y)

        # Choose color based on threshold
        if speed >= swipe_threshold:
            arrow_color = self.colors.swipe
            text_color = self.colors.swipe
            thickness = 3
        else:
            # Dimmed accent when below threshold
            arrow_color = tuple(int(c * 0.4) for c in self.colors.accent)
            text_color = self.colors.text_secondary
            thickness = 2

        # Draw arrow (clamped to frame bounds)
        end_clamped = (max(0, min(w - 1, end[0])), max(0, min(h - 1, end[1])))
        cv2.arrowedLine(frame, start, end_clamped, arrow_color, thickness, tipLength=0.25)

        # Draw numeric speed next to the arrow tip
        speed_text = f"{speed:.2f}"
        text_pos = (end_clamped[0] + 8, end_clamped[1] - 8)
        cv2.putText(frame, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        # Also draw a small background circle at centroid for clarity
        cv2.circle(frame, start, 6, arrow_color, -1)
    
    def _draw_gesture_overlays(self, frame, metrics, gestures, hand_label):
        """Draw gesture-specific visual feedback - OPTIMIZED."""
        h, w = frame.shape[:2]
        
        # Pinch: Simple line between thumb and index
        if 'pinch' in gestures:
            thumb_pos = metrics.tip_positions['thumb']
            index_pos = metrics.tip_positions['index']
            p1 = (int(thumb_pos[0] * w), int(thumb_pos[1] * h))
            p2 = (int(index_pos[0] * w), int(index_pos[1] * h))
            cv2.line(frame, p1, p2, self.colors.pinch, 2)
        
        # Zoom: Simple triangle NO animations for FPS
        if 'zoom' in gestures:
            gesture_data = gestures['zoom']
            zoom_type = gesture_data.metadata.get('zoom_type', 'in')
            
            thumb_pos = metrics.tip_positions['thumb']
            index_pos = metrics.tip_positions['index']
            middle_pos = metrics.tip_positions['middle']
            
            p1 = np.array([int(thumb_pos[0] * w), int(thumb_pos[1] * h)])
            p2 = np.array([int(index_pos[0] * w), int(index_pos[1] * h)])
            p3 = np.array([int(middle_pos[0] * w), int(middle_pos[1] * h)])
            
            pts = np.array([p1, p2, p3], np.int32).reshape((-1, 1, 2))
            color = self.colors.zoom if zoom_type == 'in' else self.colors.warning
            cv2.polylines(frame, [pts], True, color, 2)
        
        # Pointing: Simple line
        if 'pointing' in gestures:
            index_pos = metrics.tip_positions['index']
            centroid = metrics.centroid
            p1 = (int(centroid[0] * w), int(centroid[1] * h))
            p2 = (int(index_pos[0] * w), int(index_pos[1] * h))
            cv2.line(frame, p1, p2, self.colors.pointing, 2)
            cv2.circle(frame, p2, 6, self.colors.pointing, -1)
    
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
                
                # Metadata hint with detailed zoom info
                if gesture_name == 'zoom' and result.metadata:
                    # Show zoom type
                    zoom_type = result.metadata.get('zoom_type', 'N/A')
                    hint = f"({zoom_type})"
                    cv2.putText(frame, hint, (180, y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.text_secondary, 1, cv2.LINE_AA)
                    y += 20
                    
                    # Show detailed zoom parameters
                    params = [
                        f"Gap: {result.metadata.get('finger_gap', 0):.3f}",
                        f"Spread: {result.metadata.get('spread', 0):.3f}",
                        f"Change: {result.metadata.get('relative_change', 0):.2%}",
                        f"Inertia: {result.metadata.get('inertia', 0):.2f}",
                        f"Vel: {result.metadata.get('avg_velocity', 0):.3f}",
                        f"VelCons: {result.metadata.get('velocity_consistency', 0):.2f}"
                    ]
                    
                    for param in params:
                        cv2.putText(frame, param, (45, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors.text_secondary, 1, cv2.LINE_AA)
                        y += 15
                    
                    y += 5  # Extra spacing after zoom details
                elif 'zoom_type' in result.metadata:
                    hint = f"({result.metadata['zoom_type']})"
                    cv2.putText(frame, hint, (180, y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.text_secondary, 1, cv2.LINE_AA)
                    y += 25
                else:
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
