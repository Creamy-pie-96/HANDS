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
        
        # Gesture debug toggles (keyboard: Z, P, I, S, O, T)
        self.show_gesture_debug = {
            'zoom': False,
            'pinch': False,
            'pointing': False,
            'swipe': False,
            'open_hand': False,
            'thumbs': False
        }
        
        # Position cache for debug panels
        self.debug_panel_positions = {}
        self.debug_panel_layout_dirty = True
        
        # Load config
        if config:
            from source_code.config.config_manager import get_visual_setting
            self.enabled = get_visual_setting('enabled', True)
            self.show_skeleton = get_visual_setting('show_hand_skeleton', True)
            self.show_fingertips = get_visual_setting('show_fingertips', True)
            self.show_cursor = get_visual_setting('show_cursor_preview', True)
            self.show_gesture_name = get_visual_setting('show_gesture_name', True)
            self.opacity = get_visual_setting('overlay_opacity', 0.7)
            self.velocity_arrow_scale = get_visual_setting('velocity_arrow_scale', 0.25)
            self.velocity_threshold_highlight = get_visual_setting('velocity_threshold_highlight', 0.8)
            self.fingertip_dim_factor = get_visual_setting('fingertip_dim_factor', 0.4)
            
            # Debug panel settings  
            self.debug_margin = config.get('visual_feedback', 'debug_panel', 'margin', default=12)
            self.debug_spacing = config.get('visual_feedback', 'debug_panel', 'spacing', default=8)
            self.debug_base_width = config.get('visual_feedback', 'debug_panel', 'base_width', default=240)
            self.debug_line_height = config.get('visual_feedback', 'debug_panel', 'line_height', default=16)
            self.debug_title_height = config.get('visual_feedback', 'debug_panel', 'title_height', default=20)
            self.debug_bg_alpha = config.get('visual_feedback', 'debug_panel', 'background_alpha', default=0.45)
            self.debug_spacing_buffer = config.get('visual_feedback', 'debug_panel', 'spacing_buffer', default=4)
            self.debug_start_y_offset = config.get('visual_feedback', 'debug_panel', 'start_y_offset', default=8)
            self.debug_scan_h = config.get('visual_feedback', 'debug_panel', 'scan_step_horizontal', default=30)
            self.debug_scan_v = config.get('visual_feedback', 'debug_panel', 'scan_step_vertical', default=20)
            
            self.panel_max_height = config.get('visual_feedback', 'gesture_panel', 'max_height', default=200)
            self.panel_y = config.get('visual_feedback', 'gesture_panel', 'panel_y', default=10)
            self.panel_left_x = config.get('visual_feedback', 'gesture_panel', 'panel_left_x', default=10)
            self.panel_width = config.get('visual_feedback', 'gesture_panel', 'panel_width', default=300)
            self.panel_overlay_alpha = config.get('visual_feedback', 'gesture_panel', 'overlay_alpha', default=0.7)
            self.panel_frame_blend = config.get('visual_feedback', 'gesture_panel', 'frame_blend', default=0.3)
            self.panel_title_y_offset = config.get('visual_feedback', 'gesture_panel', 'title_y_offset', default=30)
            self.panel_title_x = config.get('visual_feedback', 'gesture_panel', 'title_x', default=20)
            self.panel_sep_y_offset = config.get('visual_feedback', 'gesture_panel', 'separator_y_offset', default=30)
            self.panel_sep_left_x = config.get('visual_feedback', 'gesture_panel', 'separator_left_x', default=20)
            self.panel_sep_right_x = config.get('visual_feedback', 'gesture_panel', 'separator_right_x', default=290)
            self.panel_sep_bottom = config.get('visual_feedback', 'gesture_panel', 'separator_bottom_spacing', default=20)
            self.panel_max_gestures = config.get('visual_feedback', 'gesture_panel', 'max_gestures_display', default=4)
            self.panel_indicator_x = config.get('visual_feedback', 'gesture_panel', 'gesture_indicator_x', default=25)
            self.panel_indicator_r = config.get('visual_feedback', 'gesture_panel', 'gesture_indicator_radius', default=5)
            self.panel_name_x = config.get('visual_feedback', 'gesture_panel', 'gesture_name_x', default=40)
            self.panel_name_y_adj = config.get('visual_feedback', 'gesture_panel', 'gesture_name_y_adjust', default=5)
            self.panel_hint_x = config.get('visual_feedback', 'gesture_panel', 'hint_x', default=180)
            self.panel_line_spacing = config.get('visual_feedback', 'gesture_panel', 'line_spacing', default=20)
            self.panel_param_indent = config.get('visual_feedback', 'gesture_panel', 'param_indent_x', default=45)
            self.panel_param_spacing = config.get('visual_feedback', 'gesture_panel', 'param_line_spacing', default=15)
            self.panel_spacing_hint = config.get('visual_feedback', 'gesture_panel', 'spacing_with_hint', default=25)
            self.panel_spacing_no_hint = config.get('visual_feedback', 'gesture_panel', 'spacing_no_hint', default=25)
            self.panel_no_gesture_x = config.get('visual_feedback', 'gesture_panel', 'no_gesture_x', default=40)
            
            self.cursor_trail_fade = config.get('visual_feedback', 'cursor_preview', 'trail_fade_time', default=0.5)
            self.cursor_circle_r = config.get('visual_feedback', 'cursor_preview', 'circle_radius', default=15)
            self.cursor_cross_len = config.get('visual_feedback', 'cursor_preview', 'crosshair_length', default=20)
            self.cursor_cross_gap = config.get('visual_feedback', 'cursor_preview', 'crosshair_gap', default=8)
            
            self.pulse_freq = config.get('visual_feedback', 'animation', 'pulse_frequency', default=2.0)
        else:
            self.enabled = True
            self.show_skeleton = True
            self.show_fingertips = True
            self.show_cursor = True
            self.show_gesture_name = True
            self.opacity = 0.7
            self.velocity_arrow_scale = 0.25
            self.velocity_threshold_highlight = 0.8
            self.fingertip_dim_factor = 0.4
            self.debug_margin = 12
            self.debug_spacing = 8
            self.debug_base_width = 240
            self.debug_line_height = 16
            self.debug_title_height = 20
            self.debug_bg_alpha = 0.45
            self.debug_spacing_buffer = 4
            self.debug_start_y_offset = 8
            self.debug_scan_h = 30
            self.debug_scan_v = 20
            
            self.panel_max_height = 200
            self.panel_y = 10
            self.panel_left_x = 10
            self.panel_width = 300
            self.panel_overlay_alpha = 0.7
            self.panel_frame_blend = 0.3
            self.panel_title_y_offset = 30
            self.panel_title_x = 20
            self.panel_sep_y_offset = 30
            self.panel_sep_left_x = 20
            self.panel_sep_right_x = 290
            self.panel_sep_bottom = 20
            self.panel_max_gestures = 4
            self.panel_indicator_x = 25
            self.panel_indicator_r = 5
            self.panel_name_x = 40
            self.panel_name_y_adj = 5
            self.panel_hint_x = 180
            self.panel_line_spacing = 20
            self.panel_param_indent = 45
            self.panel_param_spacing = 15
            self.panel_spacing_hint = 25
            self.panel_spacing_no_hint = 25
            self.panel_no_gesture_x = 40
            
            self.cursor_trail_fade = 0.5
            self.cursor_circle_r = 15
            self.cursor_cross_len = 20
            self.cursor_cross_gap = 8
            
            self.pulse_freq = 2.0
    
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

    def _draw_velocity_info(self, frame, metrics):
        """Draw a velocity arrow from centroid and show numeric speed.

        - Arrow: from centroid in direction of velocity scaled for visibility.
        - Numeric speed: shown next to arrow (normalized units/sec).
        - Color: `swipe` color when speed >= threshold, dim otherwise.
        """
        h, w = frame.shape[:2]

        vx, vy = metrics.velocity
        speed = float(np.hypot(vx, vy))

        # Pixel coordinates of centroid
        cx, cy = metrics.centroid
        start = (int(cx * w), int(cy * h))

        # Scale velocity to pixels for arrow length. Use diag to keep consistent
        diag = np.hypot(w, h)
        # scale factor chosen so that 1.0 normalized/sec maps to arrow_scale*diag pixels
        scale = self.velocity_arrow_scale * diag
        end_x = int(start[0] + vx * scale)
        end_y = int(start[1] + vy * scale)
        end = (end_x, end_y)

        # Choose color based on threshold
        if speed >= self.velocity_threshold_highlight:
            arrow_color = self.colors.swipe
            text_color = self.colors.swipe
            thickness = 3
        else:
            # Dimmed accent when below threshold
            arrow_color = tuple(int(c * self.fingertip_dim_factor) for c in self.colors.accent)
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
        
        # Draw gesture debug overlays for all gestures based on toggle state
        self._draw_gesture_debug_overlays(frame, metrics, gestures, hand_label)
    
    def _draw_gesture_debug_overlays(self, frame, metrics, gestures, hand_label):
        """Draw debug metadata overlays with intelligent dynamic positioning."""
        h, w = frame.shape[:2]
        
        # Collect active debug gestures
        meta_keys = ['__zoom_meta', '__pinch_meta', '__pointing_meta', '__swipe_meta', '__open_hand_meta', '__thumbs_meta']
        gesture_names = ['zoom', 'pinch', 'pointing', 'swipe', 'open_hand', 'thumbs']
        
        active_panels = []
        for meta_key, gesture_name in zip(meta_keys, gesture_names):
            if not self.show_gesture_debug.get(gesture_name, False):
                continue
            
            if meta_key not in gestures:
                continue
            
            result = gestures[meta_key]
            if not result or not getattr(result, 'metadata', None):
                continue
            
            meta = result.metadata
            params = self._format_gesture_metadata(gesture_name, meta)
            
            if not params:
                continue
            
            active_panels.append({
                'name': gesture_name,
                'result': result,
                'params': params
            })
        
        # Early exit if no active panels
        if not active_panels:
            self.debug_panel_layout_dirty = True
            return
        
        # Recalculate layout if needed (also clear old positions when set changes)
        active_names = set(p['name'] for p in active_panels)
        cached_names = set(self.debug_panel_positions.keys())
        
        if self.debug_panel_layout_dirty or active_names != cached_names:
            # Clear positions for removed panels
            for name in list(self.debug_panel_positions.keys()):
                if name not in active_names:
                    del self.debug_panel_positions[name]
            
            self._calculate_debug_panel_layout(active_panels, w, h)
            self.debug_panel_layout_dirty = False
        
        # Calculate total bounding box for background
        if active_panels and self.debug_panel_positions:
            min_x = min(pos['x'] for pos in self.debug_panel_positions.values())
            min_y = min(pos['y'] for pos in self.debug_panel_positions.values())
            max_x = max(pos['x'] + pos['width'] for pos in self.debug_panel_positions.values())
            max_y = max(pos['y'] + pos['height'] for pos in self.debug_panel_positions.values())
            
            # Add padding
            padding = 8
            bg_tl = (max(0, min_x - padding), max(0, min_y - padding))
            bg_br = (min(w, max_x + padding), min(h, max_y + padding))
            
            # Draw dynamic translucent background
            overlay = frame.copy()
            cv2.rectangle(overlay, bg_tl, bg_br, self.colors.background, -1)
            cv2.addWeighted(overlay, self.debug_bg_alpha, frame, 1.0 - self.debug_bg_alpha, 0, frame)
        
        # Draw each panel at its cached position
        for panel_info in active_panels:
            gesture_name = panel_info['name']
            if gesture_name not in self.debug_panel_positions:
                continue
            
            pos = self.debug_panel_positions[gesture_name]
            bx, by = pos['x'], pos['y']
            
            # Title
            title_color = self.colors.active if panel_info['result'].detected else self.colors.text_secondary
            cv2.putText(frame, f"{gesture_name.upper()}:", (bx, by),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2, cv2.LINE_AA)
            by += 20
            
            # Metadata
            meta_color = self.colors.accent
            for param in panel_info['params']:
                cv2.putText(frame, param, (bx, by),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.48, meta_color, 2, cv2.LINE_AA)
                by += 16
    
    def _calculate_debug_panel_layout(self, active_panels, frame_w, frame_h):
        """Calculate optimal positions for debug panels using intelligent layout."""
        margin = self.debug_margin
        spacing = self.debug_spacing
        base_width = self.debug_base_width
        line_height = self.debug_line_height
        title_height = self.debug_title_height
        
        # Calculate dimensions for each panel
        for panel in active_panels:
            panel['width'] = base_width
            panel['height'] = title_height + len(panel['params']) * line_height
        
        # Start from top-right
        start_x = frame_w - base_width - margin
        start_y = margin + self.debug_start_y_offset
        
        # Available area (top-right quadrant preference)
        occupied_regions = []
        
        for panel in active_panels:
            name = panel['name']
            pw = panel['width']
            ph = panel['height']
            
            # Try to find best position
            best_pos = self._find_best_position(
                pw, ph, start_x, start_y, 
                frame_w, frame_h, margin, spacing,
                occupied_regions
            )
            
            # Cache the position
            self.debug_panel_positions[name] = {
                'x': best_pos[0],
                'y': best_pos[1],
                'width': pw,
                'height': ph
            }
            
            # Mark region as occupied
            occupied_regions.append({
                'x': best_pos[0],
                'y': best_pos[1],
                'width': pw,
                'height': ph
            })
    
    def _find_best_position(self, width, height, start_x, start_y, 
                           frame_w, frame_h, margin, spacing, occupied):
        """Find best non-overlapping position for a panel."""
        
        # Try positions in order of preference:
        # 0. Start position (first panel)
        # 1. Horizontally to the left (same row if space available)
        # 2. Vertically stack below
        # 3. New column to the left
        
        candidates = []
        
        # Strategy 0: Start position (first panel or fallback)
        if self._is_position_valid(start_x, start_y, width, height, frame_w, frame_h, margin, occupied):
            candidates.append((start_x, start_y, 0))  # priority 0 (best)
        
        # Strategy 1: Place to the left of existing panels (horizontal preference)
        if occupied:
            # Sort by x position to find rightmost panels
            sorted_by_x = sorted(occupied, key=lambda r: r['x'], reverse=True)
            
            for region in sorted_by_x:
                # Try to the left of this region at same y
                x = region['x'] - width - spacing
                y = region['y']
                if self._is_position_valid(x, y, width, height, frame_w, frame_h, margin, occupied):
                    candidates.append((x, y, 1))  # priority 1
                    break  # Take first valid horizontal position
        
        # Strategy 2: Stack below existing panels in same column
        if occupied:
            # Sort by y position to find placement opportunities
            sorted_by_y = sorted(occupied, key=lambda r: r['y'])
            
            for region in sorted_by_y:
                # Try below this region
                x = region['x']
                y = region['y'] + region['height'] + spacing
                if self._is_position_valid(x, y, width, height, frame_w, frame_h, margin, occupied):
                    candidates.append((x, y, 2))  # priority 2
                    break  # Take first valid vertical position
        
        # Strategy 3: New column to the far left
        if occupied:
            leftmost = min(r['x'] for r in occupied)
            x = leftmost - width - spacing
            y = start_y
            if self._is_position_valid(x, y, width, height, frame_w, frame_h, margin, occupied):
                candidates.append((x, y, 3))  # priority 3
        
        # Strategy 4: Scan for any valid position (last resort)
        if not candidates:
            # Scan horizontally first (prefer horizontal placement)
            for x in range(frame_w - width - margin, margin, -30):
                for y in range(margin, frame_h - height - margin, 20):
                    if self._is_position_valid(x, y, width, height, frame_w, frame_h, margin, occupied):
                        candidates.append((x, y, 4))
                        break
                if candidates:
                    break
        
        # Return best candidate (lowest priority number)
        if candidates:
            candidates.sort(key=lambda c: c[2])
            return (candidates[0][0], candidates[0][1])
        
        # Absolute fallback: top-left corner
        return (margin, margin)
    
    def _is_position_valid(self, x, y, width, height, frame_w, frame_h, margin, occupied):
        """Check if position is valid (in bounds and doesn't overlap)."""
        # Check frame bounds
        if x < margin or y < margin:
            return False
        if x + width > frame_w - margin or y + height > frame_h - margin:
            return False
        
        # Check overlap with occupied regions (add spacing buffer from config)
        for region in occupied:
            if self._rectangles_overlap(
                x, y, width, height,
                region['x'], region['y'], region['width'], region['height'],
                buffer=self.debug_spacing_buffer
            ):
                return False
        
        return True
    
    def _rectangles_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2, buffer=0):
        """Check if two rectangles overlap (with optional buffer spacing)."""
        # Expand rectangles by buffer to ensure spacing
        return not (
            x1 + w1 + buffer < x2 or 
            x2 + w2 + buffer < x1 or 
            y1 + h1 + buffer < y2 or 
            y2 + h2 + buffer < y1
        )
    
    def _format_gesture_metadata(self, gesture_name, meta):
        """Format gesture metadata into displayable strings."""
        params = []
        
        if gesture_name == 'zoom':
            params = [
                f"Gap:{meta.get('finger_gap', 0):.3f}",
                f"Spr:{meta.get('spread', 0):.3f}",
                f"EVel:{meta.get('ewma_velocity', 0):.3f}",
                f"Conf:{meta.get('confidence', 0):.2f}",
                f"Dir:{meta.get('direction', 'none')}",
                f"dS:{meta.get('delta_spread', 0):.3f}"
            ]
            if meta.get('reason'):
                params.append(f"Rsn:{meta['reason'][:12]}")
        
        elif gesture_name == 'pinch':
            params = [
                f"Dist:{meta.get('dist_rel', 0):.3f}",
                f"Thrs:{meta.get('threshold', 0):.3f}",
                f"Hold:{meta.get('hold_count', 0)}/{meta.get('hold_frames_needed', 0)}",
                f"CDwn:{meta.get('cooldown_remaining', 0):.1f}s"
            ]
        
        elif gesture_name == 'pointing':
            ewma_spd = meta.get('ewma_speed', meta.get('speed', 0))
            params = [
                f"Dist:{meta.get('distance', 0):.3f}",
                f"MinD:{meta.get('min_extension_ratio', 0):.3f}",
                f"Spd:{ewma_spd:.3f}",
                f"MaxS:{meta.get('max_speed', 0):.2f}",
                f"Alpha:{meta.get('ewma_alpha', 0.4):.2f}",
                f"Xtra:{meta.get('extra_fingers_count', 0)}/{meta.get('max_extra_fingers', 0)}"
            ]
            if meta.get('reason'):
                params.append(f"Rsn:{meta['reason'][:12]}")
        
        elif gesture_name == 'swipe':
            params = [
                f"Dir:{str(meta.get('current_direction', meta.get('direction', 'none')))[:4]}",
                f"VelX:{meta.get('velocity_threshold_x', 0):.2f}",
                f"VelY:{meta.get('velocity_threshold_y', 0):.2f}",
                f"Conf:{meta.get('confidence', 0):.2f}"
            ]
            if meta.get('reason'):
                params.append(f"Rsn:{meta['reason'][:12]}")
        
        elif gesture_name == 'open_hand':
            params = [
                f"Cnt:{meta.get('finger_count', 0)}/{meta.get('min_fingers', 0)}",
                f"TIDist:{meta.get('thumb_index_dist', 0):.3f}",
                f"Pinch:{meta.get('is_pinching', False)}"
            ]
            if meta.get('reason'):
                params.append(f"Rsn:{meta['reason'][:12]}")
        
        elif gesture_name == 'thumbs':
            vel = meta.get('velocity', (0, 0))
            params = [
                f"Vel:({vel[0]:.2f},{vel[1]:.2f})"
            ]
        
        return params
    
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
                            if time.time() - t < self.cursor_trail_fade]
        
        # Draw trail
        for i, (tx, ty, t) in enumerate(self.cursor_trail):
            age = time.time() - t
            alpha = 1.0 - (age / self.cursor_trail_fade)
            radius = int(3 + 2 * alpha)
            color = tuple(int(c * alpha * 0.5) for c in self.colors.cursor_trail)
            cv2.circle(frame, (tx, ty), radius, color, -1, cv2.LINE_AA)
        
        # Draw current cursor
        color = self.colors.cursor_preview if active else self.colors.inactive
        cv2.circle(frame, (px, py), self.cursor_circle_r, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        
        # Draw crosshair
        cv2.line(frame, (px - self.cursor_cross_len, py), (px - self.cursor_cross_gap, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px + self.cursor_cross_gap, py), (px + self.cursor_cross_len, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py - self.cursor_cross_len), (px, py - self.cursor_cross_gap), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py + self.cursor_cross_gap), (px, py + self.cursor_cross_len), color, 2, cv2.LINE_AA)
    
    def draw_gesture_panel(self, frame, all_gestures, status_text=""):
        """Draw modern gesture status panel."""
        if not self.enabled:
            return
        
        h, w = frame.shape[:2]
        panel_height = min(self.panel_max_height, h // 3)
        panel_y = self.panel_y
        
        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.panel_left_x, panel_y), (self.panel_width, panel_y + panel_height), 
                     self.colors.background, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, self.panel_overlay_alpha, frame, self.panel_frame_blend, 0, frame)
        
        # Border
        cv2.rectangle(frame, (self.panel_left_x, panel_y), (self.panel_width, panel_y + panel_height), 
                     self.colors.accent, 2, cv2.LINE_AA)
        
        # Title
        y = panel_y + self.panel_title_y_offset
        cv2.putText(frame, "HANDS Control", (self.panel_title_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors.accent, 2, cv2.LINE_AA)
        
        y += self.panel_sep_y_offset
        cv2.line(frame, (self.panel_sep_left_x, y), (self.panel_sep_right_x, y), self.colors.accent, 1, cv2.LINE_AA)
        y += self.panel_sep_bottom
        
        # Active gestures
        active_count = 0
        for category, gestures in all_gestures.items():
            if not gestures:
                continue
            
            for gesture_name, result in gestures.items():
                # Skip metadata preview entries (they start with __)
                if gesture_name.startswith('__'):
                    continue
                
                if active_count >= self.panel_max_gestures:
                    break
                
                # Gesture icon and name
                color = self._get_gesture_color(gesture_name)
                cv2.circle(frame, (self.panel_indicator_x, y), self.panel_indicator_r, color, -1, cv2.LINE_AA)
                
                text = f"{gesture_name.upper()}"
                cv2.putText(frame, text, (self.panel_name_x, y + self.panel_name_y_adj), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.text_primary, 1, cv2.LINE_AA)
                
                # Metadata hint with detailed zoom info
                if gesture_name == 'zoom' and result.metadata:
                    zoom_type = result.metadata.get('zoom_type', 'N/A')
                    hint = f"({zoom_type})"
                    cv2.putText(frame, hint, (self.panel_hint_x, y + self.panel_name_y_adj), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.text_secondary, 1, cv2.LINE_AA)
                    y += self.panel_line_spacing
                    
                    params = [
                        f"Gap: {result.metadata.get('finger_gap', 0):.3f}",
                        f"Spread: {result.metadata.get('spread', 0):.3f}",
                        f"dS: {result.metadata.get('delta_spread', 0):.3f}",
                        f"EVel: {result.metadata.get('ewma_velocity', 0):.3f}",
                        f"Conf: {result.metadata.get('confidence', 0):.2f}",
                        f"Dir: {result.metadata.get('direction', 'none')}"
                    ]

                    for param in params:
                        cv2.putText(frame, param, (self.panel_param_indent, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors.text_secondary, 1, cv2.LINE_AA)
                        y += self.panel_param_spacing
                    
                    y += self.panel_spacing_hint
                else:
                    y += self.panel_spacing_no_hint
                
                active_count += 1
        
        # Show "No gestures" if nothing active
        if active_count == 0:
            cv2.putText(frame, "No active gestures", (self.panel_no_gesture_x, y), 
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
