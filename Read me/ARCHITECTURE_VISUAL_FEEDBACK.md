# Visual Feedback System

This document explains the visual feedback and overlay system in HANDS.

---

## Overview

The visual feedback system provides real-time debugging information overlaid on the camera feed. It helps users understand what the gesture detection system "sees" and why certain gestures may or may not be triggering.

---

## Components

### 1. Hand Skeleton

Draws lines connecting MediaPipe hand landmarks:

```python
# In visual_feedback.py
def draw_hand_skeleton(self, frame, landmarks, handedness):
    # MediaPipe provides 21 landmarks per hand
    # Connect them according to finger structure
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),     # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)            # Palm
    ]

    for start, end in connections:
        pt1 = self._landmark_to_pixel(landmarks[start])
        pt2 = self._landmark_to_pixel(landmarks[end])
        cv2.line(frame, pt1, pt2, color, thickness)
```

### 2. Fingertip Markers

Highlights fingertip positions with colored circles:

```python
def draw_fingertips(self, frame, landmarks, extended_fingers):
    fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    for i, tip_id in enumerate(fingertip_ids):
        pos = self._landmark_to_pixel(landmarks[tip_id])

        if extended_fingers[i]:
            # Bright color for extended fingers
            color = self.active_color
        else:
            # Dimmed color for folded fingers
            color = self._dim_color(self.active_color, self.fingertip_dim_factor)

        cv2.circle(frame, pos, radius, color, -1)
```

### 3. Velocity Arrows

Shows hand movement direction and speed:

```python
def draw_velocity_arrow(self, frame, landmarks, velocity):
    # Arrow from palm center in direction of velocity
    palm_center = self._get_palm_center(landmarks)

    # Scale arrow by velocity magnitude
    arrow_end = (
        palm_center[0] + velocity[0] * self.velocity_arrow_scale,
        palm_center[1] + velocity[1] * self.velocity_arrow_scale
    )

    # Color based on whether velocity exceeds threshold
    if np.linalg.norm(velocity) > self.velocity_threshold_highlight:
        color = self.highlight_color  # Above threshold
    else:
        color = self.normal_color     # Below threshold

    cv2.arrowedLine(frame, palm_center, arrow_end, color, thickness)
```

### 4. Cursor Preview

Shows where the cursor would move:

```python
def draw_cursor_preview(self, frame, cursor_pos, trail_points):
    # Draw trail (fading line of previous positions)
    for i, (pos, timestamp) in enumerate(trail_points):
        age = time.time() - timestamp
        if age < self.trail_fade_time:
            alpha = 1.0 - (age / self.trail_fade_time)
            color = self._with_alpha(self.cursor_color, alpha)
            cv2.circle(frame, pos, 2, color, -1)

    # Draw crosshair at current position
    x, y = cursor_pos
    cv2.circle(frame, (x, y), self.circle_radius, self.cursor_color, 2)
    cv2.line(frame, (x - self.crosshair_length, y),
             (x - self.crosshair_gap, y), self.cursor_color, 2)
    # ... other crosshair lines
```

---

## Debug Overlays

Each gesture has a dedicated debug overlay showing its internal state.

### Overlay Panel Structure

```
┌─────────────────────────────────────┐
│ GESTURE NAME                        │  <- Title
├─────────────────────────────────────┤
│ ● gesture_name          hint_text   │  <- Status line
│     param1: value                   │  <- Parameters
│     param2: value                   │
│     reason: why_not_detected        │
└─────────────────────────────────────┘
```

### Example: Zoom Debug Overlay

```python
def _build_zoom_overlay(self, detector_state):
    lines = [
        ("Gap", f"{detector_state['finger_gap']:.3f}"),
        ("Spr", f"{detector_state['spread']:.3f}"),
        ("Chg", f"{detector_state['spread_change']:.1%}"),
        ("Inr", f"{detector_state['confidence']:.2f}"),
        ("Vel", f"{detector_state['velocity']:.3f}"),
        ("VCon", f"{detector_state['velocity_consistency']:.2f}"),
    ]

    if not detector_state['detected']:
        lines.append(("Rsn", detector_state['reason']))

    return lines
```

### Toggle Keys

Each overlay is toggled independently:

```python
# In hands_app.py
self.show_gesture_debug = {
    'zoom': False,
    'pinch': False,
    'pointing': False,
    'swipe': False,
    'open_hand': False,
    'thumbs': False,
}

def handle_keypress(self, key):
    if key == ord('z'):
        self.show_gesture_debug['zoom'] = not self.show_gesture_debug['zoom']
    elif key == ord('x'):
        self.show_gesture_debug['pinch'] = not self.show_gesture_debug['pinch']
    # ... etc
```

---

## Panel Positioning

The system automatically finds non-overlapping positions for panels:

```python
def _find_panel_position(self, frame, panel_size, existing_panels):
    """Find an unoccupied area for the panel."""
    frame_h, frame_w = frame.shape[:2]

    # Start from top-right, scan for empty space
    for y in range(self.start_y_offset, frame_h, self.scan_step_vertical):
        for x in range(frame_w - panel_size[0], 0, self.scan_step_horizontal):
            rect = (x, y, panel_size[0], panel_size[1])

            if not self._overlaps_existing(rect, existing_panels):
                return (x, y)

    # Fallback: overlap is acceptable
    return (self.panel_left_x, self.panel_y)
```

---

## Color System

Colors are configurable and support different states:

```python
# In config.json
"colors": {
    "left_hand": [255, 200, 50],      # BGR
    "right_hand": [50, 200, 255],
    "cursor": [0, 255, 0],
    "active": [0, 255, 255],
    "inactive": [128, 128, 128],
    "background": [40, 40, 40]
}

# Usage
def get_hand_color(self, handedness):
    if handedness == 'Left':
        return self.colors['left_hand']
    else:
        return self.colors['right_hand']
```

---

## Pulsing Animation

Active gestures pulse to draw attention:

```python
def get_pulse_factor(self):
    """Returns 0.0-1.0 based on pulse frequency."""
    t = time.time()
    # Sinusoidal pulse
    return (math.sin(t * self.pulse_frequency * 2 * math.pi) + 1) / 2

def draw_active_indicator(self, frame, pos, gesture_active):
    if gesture_active:
        pulse = self.get_pulse_factor()
        radius = int(self.base_radius * (1 + 0.2 * pulse))
        alpha = 0.7 + 0.3 * pulse
        # Draw with pulsing size and opacity
```

---

## Frame Blending

Overlays use alpha blending for readability:

```python
def draw_panel_background(self, frame, rect):
    x, y, w, h = rect

    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h),
                  self.background_color, -1)

    # Blend with original frame
    cv2.addWeighted(overlay, self.overlay_alpha,
                    frame, 1 - self.overlay_alpha,
                    0, frame)
```

---

## Performance Considerations

1. **Conditional Rendering**: Only draw enabled overlays
2. **Cached Calculations**: Pre-compute landmark pixel positions once per frame
3. **Overlay Order**: Draw background → lines → text (minimize overdraw)
4. **Font Caching**: Use pre-loaded font objects

```python
def draw_all_feedback(self, frame, data):
    if not self.enabled:
        return frame

    # Pre-compute landmarks to pixels (used by multiple components)
    pixel_landmarks = self._landmarks_to_pixels(data['landmarks'])

    # Draw in order: back to front
    if self.show_hand_skeleton:
        self._draw_skeleton(frame, pixel_landmarks)

    if self.show_fingertips:
        self._draw_fingertips(frame, pixel_landmarks, data['extended'])

    # Draw debug panels last (on top)
    for gesture, enabled in self.show_gesture_debug.items():
        if enabled:
            self._draw_gesture_panel(frame, gesture, data)

    return frame
```

---

## Configuration Reference

Key visual_feedback config options:

| Category   | Parameter              | Default | Description             |
| ---------- | ---------------------- | ------- | ----------------------- |
| General    | `enabled`              | true    | Master toggle           |
| General    | `overlay_opacity`      | 0.7     | Background transparency |
| Skeleton   | `show_hand_skeleton`   | true    | Draw bone lines         |
| Fingertips | `show_fingertips`      | true    | Draw tip markers        |
| Fingertips | `fingertip_dim_factor` | 0.4     | Dimming for folded      |
| Velocity   | `velocity_arrow_scale` | 0.25    | Arrow scaling           |
| Cursor     | `trail_fade_time`      | 0.5s    | Trail duration          |
| Animation  | `pulse_frequency`      | 2.0 Hz  | Pulse speed             |

---

## Related Files

- `source_code/utils/visual_feedback.py` - Core drawing functions
- `source_code/config/config.json` - Visual settings
- `source_code/gui/status_indicator.py` - Floating indicators (separate from camera overlay)
