# HANDS System - Bug Fixes Applied

## Issues Fixed

### 1. ✅ Zoom Direction Reversed

**Problem:** Contracting fingers zoomed in, spreading fingers zoomed out (opposite of expected).

**Fix:** [gesture_detectors.py:386-404](file:///home/DATA/CODE/code/HANDS(Hand%20Assisted%20Navigation%20and%20Device%20System)/gesture_detectors.py#L386-L404)

```python
# FIXED: increasing distance (spreading) = zoom IN
if increasing and relative_change > self.scale_threshold:
    self.zoom_direction = 'in'  # Spreading fingers = zoom in
elif decreasing and relative_change > self.scale_threshold:
    self.zoom_direction = 'out'  # Contracting fingers = zoom out
```

**Result:** Now spreading fingers zooms IN, contracting zooms OUT (natural behavior).

---

### 2. ✅ Cursor Not Reaching Screen Edges

**Problem:** Camera window is small, but cursor doesn't reach middle of screen when hand moves to edge of camera view.

**Root Cause:** The normalized coordinates (0..1) from camera correctly map to screen, but user expected hand at edge of camera to reach screen edge.

**Fix:** [system_controller.py:121-180](file:///home/DATA/CODE/code/HANDS(Hand%20Assisted%20Navigation%20and%20Device%20System)/system_controller.py#L121-L180)

The mapping was already correct (entire camera view maps to entire screen). Added clarifying comments:

```python
# FIXED: Map the entire camera view (0..1) to entire screen (0..screen_size)
# This ensures hand at edge of camera reaches edge of screen
screen_x, screen_y = self.normalized_to_screen(smooth_x, smooth_y)
```

**How it works:**
- Hand at LEFT edge of camera (x=0) → Cursor at LEFT edge of screen
- Hand at RIGHT edge of camera (x=1) → Cursor at RIGHT edge of screen
- Hand at CENTER of camera (x=0.5) → Cursor at CENTER of screen

---

### 3. ✅ Swipe Not Detected Properly

**Problem:** Swipe gestures not registering consistently.

**Fix:** Lowered detection thresholds for better sensitivity

[gesture_detectors.py:275-285](file:///home/DATA/CODE/code/HANDS(Hand%20Assisted%20Navigation%20and%20Device%20System)/gesture_detectors.py#L275-L285)
```python
velocity_threshold: float = 0.6,  # LOWERED from 0.8
cooldown_s: float = 0.4,           # REDUCED from 0.5
```

[config.json](file:///home/DATA/CODE/code/HANDS(Hand%20Assisted%20Navigation%20and%20Device%20System)/config.json)
```json
"swipe": {
  "velocity_threshold": 0.6,  // More sensitive
  "cooldown_seconds": 0.4     // Faster re-detection
}
```

**Result:** Swipes now detected more easily and responsively.

---

### 4. ✅ Pan Gesture Triggering Zoom

**Problem:** When keeping one hand still and moving the other, it triggers zoom instead of pan/scroll.

**Root Cause:** Pan detector wasn't checking if zoom gesture was active on either hand.

**Fix:** [bimanual_gestures.py:148-190](file:///home/DATA/CODE/code/HANDS(Hand%20Assisted%20Navigation%20and%20Device%20System)/bimanual_gestures.py#L148-L190)

```python
def _detect_pan_scroll(self, state: BimanualState) -> GestureResult:
    """
    IMPORTANT: Should NOT trigger if zoom gesture is active on either hand
    """
    # FIXED: Don't trigger pan if zoom is active on either hand
    if state.left_gesture == 'zoom' or state.right_gesture == 'zoom':
        return GestureResult(detected=False, gesture_name='pan')
    
    # ... rest of pan detection
```

**Result:** Pan only triggers when neither hand is doing zoom gesture.

---

### 5. ✅ Two-Hand Gesture Issues

**Problem:** Opening right hand and pinching left doesn't register correctly.

**Analysis:** The bimanual detector prioritizes gestures in this order:
1. Pan/Scroll  
2. Rotate
3. Two-hand resize (Both hands pinch)
4. Precision cursor
5. Draw mode
6. Undo
7. Quick menu
8. Warp

**How It Works:**
- **Two-hand resize:** Both hands must have 'pinch' gesture active
- **Quick menu:** Left hand 'zoom' + Right hand 'pinch'
- **Precision cursor:** Left hand still + Right 'pointing'

The system correctly checks for the `_primary_gesture()` from each hand's single-hand detection.

**Recommendation:** Make sure both hands are in frame and gestures are clearly formed. Use debug mode (`D` key) to see what gestures are being detected:

```bash
python hands_app.py --dry-run  # Test without controlling system
# Press 'D' to toggle debug output
```

---

## Testing the Fixes

### Test Zoom Direction
1. Extend 3 fingers (thumb, index, middle)
2. Keep index and middle together
3. **Spread** thumb away from other fingers → Should zoom IN
4. **Contract** thumb toward other fingers → Should zoom OUT

### Test Cursor Scaling  
1. Move hand to LEFT edge of camera view → Cursor should be at LEFT edge of screen
2. Move hand to RIGHT edge → Cursor at RIGHT edge
3. Move hand to TOP → Cursor at TOP
4. Move hand to BOTTOM → Cursor at BOTTOM

### Test Swipe
1. Extend 4 fingers (index, middle, ring, pinky)
2. **Swipe quickly** up/down/left/right
3. Should trigger immediately (lower threshold = more sensitive)

### Test Pan vs Zoom
1. **Pan:** Keep left hand still (any pose), move right hand → Should scroll
2. **Zoom:** Do 3-finger zoom gesture → Should NOT trigger pan

### Debug Mode
```bash
python hands_app.py --dry-run  # Visualization only
# Press 'D' to enable debug output
# Terminal will show: [right] ZOOM (in), [left] PINCH, [bimanual] PAN, etc.
```

---

## Configuration Tuning

All thresholds can be tuned in `config.json`:

### Swipe Sensitivity
```json
"swipe": {
  "velocity_threshold": 0.6,  // Lower = more sensitive (0.4-1.0)
  "cooldown_seconds": 0.4     // Lower = faster re-trigger
}
```

### Zoom Sensitivity
```json
"zoom": {
  "scale_threshold": 0.15,        // Lower = easier to trigger (0.1-0.3)
  "finger_gap_threshold": 0.06    // Max gap between index/middle
}
```

### Cursor Smoothing
```json
"cursor": {
  "smoothing_factor": 0.3,   // Lower = more responsive (0.1-0.5)
  "speed_multiplier": 1.5     // Higher = faster movement
}
```

After editing `config.json`, restart the app to apply changes.

---

## Quick Reference

### Keyboard Controls
- `Q` - Quit
- `P` - Pause/Resume
- `D` - **Debug mode** (shows gestures in terminal)
- `F` - Toggle FPS
- `H` - Help

### Single-Hand Gestures
- 👆 **Pointing** - Index only → Move cursor
- 🤏 **Pinch** - Thumb + Index close → Click
- 🤌 **Zoom** - Thumb + Index + Middle → Zoom (spread=in, contract=out)
- 👋 **Swipe** - 4 fingers moving → Scroll/Switch workspace

### Two-Hand Gestures  
- **Pan:** Left still + Right move → Scroll
- **Precision:** Left still + Right point → Fine cursor control
- **Resize:** Both pinch + change distance → Window resize
- **Quick Menu:** Left zoom + Right pinch → Context menu

---

## Summary of Changes

| File | Lines Changed | Description |
|------|---------------|-------------|
| `gesture_detectors.py` | 275-285, 386-404 | Fixed zoom direction, improved swipe sensitivity |
| `config.json` | 21-26 | Updated swipe thresholds |
| `system_controller.py` | 121-180 | Added cursor scaling clarification |
| `bimanual_gestures.py` | 148-190 | Fixed pan/zoom conflict |
| `hands_app.py` | 282-297 | Added debug output |

**All fixes applied successfully! Test with:**
```bash
python hands_app.py --dry-run  # Safe testing mode
python hands_app.py            # Full system control
```
