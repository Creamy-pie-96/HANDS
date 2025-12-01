# HANDS Gesture Debug System - Complete Guide

## Overview

All gesture detectors now return full metadata consistently, enabling real-time parameter tuning and debugging. Each gesture has a keyboard toggle to display its live parameters on screen.

---

## Keyboard Controls

| Key | Gesture   | Toggle                         |
| --- | --------- | ------------------------------ |
| `Z` | Zoom      | Show/hide zoom parameters      |
| `X` | Pinch     | Show/hide pinch parameters     |
| `I` | Pointing  | Show/hide pointing parameters  |
| `S` | Swipe     | Show/hide swipe parameters     |
| `O` | Open Hand | Show/hide open hand parameters |
| `T` | Thumbs    | Show/hide thumbs parameters    |

**Note**: Multiple gesture debugs can be active simultaneously.

---

## Relative Measurements

All measurements in HANDS are **relative** to ensure consistency regardless of hand distance from camera:

### Already Relative Measurements:

1. **tip_distances**: Normalized (0-1) based on screen coordinates
2. **velocity**: Normalized by hand diagonal (distance-independent)
3. **diag_rel**: Hand size relative to image diagonal
4. **finger extension**: Uses ratios (tip distance / PIP distance from palm)
5. **centroid & bbox**: Normalized (0-1) coordinates
6. **All thresholds**: Configured relative to hand size or normalized coordinates

---

## Gesture-Specific Metadata

### 1. Zoom Detector

**Displayed Parameters:**

- `Gap`: Distance between index & middle fingers (must be paired)
- `Spr`: Spread distance (thumb to paired fingers)
- `Chg`: Relative spread change (percentage)
- `Inr`: Inertia/confidence level (0-1)
- `Vel`: Average velocity of spread change
- `VCon`: Velocity consistency score (0-1)
- `Rsn`: Reason for non-detection

**Metadata Keys:**

```python
{
    'finger_gap': float,          # Index-middle distance
    'spread': float,              # Thumb-fingers distance
    'relative_change': float,     # % change from baseline
    'inertia': float,             # Confidence (0-1)
    'avg_velocity': float,        # Spread change velocity
    'velocity_consistency': float, # Smoothness score (0-1)
    'zoom_type': str,             # 'in' or 'out'
    'reason': str                 # Why not detected (if applicable)
}
```

**Tuning Guide:**

- Low `inertia` → gesture building confidence
- High `velocity_consistency` → smooth motion
- Check `reason` field for why zoom isn't triggering

---

### 2. Pinch Detector

**Displayed Parameters:**

- `Dist`: Current thumb-index distance
- `Thrs`: Configured threshold
- `Hold`: Current hold count / required hold frames
- `CDwn`: Cooldown time remaining (seconds)

**Metadata Keys:**

```python
{
    'dist_rel': float,            # Thumb-index distance (normalized)
    'threshold': float,           # Detection threshold
    'hold_count': int,            # Frames held so far
    'hold_frames_needed': int,    # Required hold frames
    'in_cooldown': bool,          # Currently in cooldown?
    'cooldown_remaining': float   # Seconds until next detection
}
```

**Tuning Guide:**

- `dist_rel < threshold` → fingers close enough
- Watch `hold_count` to see stability
- `cooldown_remaining` prevents double-clicks

---

### 3. Pointing Detector

**Displayed Parameters:**

- `Dist`: Index finger distance from palm
- `MinD`: Minimum required distance
- `Spd`: Current hand speed
- `MaxS`: Maximum allowed speed
- `Xtra`: Extra fingers extended / max allowed
- `Rsn`: Reason for non-detection

**Metadata Keys:**

```python
{
    'tip_position': tuple,        # Index fingertip (x, y)
    'direction': tuple,           # Point direction vector
    'distance': float,            # Fingertip-to-palm distance
    'min_extension_ratio': float, # Required distance threshold
    'speed': float,               # Hand velocity magnitude
    'max_speed': float,           # Speed threshold
    'index_extended': bool,       # Index finger extended?
    'extra_fingers_count': int,   # Other fingers extended
    'max_extra_fingers': int,     # Tolerance for extra fingers
    'reason': str                 # Why not detected (if applicable)
}
```

**Tuning Guide:**

- `distance < min_extension_ratio` → finger too close
- `speed > max_speed` → moving too fast
- Check `reason` for specific failure

---

### 4. Swipe Detector

**Displayed Parameters:**

- `Dir`: Swipe direction (up/down/left/right)
- `Spd`: Current velocity
- `Thrs`: Velocity threshold
- `Hist`: History frames collected / min required
- `CDwn`: Cooldown remaining (seconds)
- `Rsn`: Reason for non-detection

**Metadata Keys:**

```python
{
    'direction': str,             # 'up', 'down', 'left', 'right'
    'speed': float,               # Velocity magnitude
    'velocity': tuple,            # (vx, vy) components
    'velocity_threshold': float,  # Required speed
    'history_size': int,          # Frames in history
    'min_history': int,           # Minimum frames needed
    'in_cooldown': bool,          # In cooldown?
    'cooldown_remaining': float,  # Seconds remaining
    'reason': str                 # Why not detected (if applicable)
}
```

**Tuning Guide:**

- `speed < velocity_threshold` → not fast enough
- Watch `history_size` fill up before detection possible
- `cooldown_remaining` prevents rapid re-triggering

---

### 5. Open Hand Detector

**Displayed Parameters:**

- `Cnt`: Fingers extended / minimum required
- `TIDist`: Thumb-index distance
- `Pinch`: Is hand in pinch position?
- `Rsn`: Reason for non-detection

**Metadata Keys:**

```python
{
    'finger_count': int,          # Total fingers extended
    'min_fingers': int,           # Required minimum (4 or 5)
    'thumb_index_dist': float,    # Distance between thumb & index
    'pinch_threshold': float,     # Pinch exclusion threshold
    'is_pinching': bool,          # Detected as pinch?
    'fingers_extended': dict,     # Per-finger extension state
    'reason': str                 # Why not detected (if applicable)
}
```

**Tuning Guide:**

- `finger_count < min_fingers` → not enough fingers
- `is_pinching = True` → excluded as pinch gesture
- Check `fingers_extended` for individual finger states

---

### 6. Thumbs Detector

**Displayed Parameters:**

- `Vel`: Velocity vector (vx, vy)

**Metadata Keys:**

```python
{
    'velocity': tuple             # (vx, vy) thumb movement
}
```

---

## Visual Feedback System

### How It Works:

1. All gesture detectors run every frame
2. Results stored under `__<gesture>_meta` keys (e.g., `__zoom_meta`)
3. When debug toggle is ON, metadata displayed near hand bounding box
4. Color-coded: **Green** (detected), **Gray** (not detected)

### Display Format:

```
GESTURE_NAME:
  Param1:value
  Param2:value
  Param3:value
  Rsn:reason_code
```

### Best Practices:

- Enable one gesture debug at a time for clarity
- Watch parameter values evolve in real-time
- Use `Rsn` (reason) field to diagnose issues
- Compare threshold values to current measurements

---

## Common Tuning Scenarios

### Zoom Too Sensitive:

- Increase `inertia_threshold` in config
- Increase `min_velocity` to ignore drift
- Increase `scale_threshold` for bigger changes

### Pinch Not Triggering:

- Check `dist_rel` vs `threshold` when fingers close
- Reduce `hold_frames` if timing out
- Increase `threshold_rel` for more forgiving detection

### Pointing Unstable:

- Increase `min_extension_ratio` for stricter extension
- Reduce `max_speed` if detecting during movement
- Check `extra_fingers_count` tolerance

### Swipe Missed:

- Reduce `velocity_threshold` for slower swipes
- Reduce `min_history` for faster response
- Check `direction` matches intended swipe

---

## Configuration Files

All parameters configurable in `config.json`:

```json
{
  "gesture_thresholds": {
    "zoom": { ... },
    "pinch": { ... },
    "pointing": { ... },
    "swipe": { ... },
    "open_hand": { ... },
    "thumbs": { ... }
  }
}
```

See `config_documentation.md` for parameter descriptions.

---

## Implementation Notes

### Relative Measurements Confirmed:

✓ All spatial measurements use normalized (0-1) coordinates  
✓ Velocity normalized by hand diagonal (not affected by distance)  
✓ Finger extension uses ratios (not absolute distances)  
✓ All thresholds configured relative to hand size  
✓ No hardcoded pixel values anywhere

### Metadata Consistency:

✓ All detectors return metadata on every call  
✓ Non-detection cases include `reason` field  
✓ Threshold values included for comparison  
✓ Real-time values always populated (no zeros)

---

## Quick Start

1. Run HANDS app: `python hands_app.py`
2. Press `Z` to see zoom parameters
3. Perform zoom gesture and watch values change
4. Press `X`, `I`, `S`, `O`, or `T` for other gestures
5. Adjust config.json based on observed values
6. Reload config (auto-detects changes every 30 frames)

---

_Last Updated: Current session_  
_System: HANDS v2.0 with comprehensive metadata_
