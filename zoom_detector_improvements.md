# ZoomDetector Improvements - Technical Documentation

## Overview

The ZoomDetector class has been significantly enhanced to address detection fragility and false positives. The new system uses **inertia-based confidence tracking** and **velocity-based filtering** to provide stable, intentional zoom detection.

---

## Problem Statement

### Original Issues:

1. **Flickering Detection**: Zoom would rapidly switch on/off as hand position fluctuated
2. **Drift False Positives**: Slow, unintentional hand movement triggered zoom
3. **Sudden Jump False Positives**: Quick hand repositioning caused brief false detections
4. **Poor Tunability**: No visibility into internal state, hard to adjust parameters

---

## Solution Architecture

### 1. Inertia System (Confidence Tracking)

**Concept**: Instead of immediately reporting zoom when fingers meet geometric criteria, build up confidence gradually.

**Implementation**:

```python
self.inertia = 0.0  # Ranges 0.0 (no zoom) to 1.0 (full zoom)
```

**Behavior**:

- **When conditions met**: Inertia increases by `inertia_increase` (default: 0.15) per frame
- **When conditions NOT met**: Inertia decreases by `inertia_decrease` (default: 0.3) per frame
- **Reporting threshold**: Only report zoom active when `inertia >= inertia_threshold` (default: 0.4)

**Benefits**:

- **Smooth transitions**: Prevents flickering during borderline cases
- **Hysteresis**: Different rates for building/losing confidence creates stability
- **Tunable response**: Adjust increase/decrease/threshold to balance responsiveness vs stability

**Config Parameters**:

```json
"inertia_increase": [0.15, "How fast confidence builds when zoom conditions met (per frame)"],
"inertia_decrease": [0.3, "How fast confidence fades when conditions not met (per frame)"],
"inertia_threshold": [0.4, "Minimum inertia level required to report zoom as detected"]
```

---

### 2. Velocity-Based Detection

**Concept**: Intentional zoom gestures have characteristic velocity - not too slow (drift) and not too fast (sudden jumps).

**Implementation**:

```python
self.velocity_history = deque(maxlen=5)  # Store last 5 velocity samples
avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
```

**Velocity Calculation**:

```python
relative_change = abs(current_spread - last_spread) / last_spread
```

**Checks Performed**:

1. **Minimum Velocity**: `avg_velocity >= min_velocity` (default: 0.05)
   - Filters out slow drift and stationary hands
2. **Maximum Velocity**: `avg_velocity <= max_velocity` (default: 2.0)
   - Filters out sudden jumps from hand repositioning
3. **Velocity Consistency**: `coefficient_of_variation <= velocity_consistency_threshold`
   - Ensures smooth, consistent motion
   - `CV = std_dev / mean` - lower values = more consistent
   - Default threshold: 0.7

**Benefits**:

- **Drift Immunity**: Slow, unintentional movement ignored
- **Jump Immunity**: Sudden position changes rejected
- **Smooth Motion Preference**: Erratic motion patterns filtered out

**Config Parameters**:

```json
"min_velocity": [0.05, "Minimum spread change velocity to count as intentional zoom"],
"max_velocity": [2.0, "Maximum velocity threshold - exceeding indicates hand repositioning"],
"velocity_consistency_threshold": [0.7, "Max coefficient of variation for velocity (lower = more consistent required)"]
```

---

### 3. Optional Finger Extension Check

**Change**: Made finger extension requirement configurable.

**Previous Behavior**: Always required index and middle fingers to be extended
**New Behavior**: Controlled by `require_fingers_extended` parameter

**Config Parameter**:

```json
"require_fingers_extended": [false, "Require index and middle fingers extended for zoom detection"]
```

**Rationale**: Some users prefer pinch-style zoom without strict finger extension requirements.

---

## Detection Logic Flow

```
1. Check basic geometry (finger_gap < gap_threshold)
   ├─ NO → Decrease inertia → Return None
   └─ YES → Continue

2. [Optional] Check finger extension (if require_fingers_extended=true)
   ├─ NO → Decrease inertia → Return None
   └─ YES → Continue

3. Calculate spread (thumb to paired fingers distance)

4. Check velocity conditions:
   a) Calculate relative change from last frame
   b) Update velocity history (rolling 5-frame window)
   c) Check: min_velocity ≤ avg_velocity ≤ max_velocity
   d) Check: velocity consistency (CV ≤ threshold)
   ├─ FAIL → Decrease inertia → Return None
   └─ PASS → Continue

5. Check scale change (|relative_change| > scale_threshold)
   ├─ NO → Decrease inertia → Return None
   └─ YES → Increase inertia → Continue

6. Check inertia threshold (inertia ≥ inertia_threshold)
   ├─ NO → Return None (building confidence...)
   └─ YES → Return zoom_in/zoom_out direction
```

---

## Metadata Exposure

All internal state exposed in detection result for tuning and debugging:

```python
return {
    'gesture': 'zoom_in' or 'zoom_out',
    'metadata': {
        'finger_gap': 0.05,              # Index-middle distance
        'spread': 0.15,                  # Thumb-fingers distance
        'relative_change': 0.08,         # Spread change percentage
        'inertia': 0.67,                 # Current confidence (0-1)
        'avg_velocity': 0.12,            # Rolling average velocity
        'velocity_consistency': 0.45     # Coefficient of variation
    }
}
```

**Visual Feedback**: These values displayed on-screen when zoom active (see `visual_feedback.py`)

---

## Configuration System Upgrade

### New Format

All config parameters now use `[value, description]` format:

```json
{
  "gesture_thresholds": {
    "zoom": {
      "gap_threshold": [
        0.06,
        "Maximum distance between index and middle fingers"
      ],
      "scale_threshold": [0.1, "Minimum spread change to register zoom"],
      "inertia_increase": [
        0.15,
        "How fast confidence builds when zoom conditions met"
      ]
    }
  }
}
```

### Backward Compatibility

Old format still works - `config_manager.py` automatically extracts values:

```python
# Both formats supported:
"param": 0.5           # Old format
"param": [0.5, "..."]  # New format
```

### GUI Enhancements

- **Tooltip system**: Hover over ℹ️ icon to see parameter descriptions
- **Context-aware editing**: Understand what each parameter controls

---

## Tuning Guide

### For More Responsive Zoom:

- **Increase** `inertia_increase` (faster confidence buildup)
- **Decrease** `inertia_threshold` (activate at lower confidence)
- **Decrease** `min_velocity` (accept slower motion)

### For More Stable Zoom:

- **Decrease** `inertia_increase` (slower confidence buildup)
- **Increase** `inertia_threshold` (require higher confidence)
- **Increase** `min_velocity` (ignore slow drift)
- **Decrease** `velocity_consistency_threshold` (require smoother motion)

### For Less Strict Detection:

- Set `require_fingers_extended: false`
- **Increase** `gap_threshold` (allow wider finger spacing)
- **Increase** `max_velocity` (allow faster motion)

---

## Testing Results

✓ Config manager loads new format correctly  
✓ ZoomDetector instantiates with all new parameters  
✓ Visual feedback system displays metadata  
✓ Backward compatibility maintained  
✓ All Python modules import without errors

---

## Files Modified

1. **gesture_detectors.py**: Complete ZoomDetector overhaul
2. **config.json**: Converted to [value, description] format
3. **config_manager.py**: Added `get_with_description()` method
4. **config_gui.py**: Added tooltip system for descriptions
5. **visual_feedback.py**: Added zoom metadata display panel

---

## Key Parameters Summary

| Parameter                        | Default | Purpose                       |
| -------------------------------- | ------- | ----------------------------- |
| `inertia_increase`               | 0.15    | Confidence buildup rate       |
| `inertia_decrease`               | 0.3     | Confidence decay rate         |
| `inertia_threshold`              | 0.4     | Activation threshold          |
| `min_velocity`                   | 0.05    | Drift filter                  |
| `max_velocity`                   | 2.0     | Jump filter                   |
| `velocity_consistency_threshold` | 0.7     | Smoothness requirement        |
| `require_fingers_extended`       | false   | Finger extension check toggle |

---

_Last Updated: Current session_  
_Author: GitHub Copilot_
