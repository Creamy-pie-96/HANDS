# HANDS Gesture Debug System - Complete Guide

## Overview

HANDS(Hand Assisted Navigation and Device-control System) is a AI driven hand gesture based navigation and device control system
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

### General Keys

These keys control the application itself and general displays.

| Key | Action                          |
| --- | ------------------------------- |
| `Q` | Quit application                |
| `P` | Pause/Resume gesture control    |
| `D` | Toggle debug info (terminal)    |
| `F` | Toggle FPS display on-screen    |
| `H` | Show this help (prints controls) |


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

**Tuning Guide:**

- `finger_count < min_fingers` → not enough fingers
- `is_pinching = True` → excluded as pinch gesture
- Check `fingers_extended` for individual finger states

---

### 6. Thumbs Detector

**Displayed Parameters:**

- `Vel`: Velocity vector (vx, vy)

---

## How to tune

---

## Quick Start

1. Run HANDS app: `python hands_app.py`
2. Press `Z` to see zoom parameters
3. Perform zoom gesture and watch values change
4. Press `X`, `I`, `S`, `O`, or `T` for other gestures
5. Adjust config.json based on observed values
6. Reload config (auto-detects changes every 30 frames)

---

_Last Updated: 02-12-2025 03:19 am_  
_System: HANDS v1.0_
