# HANDS Gesture Debug System - Complete Guide

## Overview

## HANDS(Hand Assisted Navigation and Device-control System) is a AI driven hand gesture based navigation and device control system

---

## Keyboard Controls

### Togle keys

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

| Key | Action                           |
| --- | -------------------------------- |
| `Q` | Quit application                 |
| `P` | Pause/Resume gesture control     |
| `D` | Toggle debug info (terminal)     |
| `F` | Toggle FPS display on-screen     |
| `H` | Show this help (prints controls) |

---

## Gesture Controls

| Gesture                             | Description                           |
| ----------------------------------- | ------------------------------------- |
| Pointing (👆)                       | Move cursor (index finger)            |
| Pinch (🤏)                          | Click / Drag (thumb + index)          |
| Zoom (🤌)                           | System zoom in/out (3 fingers)        |
| Swipe (👋)                          | Scroll / Switch workspace (4 fingers) |
| Open hand (✋)                      | Reserved (5 fingers)                  |
| Thumbs Up (👍)                      | Thumbs up — Confirm / Accept          |
| Thumbs Down (👎)                    | Thumbs down — Reject / Decline        |
| Thumbs Up (👍) — Moving Up          | Thumbs up + upward velocity           |
| Thumbs Up (👍) — Moving Down        | Thumbs up + downward velocity         |
| Thumbs Down (👎)— Moving Up         | Thumbs down + upward velocity         |
| Thumbs Up (👍⬆️)                    | Thumbs up + upward velocity           |
| Thumbs Up (👍⬇️)                    | Thumbs up + downward velocity         |
| Thumbs Down (👎⬆️)                  | Thumbs down + upward velocity         |
| Thumbs Down (👎⬇️)                  | Thumbs down + downward velocity       |
| Two-hand — Left still + Right move  | Pan / Scroll                          |
| Two-hand — Left still + Right point | Precision cursor                      |

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

---

### 2. Pinch Detector

**Displayed Parameters:**

- `Dist`: Current thumb-index distance
- `Thrs`: Configured threshold
- `Hold`: Current hold count / required hold frames
- `CDwn`: Cooldown time remaining (seconds)

---

### 3. Pointing Detector

**Displayed Parameters:**

- `Dist`: Index finger distance from palm
- `MinD`: Minimum required distance
- `Spd`: Current hand speed
- `MaxS`: Maximum allowed speed
- `Xtra`: Extra fingers extended / max allowed
- `Rsn`: Reason for non-detection

---

### 4. Swipe Detector

**Displayed Parameters:**

- `Dir`: Swipe direction (up/down/left/right)
- `Spd`: Current velocity
- `Thrs`: Velocity threshold
- `Hist`: History frames collected / min required
- `CDwn`: Cooldown remaining (seconds)
- `Rsn`: Reason for non-detection

---

### 5. Open Hand Detector

**Displayed Parameters:**

- `Cnt`: Fingers extended / minimum required
- `TIDist`: Thumb-index distance
- `Pinch`: Is hand in pinch position?
- `Rsn`: Reason for non-detection

---

### 6. Thumbs Detector

**Displayed Parameters:**

- `Vel`: Velocity vector (vx, vy)

---

## How to tune

### Open the app

- Run it and this will automatically Activate your Python environment and run the app:

```bash
./start_hands.sh
```

- Ensure the camera window `HANDS Control` is focused. Use `H` to print keyboard help, and use `Z/X/I/S/O/T` to toggle per-gesture debug overlays. `D` toggles verbose terminal debug, `F` toggles the on-screen FPS counter. The app auto-reloads `config.json` when it changes (default every 30 frames).

### Tuning workflow (in-app)

- Enable the overlay for the gesture you want to tune (e.g., press `Z` for Zoom). The overlay will show the live metrics the detector uses (and a short "reason" when detection fails).
- Perform the gesture slowly and then with deliberate variations while watching the overlay values and the terminal (if `D` is on). Note which metadata fields are close to thresholds or oscillating.
- Edit `config.json` using the included GUI (`./run_config.sh`) or a text editor. Save the file and the app will auto-reload the config (or restart the app if necessary).
- Iterate: tweak one parameter at a time, test with a few repetitions, then revert if behaviour degrades.

### What fields you can tune (and how increasing/decreasing affects behavior)

Below are the main tunable fields found in `config.json` and what changing them does. If you use the `Config Editor` (`config_gui.py`), hover the small ℹ️ icons to see the stored descriptions.

**gesture_thresholds.pinch**

- `threshold_rel`: Maximum normalized distance between thumb and index to count as a pinch.
  - Increase → easier to trigger (fingers can be further apart).
  - Decrease → requires fingers to be closer (more strict).
- `hold_frames`: Number of consecutive frames pinch must be held to trigger.
  - Increase → requires steadier hold before a click (reduces false positives).
  - Decrease → faster response but more susceptible to flicker.
- `cooldown_seconds`: Minimum seconds between pinch detections.
  - Increase → reduces accidental double-clicks (slower re-triggering).
  - Decrease → allows more frequent pinch events.

**gesture_thresholds.pointing**

- `min_extension_ratio`: How far index tip must extend from the palm to be considered pointing.
  - Increase → requires a more pronounced pointing pose.
  - Decrease → accepts smaller extensions as pointing.
- `max_speed`: Maximum allowed hand velocity for a stable pointing detection.
  - Increase → allows pointing while moving faster (less stable cursor).
  - Decrease → stricter: pointing is ignored if hand is moving quickly.
- `max_extra_fingers`: How many extra fingers may be extended while still counting as pointing.
  - Increase → more permissive (tolerant of extra fingers).
  - Decrease → stricter (only index or index+one allowed).

**gesture_thresholds.swipe**

- `velocity_threshold`: Minimum normalized velocity to report a swipe.
  - Increase → requires faster motion to trigger (reduces false swipes).
  - Decrease → easier to trigger with slower swipes.
- `cooldown_seconds`: Minimum time between swipe detections.
  - Increase → fewer repeated swipes.
  - Decrease → allows more frequent swipes.
- `history_size` / `min_history`: Number of frames used to compute velocity and minimum frames required.
  - Increase `history_size` → smoother velocity estimate (slower to react).
  - Decrease `history_size` → more responsive but noisier velocity.

**gesture_thresholds.finger_extension**

- `open_ratio` / `close_ratio`: Ratios used to decide whether a finger is extended (with hysteresis).
  - Increase `open_ratio` → requires larger tip/pip distance to count as extended.
  - Decrease `open_ratio` → easier to consider a finger extended.
  - `close_ratio` should be slightly lower than `open_ratio` to avoid flicker.
- `motion_speed_threshold` and `motion_sigmoid_k`: Used to relax thresholds when the whole hand is moving quickly.
  - Increase `motion_speed_threshold` → extension logic ignores motion until a higher speed.
  - Increase `motion_sigmoid_k` → sharper transition between relaxed and strict modes.

**gesture_thresholds.zoom**

- `scale_threshold`: Minimum relative spread change required to trigger zoom.
  - Increase → requires a larger zoom motion to register (less sensitive).
  - Decrease → more sensitive to subtle spread changes (can increase false positives).
- `finger_gap_threshold`: Maximum allowed gap between index & middle fingers (they must be together).
  - Increase → more permissive about the pair staying together.
  - Decrease → stricter pairing (avoids accidental zoom when fingers drift).
- `history_size`: Number of frames for trend detection (smoothing).
  - Increase → smoother but slower reaction.
  - Decrease → more responsive but noisier.
- `inertia_increase`, `inertia_decrease`, `inertia_threshold`: Controls confidence buildup and decay for zoom.
  - Increase `inertia_increase` → zoom reaches reported-detected state faster.
  - Increase `inertia_decrease` → zoom confidence decays faster when motion stops.
  - Increase `inertia_threshold` → require higher confidence to report zoom.
- `min_velocity` / `max_velocity`: Velocity bounds used to filter drift and spikes.
  - Increase `min_velocity` → ignores very slow spread changes.
  - Decrease `min_velocity` → allows slower intentional zooms.
  - Decrease `max_velocity` → filters out very fast (likely noisy) jumps.
- `velocity_consistency_threshold`: How consistent the velocity must be to accept zoom.
  - Increase → requires smoother, consistent motion.
  - Decrease → allows more variable motion.
  - `require_fingers_extended`: If true, require three fingers clearly extended for zoom.

**gesture_thresholds.open_hand**

- `min_fingers`: Minimum number of extended fingers to count as an open hand.
  - Increase → requires more fingers (stricter open-hand detection).
  - Decrease → easier to satisfy (may collide with thumbs/pinch states).
- `pinch_exclusion_distance`: If thumb-index is closer than this, the open-hand detection is suppressed.
  - Increase → more likely to exclude open-hand when any pinch-like contact is present.
  - Decrease → less aggressive exclusion.

**gesture_thresholds.thumbs**

- `velocity_threshold`: Minimum thumb velocity (normalized) required to treat an up/down motion as an active thumbs movement.
  - Increase → requires faster motion to detect moving variants.
  - Decrease → catches gentler vertical motions but may increase false positives.

**system_control.cursor**

- `smoothing_factor` (EWMA alpha): Controls cursor smoothing.
  - Increase → less smoothing (more immediate movement, possibly jittery).
  - Decrease → more smoothing (laggy but stable cursor).
- `speed_multiplier`: Scale of cursor movement.
  - Increase → faster cursor movement for the same hand displacement.
  - Decrease → slower, finer-grained movement.
- `precision_damping`: Applied when in precision mode (two-hand precision cursor).
  - Decrease → finer control (multiply movement by a smaller factor).
- `screen_bounds_padding`: Distance (in pixels) from screen edges where cursor movement stops.
  - Increase → larger safety zone (cursor stops further from edge).
  - Decrease → cursor can get closer to edges.
- `fallback_screen_width` / `fallback_screen_height`: Resolution used when actual screen dimensions cannot be detected.
  - Set to your typical screen resolution (e.g., 1920x1080) if detection fails.

**system_control.click**

- `double_click_timeout`: Max time between pinches for a double click.
  - Increase → allows slower double-clicks.
  - Decrease → requires faster sequential pinches.
- `drag_hold_duration`: Duration required to start a drag after a pinch.
  - Increase → longer hold needed to start drag.
  - Decrease → drag starts sooner.

**system_control.scroll / zoom**

- `sensitivity` values scale how aggressively scroll/zoom commands are sent to the OS or app.
  - Increase → larger scroll/zoom per detected gesture.
  - Decrease → gentler scroll/zoom.

**visual_feedback**

- `overlay_opacity`: Transparency of the on-screen overlays.
  - Increase → overlays more visible (may obscure camera feed).
  - Decrease → overlays fainter.
- `show_hand_skeleton`, `show_fingertips`, `show_cursor_preview`, `show_gesture_name`: Toggle debug visuals on/off.

**visual_feedback.debug_panel** (Position/Layout of the main debug overlay)

- `start_y_offset`: Initial Y position offset from screen top for debug panel placement (pixels).
  - Increase → panel starts lower on the screen.
  - Decrease → panel starts higher.
- `scan_step_horizontal` / `scan_step_vertical`: Step size when scanning for an unoccupied screen region for the panel.
  - Increase → faster scanning but may miss tight spaces.
  - Decrease → finer scanning, finds smaller gaps.

**visual_feedback.gesture_panel** (Position/Styling of per-gesture metadata overlays)

- `max_height`: Maximum panel height before scrolling/clipping (pixels).
- `panel_y`: Starting Y position for gesture panel (pixels from top).
- `panel_left_x`: X position from screen left edge (pixels).
- `panel_width`: Width of the gesture info panel (pixels).
- `overlay_alpha`: Transparency of overlay background (0-1).
  - Increase → more opaque (easier to read, more intrusive).
  - Decrease → more transparent (subtle but harder to read).
- `frame_blend_alpha`: Transparency when blending panel onto the video frame (0-1).
- `title_y_offset`: Y offset from panel top for the title text (pixels).
- `title_x`: X position of the title text within panel (pixels).
- `separator_y_offset` / `separator_start_x` / `separator_end_x`: Position and length of the line separating title from content.
- `indicator_start_y`: Y position where gesture indicators begin (pixels from panel top).
- `indicator_x`: X position of the gesture indicator symbols (pixels).
- `name_x`: X position of gesture name text (pixels).
- `hint_x`: X position of gesture hint/help text (pixels).
- `line_spacing`: Vertical spacing between gesture lines (pixels).
  - Increase → more space between gestures (easier to read but takes more screen space).
  - Decrease → tighter layout.
- `param_indent_x`: X indent for parameter detail lines (pixels).
- `param_line_spacing`: Vertical spacing between parameter detail lines (pixels).
- `spacing_with_hint`: Extra Y spacing when a hint line is present (pixels).
- `spacing_no_hint`: Y spacing when no hint is shown (pixels).
- `no_gesture_x`: X position for "No gestures detected" message (pixels).

**visual_feedback.cursor_preview** (Cursor visualization)

- `trail_fade_time`: How long cursor trail takes to fade out (seconds).
  - Increase → trail persists longer (more visible path).
  - Decrease → trail fades quickly (cleaner display).
- `circle_radius`: Radius of the circle drawn around cursor position (pixels).
  - Increase → larger cursor indicator.
  - Decrease → smaller, subtler cursor.
- `crosshair_length`: Length of crosshair arms from center (pixels).
- `crosshair_gap`: Gap between center and start of crosshair arms (pixels).

**visual_feedback.animation**

- `pulse_frequency`: Frequency of pulsing animation for active gestures (Hz).
  - Increase → faster pulsing (more attention-grabbing).
  - Decrease → slower, gentler pulsing.

**camera / performance / display**

- `fps`: Requested camera FPS — higher FPS gives finer temporal resolution but may lower frame quality or increase CPU usage.
- `min_detection_confidence` / `min_tracking_confidence`: MediaPipe thresholds; increasing them reduces false detections but may drop hands intermittently.
- `gesture_history_maxlen`: Maximum number of frames stored in gesture history buffer (per hand).
  - Increase → more memory used but allows longer-term gesture trend analysis.
  - Decrease → less memory, shorter history for gesture detection.
- `bimanual_history_maxlen`: Maximum number of frames stored in two-hand gesture history buffer.
  - Increase → better smoothing for two-hand gestures but uses more memory.
  - Decrease → faster response but less stable two-hand detection.

---

## Quick Start

1. Go to app directory. Run the HANDS app :

```bash
./start_hands.sh
```

2. Goto to app directory. Run the Config Editor GUI to edit parameters interactively (recommended):

```bash
./run_config.sh
```

3. Open the camera window and enable the per-gesture debug overlay(s) you want to inspect:

- Press `Z` to show Zoom metadata (Gap, Spr, Chg, Inr, Vel, VCon, Rsn) and tune the zoom fields.
- Press `X` to show Pinch metadata (Dist, Thrs, Hold, CDwn).
- Press `I` to show Pointing metadata (Dist, MinD, Spd, Xtra).
- Press `S` to show Swipe metadata (Dir, Spd, Thrs, Hist).
- Press `O` to show Open hand metadata (Cnt, TIDist, Pinch).
- Press `T` to show Thumbs metadata (Vel and thumbs state).

4. While watching the overlays (and terminal if `D` is enabled), tweak the relevant fields in the Config Editor or `config.json`, then save.

5. The HANDS app auto-reloads the `config.json` (by default every 30 frames). If you don't see changes applied immediately, save and restart the HANDS app.

## **This way you can fine-tune the app live while using it to match your preference and your own unique body language and speed**

**Note: If you are new and have not installed necessery packages run the command:**

```bash
    install.sh
```

_Last Updated: 02-12-2025 18:51_  
_System: HANDS v1.0_
