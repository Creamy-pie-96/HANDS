# HANDS Configuration Documentation

This document describes the current `config.json` fields for the HANDS system and explains what each parameter controls and how changing values affects behavior. Use the interactive `Config Editor` (`config_gui.py`) or edit `config.json` directly.

## Top-level structure

- `gesture_thresholds`: per-gesture tuning parameters (pinch, pointing, swipe, zoom, open_hand, thumbs, finger_extension).
- `system_control`: how detected gestures map to OS actions (cursor, click, scroll, zoom settings).
- `visual_feedback`: on-screen visualization toggles and styling.
- `camera`, `performance`, `display`: capture and runtime settings.

## Gesture Thresholds (`gesture_thresholds`)

These control detection sensitivity and behavior for each gesture group. Values below show the current defaults from `config.json` and notes on impact.

### `pinch`

- `threshold_rel`: 0.055 — Maximum normalized distance (tip-tip relative to hand size) for pinch detection.
  - Increase: easier to trigger (fingers can be further apart).
  - Decrease: requires closer contact.
- `hold_frames`: 3 — Frames the pinch must be continuously held before it is confirmed.
  - Increase: reduces accidental short pinches, slower response.
  - Decrease: faster clicks, may flicker.
- `cooldown_seconds`: 0.6 — Minimum time between pinch events.
  - Increase: prevents rapid re-triggering.
  - Decrease: allows faster repeated pinches.

### `pointing`

- `min_extension_ratio`: 0.12 — Minimum relative extension required for the index finger to count as pointing.
  - Increase: requires more pronounced pointing.
  - Decrease: allows subtler pointing.
- `max_speed`: 0.5 — Max hand velocity for stable pointing.
  - Increase: allows pointing while moving faster (less precise).
  - Decrease: stricter, ignores pointing during motion.
- `max_extra_fingers`: 1 — How many other fingers can be slightly extended while still counting as pointing.
  - Increase: more tolerant of extra fingers.
  - Decrease: stricter (only index or index+one).

### `swipe`

- `velocity_threshold`: 0.45 — Minimum normalized hand velocity to report a swipe.
  - Increase: requires faster swipes, fewer false positives.
  - Decrease: allows slower swipes.
- `cooldown_seconds`: 0.4 — Minimum seconds between swipe detections.
- `history_size`: 8 — Frames used to compute velocity for swipe.
- `min_history`: 3 — Minimum frames required before attempting swipe detection.
  - Longer history → smoother velocity estimate but slower detection.

### `finger_extension`

- `open_ratio`: 1.2 — Tip/PIP ratio threshold for a finger to be considered extended.
- `close_ratio`: 1.1 — Lower hysteresis threshold to consider finger closed.
- `motion_speed_threshold`: 0.2 — Hand speed above which extension logic is relaxed.
- `motion_sigmoid_k`: 20.0 — Sharpness of the motion-based relaxation curve.
  - Use these to make extension detection robust during hand motion (increase `motion_speed_threshold` to require faster motion before relaxing).

### `zoom`

- `scale_threshold`: 0.1 — Minimum relative spread change required to consider zooming.
  - Increase: less sensitive to small spread changes.
  - Decrease: more sensitive, may catch drift.
- `finger_gap_threshold`: 0.1 — Max gap between index & middle tips for them to be considered a pair.
- `history_size`: 5 — Frames used to analyze spread trend.
- `inertia_increase`: 0.4 — How quickly zoom confidence builds per valid frame.
- `inertia_decrease`: 0.1 — How quickly confidence decays when invalid.
- `inertia_threshold`: 0.4 — Confidence threshold to report zoom detection.
- `min_velocity`: 0.05 — Minimum spread velocity to be considered intentional.
- `max_velocity`: 2.0 — Upper velocity bound to ignore spikes/noise.
- `velocity_consistency_threshold`: 0.7 — How smooth/consistent velocity must be.
- `require_fingers_extended`: false — If true, require all 3 fingers clearly extended for zoom.

### `open_hand`

- `min_fingers`: 4 — Minimum fingers extended to register an open hand.
- `pinch_exclusion_distance`: 0.08 — Thumb-index proximity below this suppresses open-hand detection.

### `thumbs`

- `velocity_threshold`: 0.2 — Minimum thumb vertical velocity to detect thumbs moving variants (up/down moving gestures).
  - Increase: require faster thumb motion to trigger moving variants.
  - Decrease: detect gentler thumb motions (can increase false positives).

## System Control (`system_control`)

Controls how gestures map to system actions.

### `cursor` (defaults shown)

- `smoothing_factor`: 0.3 — EWMA alpha for cursor smoothing (0-1). Lower = smoother but more lag.
- `speed_multiplier`: 1.5 — Scales cursor displacement from hand motion.
- `bounds_padding`: 10 — Pixels to avoid at screen edges.
- `precision_damping`: 0.3 — Multiplier applied in precision cursor mode.

### `click`

- `double_click_timeout`: 0.5 — Seconds allowed between pinches to register a double click.
- `drag_hold_duration`: 1.0 — Seconds pinch must be held to start drag.

### `scroll` and `zoom`

- `scroll.sensitivity`: 30 — Scroll amount multiplier.
- `zoom.sensitivity`: 5 — Zoom multiplier.
- `zoom.use_system_zoom`: true — If enabled, performs system-level zoom shortcuts.

## Visual Feedback (`visual_feedback`)

- `enabled`: true — Master toggle for all overlays.
- `show_hand_skeleton`: true — Draw hand skeleton lines.
- `show_fingertips`: true — Draw fingertip markers.
- `show_cursor_preview`: true — Show where cursor will move.
- `show_gesture_name`: true — Display detected gesture name.
- `overlay_opacity`: 0.7 — Overlay transparency (0-1).
- `colors`: per-element BGR color arrays for left/right hands, cursor, active highlights, background.

## Camera / Performance / Display

- `camera.index`: 0 — Default camera device.
- `camera.width` / `camera.height`: 640 / 480 — Capture resolution.
- `camera.fps`: 60 — Requested FPS (actual may vary by camera/system).

- `performance.show_fps`: true — Show FPS counter on-screen.
- `performance.show_debug_info`: false — Print verbose detection info to terminal.
- `performance.max_hands`: 2 — Max hands to detect.
- `performance.min_detection_confidence`: 0.7 — MediaPipe detection threshold.
- `performance.min_tracking_confidence`: 0.3 — MediaPipe tracking threshold.

- `display.window_width` / `window_height`: 1280 / 720 — Default window size.
- `display.flip_horizontal`: true — Mirror the video output.
- `display.fps_update_interval`: 30 — Frames between FPS recalculation and config reload checks.

## Using the Config Editor

- Run `python3 config_gui.py` to open the interactive editor. The editor shows each field (value + description). Hover the ℹ️ icon to read the stored description, edit values, and press "Save & Apply" to write back to `config.json`.
- The HANDS app watches `config.json` and auto-reloads settings (every `display.fps_update_interval` frames by default). If a change is not applied, restart the HANDS app.
