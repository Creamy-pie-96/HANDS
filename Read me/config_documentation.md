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
- `ewma_alpha`: 0.3 — EWMA smoothing factor for velocity (0-1).
  - Increase: more responsive but noisier.
  - Decrease: smoother but more latency.
- `hold_frames`: 5 — Frames to wait before confirming static thumbs_up/thumbs_down.
  - Purpose: gives time to detect if user intends to move thumb (e.g., thumbs_up_moving_up).
  - Increase: more time to detect movement intent, reduces false positives for static gestures.
  - Decrease: faster static gesture confirmation.
- `confidence_ramp_up`: 0.3 — Confidence increase per frame when consistent movement detected.
- `confidence_decay`: 0.2 — Confidence decrease per frame when movement stops/changes.
- `confidence_threshold`: 0.6 — Minimum confidence to report moving thumbs gesture.

## System Control (`system_control`)

Controls how gestures map to system actions.

### `cursor` (defaults shown)

- `smoothing_factor`: 0.3 — EWMA alpha for cursor smoothing (0-1). Lower = smoother but more lag.
- `speed_multiplier`: 1.5 — Scales cursor displacement from hand motion.
- `bounds_padding`: 10 — Pixels to avoid at screen edges.
- `precision_damping`: 0.3 — Multiplier applied in precision cursor mode.
- `screen_bounds_padding`: 10 — Distance from screen edges where cursor movement is clamped (pixels).
  - Increase: larger safety zone preventing cursor from reaching edges.
  - Decrease: cursor can move closer to screen boundaries.
- `fallback_screen_width`: 1920 — Screen width used when detection fails (pixels).
- `fallback_screen_height`: 1080 — Screen height used when detection fails (pixels).
  - Set these to match your typical screen resolution if automatic detection is unreliable.

### `click`

- `double_click_timeout`: 0.5 — Seconds allowed between pinches to register a double click.
- `drag_hold_duration`: 1.0 — Seconds pinch must be held to start drag.

### `scroll` and `zoom`

- `scroll.sensitivity`: 30 — Scroll amount multiplier.
- `scroll.speed_neutral`: 1.0 — Neutral gesture velocity where no speed modulation is applied.
- `scroll.speed_factor`: 0.2 — Max influence of velocity on scroll rate (0.2 = ±20% adjustment).
  - The effective sensitivity is modulated by gesture velocity using:
    $S_{\text{eff}} = \text{sensitivity} \times [1.0 + \text{speed\_factor} \times (V_{\text{norm}} - \text{speed\_neutral})]$
  - Faster gestures → higher effective sensitivity → shorter delay between actions.
  - Slower gestures → lower effective sensitivity → longer delay.
- `zoom.sensitivity`: 5 — Zoom multiplier (controls delay between zoom keypresses).
- `zoom.speed_neutral`: 1.0 — Neutral gesture velocity where no speed modulation is applied.
- `zoom.speed_factor`: 0.2 — Max influence of velocity on zoom rate (0.2 = ±20% adjustment).
- `zoom.use_system_zoom`: true — If enabled, performs system-level zoom shortcuts (Ctrl+Plus/Minus).

## Visual Feedback (`visual_feedback`)

- `enabled`: true — Master toggle for all overlays.
- `show_hand_skeleton`: true — Draw hand skeleton lines.
- `show_fingertips`: true — Draw fingertip markers.
- `show_cursor_preview`: true — Show where cursor will move.
- `show_gesture_name`: true — Display detected gesture name.
- `overlay_opacity`: 0.7 — Overlay transparency (0-1).
- `colors`: per-element BGR color arrays for left/right hands, cursor, active highlights, background.

### `visual_feedback.debug_panel` (Position/Layout of main debug overlay)

- `start_y_offset`: 8 — Initial Y offset from top of screen for debug panel placement (pixels).
  - Increase: panel appears lower on screen.
  - Decrease: panel starts higher.
- `scan_step_horizontal`: -30 — Horizontal step size when scanning for unoccupied space (pixels).
  - Negative values scan right-to-left; larger magnitude = faster scanning but coarser.
- `scan_step_vertical`: 20 — Vertical step size when scanning for placement (pixels).
  - Increase: faster vertical scanning but may skip tight gaps.

### `visual_feedback.gesture_panel` (Per-gesture metadata overlays)

- `max_height`: 200 — Maximum panel height before content is clipped (pixels).
- `panel_y`: 10 — Starting Y position from top of screen (pixels).
- `panel_left_x`: 10 — X position from left screen edge (pixels).
- `panel_width`: 300 — Width of gesture info panel (pixels).
- `overlay_alpha`: 0.7 — Background transparency (0-1). Higher = more opaque/visible.
- `frame_blend_alpha`: 0.3 — Transparency when blending panel onto video frame (0-1).
- `title_y_offset`: 20 — Y offset for title text from panel top (pixels).
- `title_x`: 20 — X position of title text within panel (pixels).
- `separator_y_offset`: 25 — Y position of title separator line from panel top (pixels).
- `separator_start_x`: 10 — Starting X position of separator line (pixels).
- `separator_end_x`: 290 — Ending X position of separator line (pixels).
- `indicator_start_y`: 40 — Y position where gesture indicators begin (pixels from panel top).
- `indicator_x`: 25 — X position of gesture indicator symbols (pixels).
- `name_x`: 45 — X position of gesture name text (pixels).
- `hint_x`: 180 — X position for gesture hint/help text (pixels).
- `line_spacing`: 20 — Vertical spacing between gesture lines (pixels).
  - Increase: more readable but takes more screen space.
  - Decrease: compact layout.
- `param_indent_x`: 45 — X indent for parameter detail lines (pixels).
- `param_line_spacing`: 15 — Vertical spacing between parameter lines (pixels).
- `spacing_with_hint`: 25 — Extra Y spacing when hint text is present (pixels).
- `spacing_no_hint`: 25 — Y spacing when no hint is shown (pixels).
- `no_gesture_x`: 40 — X position for "No gestures detected" message (pixels).

### `visual_feedback.cursor_preview` (Cursor visualization)

- `trail_fade_time`: 0.5 — Duration for cursor trail to fade out (seconds).
  - Increase: trail persists longer (shows movement path).
  - Decrease: trail fades quickly.
- `circle_radius`: 15 — Radius of cursor preview circle (pixels).
  - Increase: larger, more visible cursor indicator.
  - Decrease: smaller, subtler cursor.
- `crosshair_length`: 20 — Length of crosshair arms from center (pixels).
- `crosshair_gap`: 8 — Gap between center and start of crosshair (pixels).

### `visual_feedback.animation`

- `pulse_frequency`: 2.0 — Frequency of pulsing effect for active gestures (Hz).
  - Increase: faster pulsing (more attention-grabbing).
  - Decrease: slower, gentler pulse effect.

## Camera / Performance / Display

- `camera.index`: 0 — Default camera device.
- `camera.width` / `camera.height`: 640 / 480 — Capture resolution.
- `camera.fps`: 60 — Requested FPS (actual may vary by camera/system).

- `performance.use_gpu`: true — Use GPU acceleration for hand detection.
  - When enabled, HANDS uses the MediaPipe Tasks API with GPU delegate for faster inference.
  - Provides significant FPS improvement (typically 1.5x-2x faster than CPU).
  - Automatically falls back to CPU if GPU is unavailable or initialization fails.
  - Set to `false` to force CPU-only processing.
  - **Requirements**: Compatible GPU with OpenGL ES 3.1+ or Vulkan support.
- `performance.show_fps`: true — Show FPS counter on-screen (displays [GPU] or [CPU] indicator).
- `performance.show_debug_info`: false — Print verbose detection info to terminal.
- `performance.max_hands`: 2 — Max hands to detect.
- `performance.min_detection_confidence`: 0.7 — MediaPipe detection threshold.
- `performance.min_tracking_confidence`: 0.3 — MediaPipe tracking threshold.
- `performance.gesture_history_maxlen`: 16 — Maximum frames stored in gesture history buffer per hand.
  - Increase: more memory used, longer history for trend analysis.
  - Decrease: less memory, shorter history window.
- `performance.bimanual_history_maxlen`: 10 — Maximum frames stored in two-hand gesture history buffer.

  - Increase: smoother two-hand gesture detection, uses more memory.
  - Decrease: faster response, less stable detection.

- `display.window_width` / `window_height`: 1280 / 720 — Default window size.
- `display.flip_horizontal`: true — Mirror the video output.
- `display.fps_update_interval`: 30 — Frames between FPS recalculation and config reload checks.

## Using the Config Editor

- Run `python3 config_gui.py` to open the interactive editor. The editor shows each field (value + description). Hover the ℹ️ icon to read the stored description, edit values, and press "Save & Apply" to write back to `config.json`.
- The HANDS app watches `config.json` and auto-reloads settings (every `display.fps_update_interval` frames by default). If a change is not applied, restart the HANDS app.
