# HANDS Configuration Documentation

This document explains the configuration fields available in `config.json` for the HANDS system. You can tune these values using the `config_gui.py` application or by editing the file directly.

## Gesture Thresholds (`gesture_thresholds`)

These settings control the sensitivity and detection logic for various gestures.

### `thumbs` (New)
Controls the detection of Thumbs Up/Down gestures and their movement.
- **`velocity_threshold`**: (Default: 0.2) The vertical velocity required to trigger "Moving Up" or "Moving Down" variants. Higher values require faster movement.
- **`description`**: Description of the gesture group.

### `pinch`
Controls the Thumb-Index pinch gesture (used for clicking/dragging).
- **`threshold_rel`**: (Default: 0.055) Maximum distance between thumb and index finger tips (relative to hand size) to register a pinch.
- **`hold_frames`**: (Default: 5) Number of frames the pinch must be held to be confirmed.
- **`cooldown_seconds`**: (Default: 0.6) Time before another pinch can be detected.

### `pointing`
Controls the single-finger pointing gesture (used for cursor movement).
- **`min_extension_ratio`**: (Default: 0.12) Minimum extension of the index finger relative to the palm.
- **`max_speed`**: (Default: 0.5) Maximum speed allowed for precise pointing.
- **`max_extra_fingers`**: (Default: 1) How many other fingers can be slightly extended before pointing is invalidated.

### `swipe`
Controls directional swipe gestures (used for scrolling/switching).
- **`velocity_threshold`**: (Default: 0.8) Minimum velocity to register a swipe.
- **`cooldown_seconds`**: (Default: 0.5) Time between swipes.
- **`history_size`**: (Default: 8) Number of frames used to calculate velocity.

### `zoom`
Controls the 3-finger pinch-to-zoom gesture.
- **`scale_threshold`**: (Default: 0.15) Minimum scale change to trigger zoom.
- **`finger_gap_threshold`**: (Default: 0.06) Maximum gap between index and middle fingers (they should be together).
- **`history_size`**: (Default: 5) Frames to track for smooth zooming.

### `open_hand`
Controls the open palm gesture.
- **`min_fingers`**: (Default: 4) Minimum number of extended fingers.
- **`pinch_exclusion_distance`**: (Default: 0.08) Minimum distance between thumb and index to ensure it's not a pinch.

### `finger_extension`
Global settings for determining if a finger is extended.
- **`open_ratio`**: (Default: 1.20) Ratio of tip-to-palm vs pip-to-palm distance to consider open.
- **`close_ratio`**: (Default: 1.10) Ratio to consider closed (hysteresis).
- **`motion_speed_threshold`**: (Default: 0.15) Speed threshold for dynamic adjustment.
- **`motion_sigmoid_k`**: (Default: 20.0) Sigmoid steepness for motion adjustment.

## System Control (`system_control`)

Settings for how gestures translate to system actions.

### `cursor`
- **`smoothing_factor`**: (Default: 0.3) Smoothing applied to cursor movement (0.0-1.0). Lower is smoother but more laggy.
- **`speed_multiplier`**: (Default: 1.5) Multiplier for cursor speed.
- **`bounds_padding`**: (Default: 10) Padding from screen edges.
- **`precision_damping`**: (Default: 0.3) Speed reduction factor in precision mode.

### `click`
- **`double_click_timeout`**: (Default: 0.5) Max time between clicks for a double-click.
- **`drag_hold_duration`**: (Default: 1.0) Time to hold pinch to initiate drag.

### `scroll`
- **`sensitivity`**: (Default: 30) Scroll speed multiplier.
- **`horizontal_enabled`**: (Default: true) Enable horizontal scrolling.
- **`vertical_enabled`**: (Default: true) Enable vertical scrolling.

### `zoom`
- **`sensitivity`**: (Default: 5) Zoom speed multiplier.
- **`use_system_zoom`**: (Default: true) Whether to use OS zoom shortcuts.

## Visual Feedback (`visual_feedback`)

Settings for the on-screen overlay.
- **`enabled`**: (Default: true) Master switch for visuals.
- **`show_hand_skeleton`**: (Default: true) Draw hand lines.
- **`show_fingertips`**: (Default: true) Draw dots on fingertips.
- **`show_cursor_preview`**: (Default: true) Show where the cursor would be.
- **`show_gesture_name`**: (Default: true) Display name of detected gesture.
- **`overlay_opacity`**: (Default: 0.7) Opacity of the overlay window.
- **`colors`**: RGB color definitions for various elements.

## Camera (`camera`)
- **`index`**: (Default: 0) Camera device ID.
- **`width`**: (Default: 640) Capture width.
- **`height`**: (Default: 480) Capture height.
- **`fps`**: (Default: 60) Target FPS.

## Performance (`performance`)
- **`show_fps`**: (Default: true) Display FPS counter.
- **`show_debug_info`**: (Default: false) Print debug info to terminal.
- **`max_hands`**: (Default: 2) Maximum number of hands to track.
- **`min_detection_confidence`**: (Default: 0.7) MediaPipe detection threshold.
- **`min_tracking_confidence`**: (Default: 0.3) MediaPipe tracking threshold.
