# HANDS Configuration Documentation

Complete reference for all `config.json` parameters. Each parameter is stored as `[value, description]` in the config file.

---

## Top-Level Structure

```json
{
  "gestures_enabled": { ... },      // Enable/disable individual gestures
  "gesture_thresholds": { ... },    // Detection sensitivity parameters
  "system_control": { ... },        // How gestures map to OS actions
  "bimanual_gestures": { ... },     // Two-hand gesture settings
  "visual_feedback": { ... },       // On-screen overlay settings
  "camera": { ... },                // Capture settings
  "performance": { ... },           // Runtime optimization
  "display": { ... }                // Window and indicator settings
}
```

---

## Gestures Enabled (`gestures_enabled`)

Controls which gestures trigger system actions. Disabled gestures are still detected and shown (with a red dot indicator) but don't perform actions.

| Parameter                 | Default | Description                       |
| ------------------------- | ------- | --------------------------------- |
| `pointing`                | true    | Cursor control                    |
| `pinch`                   | true    | Click/drag                        |
| `zoom_in`                 | true    | System zoom in                    |
| `zoom_out`                | true    | System zoom out                   |
| `swipe_left`              | true    | Workspace switch left             |
| `swipe_right`             | true    | Workspace switch right            |
| `swipe_up`                | true    | Scroll up                         |
| `swipe_down`              | true    | Scroll down                       |
| `thumbs_up`               | true    | Static thumbs up (reserved)       |
| `thumbs_down`             | true    | Static thumbs down (quit gesture) |
| `thumbs_up_moving_up`     | true    | Volume up                         |
| `thumbs_up_moving_down`   | true    | Volume down                       |
| `thumbs_down_moving_up`   | true    | Brightness up                     |
| `thumbs_down_moving_down` | true    | Brightness down                   |
| `open_hand`               | true    | Open hand gesture                 |
| `pan`                     | true    | Bimanual pan/scroll               |
| `precision_cursor`        | true    | Bimanual precision cursor         |

---

## Gesture Thresholds (`gesture_thresholds`)

### Pinch

| Parameter          | Default | Effect of ↑                      | Effect of ↓                  |
| ------------------ | ------- | -------------------------------- | ---------------------------- |
| `threshold_rel`    | 0.2     | Easier trigger (fingers further) | Stricter (fingers closer)    |
| `hold_frames`      | 3       | Slower response, less flicker    | Faster, more false positives |
| `cooldown_seconds` | 0.4     | Fewer repeated clicks            | More rapid clicks allowed    |

### Pointing

| Parameter             | Default | Effect of ↑                         | Effect of ↓                    |
| --------------------- | ------- | ----------------------------------- | ------------------------------ |
| `min_extension_ratio` | 1.01    | More pronounced pointing needed     | Subtle pointing accepted       |
| `max_speed`           | 1.0     | Pointing works during faster motion | Stricter (ignores fast motion) |
| `max_extra_fingers`   | 0       | More tolerant of extra fingers      | Stricter (only index)          |
| `ewma_alpha`          | 0.3     | More responsive, noisier            | Smoother, more lag             |

### Swipe

| Parameter              | Default | Effect of ↑                      | Effect of ↓            |
| ---------------------- | ------- | -------------------------------- | ---------------------- |
| `ewma_alpha`           | 0.3     | More responsive                  | Smoother velocity      |
| `velocity_threshold_x` | 0.25    | Faster swipe needed (horizontal) | Slower swipes detected |
| `velocity_threshold_y` | 0.1     | Faster swipe needed (vertical)   | Slower swipes detected |
| `confidence_ramp_up`   | 0.75    | Faster detection                 | More gradual detection |
| `confidence_decay`     | 0.5     | Faster recovery                  | Slower recovery        |
| `confidence_threshold` | 0.6     | Higher confidence needed         | Lower threshold        |
| `max_velocity_x/y`     | 2.0     | More noise tolerance             | Stricter filtering     |

### Zoom

| Parameter                  | Default | Effect of ↑                            | Effect of ↓            |
| -------------------------- | ------- | -------------------------------------- | ---------------------- |
| `finger_gap_threshold`     | 0.10    | Fingers can be further apart           | Fingers must be closer |
| `ewma_alpha`               | 0.3     | More responsive                        | Smoother               |
| `velocity_threshold`       | 0.08    | Faster spread needed                   | Slower spread detected |
| `confidence_ramp_up`       | 0.95    | Faster zoom activation                 | Slower activation      |
| `confidence_decay`         | 0.15    | Faster decay                           | Slower decay           |
| `confidence_threshold`     | 0.6     | Higher confidence needed               | Lower threshold        |
| `max_velocity`             | 2.0     | More noise tolerance                   | Stricter filtering     |
| `require_fingers_extended` | false   | When true, requires 3 fingers extended | More lenient           |

### Open Hand

| Parameter                  | Default | Effect of ↑                     | Effect of ↓          |
| -------------------------- | ------- | ------------------------------- | -------------------- |
| `min_fingers`              | 4       | More fingers needed             | Fewer fingers needed |
| `pinch_exclusion_distance` | 0.08    | More aggressive pinch exclusion | Less exclusion       |

### Thumbs

| Parameter              | Default | Effect of ↑                              | Effect of ↓             |
| ---------------------- | ------- | ---------------------------------------- | ----------------------- |
| `velocity_threshold`   | 0.2     | Faster motion needed for moving variants | Slower motion detected  |
| `ewma_alpha`           | 0.3     | More responsive                          | Smoother                |
| `hold_frames`          | 5       | Longer wait for static gesture           | Faster static detection |
| `confidence_ramp_up`   | 0.3     | Faster moving gesture detection          | Slower detection        |
| `confidence_decay`     | 0.2     | Faster decay                             | Slower decay            |
| `confidence_threshold` | 0.6     | Higher confidence needed                 | Lower threshold         |

### Finger Extension

| Parameter                | Default | Description                        |
| ------------------------ | ------- | ---------------------------------- |
| `open_ratio`             | 1.20    | Tip/PIP ratio to consider extended |
| `close_ratio`            | 1.10    | Hysteresis threshold (lower)       |
| `motion_speed_threshold` | 0.15    | Speed above which logic relaxes    |
| `motion_sigmoid_k`       | 20.0    | Sharpness of relaxation curve      |

---

## System Control (`system_control`)

### Cursor

| Parameter                | Default | Effect of ↑                       | Effect of ↓             |
| ------------------------ | ------- | --------------------------------- | ----------------------- |
| `smoothing_factor`       | 0.6     | Less smoothing, more responsive   | More smoothing, laggy   |
| `speed_multiplier`       | 1.9     | Faster cursor movement            | Slower, finer movement  |
| `bounds_padding`         | 1       | Larger edge safety zone           | Cursor closer to edges  |
| `precision_damping`      | 0.3     | Less damping in precision mode    | Finer precision control |
| `dead_zone`              | 0.01    | Larger dead zone (less jitter)    | More responsive         |
| `magnetic_radius`        | 0.005   | Larger magnetic area              | Smaller magnetic area   |
| `magnetic_factor`        | 0.7     | Less magnetic effect              | Stronger sticking       |
| `fallback_screen_width`  | 1920    | Default width if detection fails  | -                       |
| `fallback_screen_height` | 1080    | Default height if detection fails | -                       |

### Click

| Parameter              | Default | Description                                  |
| ---------------------- | ------- | -------------------------------------------- |
| `double_click_timeout` | 0.5     | Max seconds between pinches for double-click |
| `drag_hold_duration`   | 1.0     | Seconds pinch must be held to start drag     |

### Zoom

| Parameter         | Default | Description                        |
| ----------------- | ------- | ---------------------------------- |
| `sensitivity`     | 1.9     | Zoom action multiplier             |
| `speed_neutral`   | 1.0     | Neutral velocity (no modulation)   |
| `speed_factor`    | 0.2     | Max velocity influence (±20%)      |
| `base_delay`      | 0.1     | Base delay between zoom keypresses |
| `use_system_zoom` | true    | Use Ctrl+Plus/Minus for zoom       |

### Swipe (Velocity Sensitivity per Direction)

Each swipe direction has its own velocity sensitivity config:

**`swipe_left` / `swipe_right`** (Workspace Switch):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity` | 1.0 | Action multiplier |
| `speed_neutral` | 1.0 | Neutral velocity |
| `speed_factor` | 0.3 | Max velocity influence (±30%) |
| `base_delay` | 0.3 | Base delay between switches |

**`swipe_up` / `swipe_down`** (Scroll):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity` | 5.0 | Scroll multiplier (higher = faster) |
| `speed_neutral` | 1.0 | Neutral velocity |
| `speed_factor` | 0.4 | Max velocity influence (±40%) |
| `base_delay` | 0.05 | Base delay (lower = smoother scroll) |

### Thumbs Moving (Volume/Brightness)

Each thumbs moving gesture has velocity sensitivity:

| Parameter       | Default | Description                   |
| --------------- | ------- | ----------------------------- |
| `sensitivity`   | 1.0     | Action multiplier             |
| `speed_neutral` | 0.5     | Neutral velocity              |
| `speed_factor`  | 0.4     | Max velocity influence (±40%) |
| `base_delay`    | 0.15    | Base delay between actions    |

---

## Bimanual Gestures (`bimanual_gestures`)

| Parameter                     | Default | Description                      |
| ----------------------------- | ------- | -------------------------------- |
| `hand_still_threshold`        | 0.3     | Max velocity for "still" hand    |
| `pan_velocity_threshold`      | 0.4     | Min velocity for pan detection   |
| `draw_velocity_threshold`     | 0.2     | Min velocity for draw mode       |
| `precision_damping_factor`    | 0.3     | Damping in precision cursor mode |
| `two_hand_distance_threshold` | 0.1     | Min distance between hands       |
| `warp_min_distance`           | 0.3     | Min distance for warp gesture    |

---

## Visual Feedback (`visual_feedback`)

### General

| Parameter                      | Default | Description                      |
| ------------------------------ | ------- | -------------------------------- |
| `enabled`                      | true    | Master toggle for overlays       |
| `show_hand_skeleton`           | true    | Draw hand skeleton lines         |
| `show_fingertips`              | true    | Draw fingertip markers           |
| `show_cursor_preview`          | true    | Show cursor destination          |
| `show_gesture_name`            | true    | Display detected gesture         |
| `overlay_opacity`              | 0.7     | Overlay transparency (0-1)       |
| `velocity_arrow_scale`         | 0.25    | Scale of velocity arrows         |
| `velocity_threshold_highlight` | 0.8     | Threshold for arrow highlighting |
| `fingertip_dim_factor`         | 0.4     | Dimming for inactive fingertips  |

### Debug Panel

| Parameter          | Default | Description             |
| ------------------ | ------- | ----------------------- |
| `margin`           | 12      | Panel margin (pixels)   |
| `spacing`          | 8       | Line spacing            |
| `base_width`       | 240     | Panel width             |
| `line_height`      | 16      | Height per line         |
| `title_height`     | 20      | Title section height    |
| `background_alpha` | 0.45    | Background transparency |

### Gesture Panel

| Parameter       | Default | Description             |
| --------------- | ------- | ----------------------- |
| `max_height`    | 200     | Max panel height        |
| `panel_y`       | 10      | Y position from top     |
| `panel_left_x`  | 10      | X position from left    |
| `panel_width`   | 300     | Panel width             |
| `overlay_alpha` | 0.7     | Background transparency |
| `line_spacing`  | 20      | Spacing between lines   |

### Cursor Preview

| Parameter          | Default | Description                   |
| ------------------ | ------- | ----------------------------- |
| `trail_fade_time`  | 0.5     | Trail fade duration (seconds) |
| `circle_radius`    | 15      | Cursor circle radius          |
| `crosshair_length` | 20      | Crosshair arm length          |
| `crosshair_gap`    | 8       | Gap from center               |

### Animation

| Parameter         | Default | Description                    |
| ----------------- | ------- | ------------------------------ |
| `pulse_frequency` | 2.0     | Pulse animation frequency (Hz) |

---

## Camera (`camera`)

| Parameter | Default | Description         |
| --------- | ------- | ------------------- |
| `index`   | 0       | Camera device index |
| `width`   | 640     | Capture width       |
| `height`  | 480     | Capture height      |
| `fps`     | 60      | Requested FPS       |

---

## Performance (`performance`)

| Parameter                  | Default | Description                     |
| -------------------------- | ------- | ------------------------------- |
| `use_gpu`                  | true    | Enable GPU acceleration         |
| `show_fps`                 | true    | Show FPS counter                |
| `show_debug_info`          | false   | Verbose terminal debug          |
| `max_hands`                | 2       | Maximum hands to detect         |
| `min_detection_confidence` | 0.7     | MediaPipe detection threshold   |
| `min_tracking_confidence`  | 0.3     | MediaPipe tracking threshold    |
| `gesture_history_maxlen`   | 16      | Per-hand gesture history buffer |
| `bimanual_history_maxlen`  | 10      | Two-hand gesture history buffer |

---

## Display (`display`)

| Parameter             | Default | Description                         |
| --------------------- | ------- | ----------------------------------- |
| `window_width`        | 1280    | Camera window width                 |
| `window_height`       | 720     | Camera window height                |
| `flip_horizontal`     | true    | Mirror video output                 |
| `fps_update_interval` | 30      | Frames between FPS/config updates   |
| `visual_mode`         | "full"  | Display mode (full/indicator/debug) |
| `show_camera_window`  | true    | Explicitly show camera window       |

### Status Indicator

| Parameter | Default | Description                |
| --------- | ------- | -------------------------- |
| `enabled` | true    | Enable floating indicators |
| `size`    | 64      | Indicator size (pixels)    |
| `opacity` | 0.8     | Indicator opacity          |

**Per-hand settings** (`left_hand`, `right_hand`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `position` | "top-right"/"top-left" | Screen position |
| `margin_x` | 20/90 | X margin from edge |
| `margin_y` | 20 | Y margin from edge |

---

## Velocity Modulation Formula

For velocity-sensitive actions (zoom, scroll, volume, brightness):

$$S_{eff} = sensitivity \times [1.0 + speed\_factor \times (V_{norm} - speed\_neutral)]$$

The effective delay between actions:

$$delay = \frac{base\_delay}{S_{eff}}$$

- **Faster gestures** → Higher $S_{eff}$ → Shorter delay → More actions
- **Slower gestures** → Lower $S_{eff}$ → Longer delay → Fewer actions

---

## Using the Config Editor

1. Run: `./run_config.sh` (or `python3 config_gui.py`)
2. Navigate to the parameter category
3. Hover ℹ️ icons for parameter descriptions
4. Edit values and click "Save & Apply"
5. The HANDS app auto-reloads every `fps_update_interval` frames

---

_Last Updated: 2025_  
_System: HANDS v2.0_
