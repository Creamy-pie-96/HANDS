# HANDS User Guide

## Overview

**HANDS (Hand Assisted Navigation and Device System)** is an AI-driven hand gesture control system that lets you operate your computer using webcam-captured hand movements. It uses MediaPipe for real-time hand tracking with GPU acceleration support.

---

## Quick Start

### 1. Installation

```bash
./install.sh
```

### 2. Run the Application

```bash
cd app
./start_hands.sh
```

### 3. Run the Config Editor (Optional)

```bash
cd app
./run_config.sh
```

---

## Gesture Controls

### Single Hand Gestures (Right Hand = Primary)

| Gesture                 | Visual | Action                                     | Enabled by Default |
| ----------------------- | ------ | ------------------------------------------ | ------------------ |
| Pointing                | 👆     | Move cursor (index finger extended)        | ✅                 |
| Pinch                   | 🤏     | Click / Drag (thumb + index together)      | ✅                 |
| Zoom In                 | 🔍+    | System zoom in (spread 3 fingers apart)    | ✅                 |
| Zoom Out                | 🔍-    | System zoom out (pinch 3 fingers together) | ✅                 |
| Swipe Up                | 👋↑    | Scroll up                                  | ✅                 |
| Swipe Down              | 👋↓    | Scroll down                                | ✅                 |
| Swipe Left              | 👋←    | Switch workspace left                      | ✅                 |
| Swipe Right             | 👋→    | Switch workspace right                     | ✅                 |
| Open Hand               | ✋     | Reserved (5 fingers extended)              | ✅                 |
| Thumbs Up               | 👍     | Reserved for future use                    | ✅                 |
| Thumbs Down             | 👎     | **Hold 3 seconds to quit app**             | ✅                 |
| Thumbs Up + Move Up     | 👍↑    | Volume up                                  | ✅                 |
| Thumbs Up + Move Down   | 👍↓    | Volume down                                | ✅                 |
| Thumbs Down + Move Up   | 👎↑    | Brightness up                              | ✅                 |
| Thumbs Down + Move Down | 👎↓    | Brightness down                            | ✅                 |

### Bimanual Gestures (Two Hands)

| Gesture          | Description                                | Action                           |
| ---------------- | ------------------------------------------ | -------------------------------- |
| Precision Cursor | Left hand still (✋) + Right pointing (👆) | Fine cursor control with damping |
| Pan              | Left hand still (✋) + Right swipe (👋)    | Two-handed scroll                |

### Disabling Gestures

Each gesture can be individually enabled or disabled in `config.json` under `gestures_enabled`. When a gesture is disabled:

- The gesture is still **detected and shown** in the indicator
- A **small red dot** appears in the bottom-right corner of the indicator
- The gesture **does not trigger any system action**

This is useful for:

- Temporarily disabling gestures you don't need
- Preventing accidental triggers during specific tasks
- Testing gesture detection without system effects

---

## Keyboard Controls

### Debug Overlays (Toggle per-gesture debug info)

| Key | Gesture   | Description                                               |
| --- | --------- | --------------------------------------------------------- |
| `Z` | Zoom      | Show zoom parameters (Gap, Spread, Velocity)              |
| `X` | Pinch     | Show pinch parameters (Distance, Threshold, Hold)         |
| `I` | Pointing  | Show pointing parameters (Distance, Speed, Extra fingers) |
| `S` | Swipe     | Show swipe parameters (Direction, Speed, Threshold)       |
| `O` | Open Hand | Show open hand parameters (Finger count, Pinch exclusion) |
| `T` | Thumbs    | Show thumbs parameters (Velocity, State)                  |

**Note**: Multiple debug overlays can be active simultaneously.

### General Controls

| Key | Action                               |
| --- | ------------------------------------ |
| `Q` | Quit application                     |
| `P` | Pause/Resume gesture control         |
| `D` | Toggle verbose debug info (terminal) |
| `F` | Toggle FPS display on-screen         |
| `H` | Print help (keyboard controls)       |

---

## Status Indicator

The floating status indicator shows which gesture is currently detected:

- **Right indicator**: Shows right hand gesture
- **Left indicator**: Shows left hand gesture (when enabled)
- **Red dot overlay**: Indicates the gesture is disabled (detected but won't trigger action)

### Indicator States

| Color  | Meaning                                         |
| ------ | ----------------------------------------------- |
| Blue   | Normal detection                                |
| Red    | Special state (paused, dry-run, exit countdown) |
| Yellow | Warning/transition state                        |

---

## Velocity Modulation

Some gestures use velocity to control action intensity:

- **Faster movement** = more frequent actions (shorter delay)
- **Slower movement** = less frequent actions (longer delay)
- **Neutral velocity** = default action rate

This applies to:

- Zoom (faster spread = faster zoom)
- Scroll (faster swipe = faster scroll)
- Volume/Brightness (faster thumb movement = faster change)

Configurable parameters:

- `sensitivity`: Base action multiplier
- `speed_neutral`: Velocity where no modulation occurs
- `speed_factor`: How much velocity affects action rate (±%)
- `base_delay`: Time between actions at neutral velocity

---

## Tuning Workflow

### In-App Tuning

1. Run the HANDS app: `./start_hands.sh`
2. Press debug keys (`Z`, `X`, `I`, `S`, `O`, `T`) to show overlays
3. Perform gestures while watching the metrics
4. Note which values are close to thresholds

### Config Editor Tuning

1. Run the config editor: `./run_config.sh`
2. Navigate to the parameter you want to adjust
3. Hover ℹ️ icons for parameter descriptions
4. Edit values and click "Save & Apply"
5. The app auto-reloads config every 30 frames

### Manual Config Editing

Edit `source_code/config/config.json` directly. The app will auto-reload changes.

---

## Gesture-Specific Metadata (Debug Overlays)

### Zoom (`Z`)

- `Gap`: Distance between index & middle fingers
- `Spr`: Spread distance (thumb to paired fingers)
- `Chg`: Relative spread change (%)
- `Inr`: Confidence/inertia level (0-1)
- `Vel`: Spread change velocity
- `VCon`: Velocity consistency score
- `Rsn`: Reason for non-detection

### Pinch (`X`)

- `Dist`: Thumb-index distance
- `Thrs`: Configured threshold
- `Hold`: Current/required hold frames
- `CDwn`: Cooldown remaining (seconds)

### Pointing (`I`)

- `Dist`: Index finger extension from palm
- `MinD`: Minimum required distance
- `Spd`: Current hand speed
- `MaxS`: Maximum allowed speed
- `Xtra`: Extra fingers extended / max allowed
- `Rsn`: Reason for non-detection

### Swipe (`S`)

- `Dir`: Swipe direction (up/down/left/right)
- `Spd`: Current velocity
- `Thrs`: Velocity threshold
- `Hist`: History frames / minimum required
- `CDwn`: Cooldown remaining (seconds)
- `Rsn`: Reason for non-detection

### Open Hand (`O`)

- `Cnt`: Fingers extended / minimum required
- `TIDist`: Thumb-index distance
- `Pinch`: Is hand in pinch position?
- `Rsn`: Reason for non-detection

### Thumbs (`T`)

- `Vel`: Velocity vector (vx, vy)
- `State`: Current thumbs state (up/down/moving)

---

## Display Modes

Configure in `config.json` under `display`:

| Mode        | Description                                      |
| ----------- | ------------------------------------------------ |
| `full`      | Camera window + status indicators + all overlays |
| `indicator` | Status indicators only (no camera window)        |
| `debug`     | Camera window with all debug info                |

Additional options:

- `show_camera_window`: Explicitly show/hide camera preview
- `status_indicator.enabled`: Enable/disable floating indicators

---

## Troubleshooting

### Gesture Not Detected

1. Check lighting conditions (bright, even lighting works best)
2. Ensure hand is fully visible in camera frame
3. Move hand more slowly (fast motion can disrupt tracking)
4. Check debug overlay for threshold values

### Gesture Detected But No Action

1. Check if gesture is enabled in `gestures_enabled` config
2. Look for red dot on indicator (disabled gesture)
3. Verify system control is not paused (`P` key)
4. Check if in dry-run mode (started with `--dry-run`)

### Volume/Brightness Not Working

- **Volume**: Requires PulseAudio (`pactl`) or XF86 media keys
- **Brightness**: Requires `brightnessctl`, `xbacklight`, or DBus interface

### High Latency

1. Enable GPU acceleration: `performance.use_gpu: true`
2. Reduce camera resolution: `camera.width`, `camera.height`
3. Increase smoothing: `cursor.smoothing_factor`

### Status Indicator Click-Through Issues

On some X11 systems, the indicator may capture clicks. The app uses XShape extension to create an empty input region. If issues persist, try:

- Running on Wayland instead of X11
- Adjusting window manager settings

---

## System Requirements

- **Python**: 3.10+
- **Camera**: Webcam with 640x480+ resolution
- **GPU** (optional): OpenGL ES 3.1+ or Vulkan for GPU acceleration
- **Linux**: X11 or Wayland display server

### Dependencies

- PyQt6 (GUI)
- MediaPipe (Hand tracking)
- OpenCV (Camera capture)
- pynput (Mouse/keyboard control)
- NumPy

---

_Last Updated: 2025_  
_System: HANDS v2.0_
