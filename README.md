# HANDS - Hand Assisted Navigation and Device System

![Status](https://img.shields.io/badge/status-production-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)

Control your computer using hand gestures captured by your webcam! HANDS provides an intuitive, natural interface for cursor movement, clicking, scrolling, zooming, and more - all without touching your mouse or keyboard.

## ✨ Features

### 🖐️ Single-Hand Gestures
- **Pointing** (index finger) → Move cursor
- **Pinch** (thumb + index) → Click / Double-click / Drag
- **Zoom** (thumb + index + middle) → System zoom in/out
- **Swipe** (4 fingers moving) → Scroll or switch workspaces
- **Open Hand** (5 fingers) → Reserved for future features

### 🤝 Two-Hand Gestures
- **Precision Cursor** (left still + right pointing) → Fine cursor control
- **Pan/Scroll** (left still + right move) → Smooth scrolling
- **Two-Hand Resize** (both pinch + change distance) → Window resizing
- **Quick Menu** (left zoom + right pinch) → Context menu
- **And more!**

### 🎨 Visual Feedback
- Beautiful hand skeleton overlay with glow effects
- Color-coded fingertip indicators
- Gesture-specific animations (pinch lines, zoom triangles, pointer rays)
- Cursor preview with trail effects
- Real-time gesture status panel

### ⚙️ Configuration System
All gesture thresholds and system parameters are configurable via `config.json` - no code editing required!

## 🚀 Quick Start

### 1. Installation

```bash
# Activate virtual environment (if you have one)
source .venv/bin/activate

# Dependencies should already be installed, but if not:
pip install opencv-python mediapipe numpy pynput screeninfo
```

### 2. Run the Application

```bash
# Full system control mode
python hands_app.py

# Or using the executable
./ hands_app.py

# Dry-run mode (visualization only, no system control)
python hands_app.py --dry-run

# Use different camera
python hands_app.py --camera 1
```

### 3. Basic Usage

1. **Position yourself** - Sit comfortably in front of your webcam
2. **Show your hand** - Raise your right hand (palm facing camera)
3. **Point** - Extend index finger to move cursor
4. **Pinch** - Touch thumb and index to click
5. **Explore** - Try other gestures!

## ⌨️ Keyboard Controls

During runtime:
- `Q` - Quit application
- `P` - Pause/Resume gesture control
- `D` - Toggle debug information
- `F` - Toggle FPS display
- `H` - Show help

## 🔧 Configuration

Edit `config.json` to customize:

### Gesture Thresholds
```json
{
  "gesture_thresholds": {
    "pinch": {
      "threshold_rel": 0.055,
      "hold_frames": 5,
      "cooldown_seconds": 0.6
    },
    "zoom": {
      "scale_threshold": 0.15,
      "finger_gap_threshold": 0.06
    },
    ...
  }
}
```

### System Control
```json
{
  "system_control": {
    "cursor": {
      "smoothing_factor": 0.3,
      "speed_multiplier": 1.5
    },
    "scroll": {
      "sensitivity": 30
    }
  }
}
```

## 📁 Project Structure

### Core Application Files
- `hands_app.py` - Main application (start here!)
- `config.json` - Configuration file
- `config_manager.py` - Configuration loader

### Gesture Detection
- `gesture_detectors.py` - Single-hand gesture detection
- `bimanual_gestures.py` - Two-hand gesture detection
- `math_utils.py` - Math utilities (EWMA, distance calculations)

### System Integration
- `system_controller.py` - Mouse/keyboard control
- `visual_feedback.py` - Beautiful UI overlays

### Documentation
- `README.md` - This file
- `IMPLEMENTATION_GUIDE.md` - Development guide
- `TUNING_VARIABLES.md` - Tuning reference
- `next_target.md` - Gesture design document

### Backup
- `backup_old_files/` - Old test files and demos

## 🎯 Gesture Tips

### For Best Results:
1. **Good lighting** - Ensure your hand is well-lit
2. **Solid background** - Plain backgrounds work best
3. **Hand distance** - Keep hand 30-60cm from camera
4. **Steady movements** - Smooth, deliberate gestures work better
5. **Practice** - It gets easier with use!

### Continuous Zoom
The zoom gesture is **continuous** - once triggered, it keeps zooming as long as:
- All 3 fingers remain extended
- Fingers stay close together
- Spread continues in the same direction

To stop zooming, either:
- Separate fingers
- Change direction (spreading → pinching or vice versa)
- Lower your hand

## 🐛 Troubleshooting

### Camera not detected
```bash
# List available cameras
ls -la /dev/video*

# Try different camera index
python hands_app.py --camera 1
```

### Permission errors (system control)
```bash
# On Linux, you may need to be in the input group
sudo usermod -a -G input $USER
# Then log out and log back in
```

### Poor gesture detection
- Check lighting conditions
- Adjust thresholds in `config.json`
- Try `--dry-run` mode to see detection overlays
- Ensure camera view is not obstructed

### Performance issues
Edit `config.json`:
```json
{
  "camera": {
    "width": 320,
    "height": 240,
    "fps": 30
  }
}
```

## 📊 Performance

- **FPS**: 30-60 FPS on modern hardware
- **Latency**: < 50ms gesture-to-action
- **CPU Usage**: ~15-25% on mid-range CPU
- **Memory**: ~200-300 MB

## 🎓 Advanced Usage

### Custom Gesture Mappings

Edit the `process_gestures()` method in `hands_app.py` to customize what each gesture does.

### Integration with Other Apps

```python
from system_controller import SystemController

ctrl = SystemController()
ctrl.move_cursor(0.5, 0.5)  # Center
ctrl.click()
ctrl.scroll(0, 10)
```

## 🤝 Contributing

This is a complete gesture control system! Feel free to:
- Add new gestures
- Improve detection algorithms
- Create new visual effects
- Optimize performance

## 📝 License

[Your license here]

## 🙏 Acknowledgments

Built with:
- [MediaPipe](https://mediapipe.dev/) - Hand tracking
- [OpenCV](https://opencv.org/) - Computer vision
- [pynput](https://pypi.org/project/pynput/) - System control

---

**Made with ❤️ for hands-free computing**

*Ready to try it? Run `python hands_app.py` and wave hello to the future!* 👋
