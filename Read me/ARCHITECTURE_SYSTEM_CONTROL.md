# System Controller Architecture

This document explains how HANDS translates gesture detections into system actions.

---

## Overview

The System Controller (`system_controller.py`) is the bridge between gesture detection and OS-level actions. It handles:

- Mouse control (move, click, drag)
- Keyboard shortcuts (zoom, workspace switch)
- Media controls (volume, brightness)
- Rate limiting with velocity modulation

---

## Core Concepts

### 1. Velocity Sensitivity

Actions that repeat (scroll, zoom, volume) use velocity to modulate timing:

```python
@dataclass
class VelocitySensitivityConfig:
    base_sensitivity: float = 1.0   # Action multiplier
    speed_neutral: float = 1.0      # Neutral velocity (no modulation)
    speed_factor: float = 0.2       # Max velocity influence (±20%)
    base_delay: float = 0.5         # Base time between actions

class VelocitySensitivity:
    def __init__(self, config: VelocitySensitivityConfig):
        self.config = config
        self.last_action_time = 0.0

    def calculate_effective_sensitivity(self, velocity_norm: float) -> float:
        """
        Seff = base_sensitivity × [1.0 + speed_factor × (Vnorm - speed_neutral)]
        """
        c = self.config
        modulation = c.speed_factor * (velocity_norm - c.speed_neutral)
        return c.base_sensitivity * (1.0 + modulation)

    def should_perform_action(self, velocity_norm: float) -> bool:
        """Check if enough time has passed since last action."""
        effective_sens = self.calculate_effective_sensitivity(velocity_norm)
        effective_delay = self.config.base_delay / max(0.1, effective_sens)

        now = time.time()
        if now - self.last_action_time >= effective_delay:
            self.last_action_time = now
            return True
        return False
```

**Effect:**

- Faster gestures → higher effective sensitivity → shorter delay → more actions
- Slower gestures → lower effective sensitivity → longer delay → fewer actions

### 2. Gesture-to-Action Mapping

Each gesture type has a dedicated action method:

| Gesture             | Method                   | Action            |
| ------------------- | ------------------------ | ----------------- |
| `pointing`          | `move_cursor()`          | Mouse movement    |
| `pinch`             | `handle_pinch_gesture()` | Click/drag        |
| `zoom_in/out`       | `zoom()`                 | Ctrl+Plus/Minus   |
| `swipe_up/down`     | `scroll()`               | Mouse wheel       |
| `swipe_left/right`  | `workspace_switch()`     | Ctrl+Alt+Arrow    |
| `thumbs_*_moving_*` | `thumbs_action()`        | Volume/Brightness |

---

## Mouse Control

### Cursor Movement

Uses EWMA (Exponential Weighted Moving Average) for smooth cursor positioning:

```python
def move_cursor(self, x: float, y: float, precision_mode: bool = False):
    # Map normalized [0,1] to screen pixels
    screen_x = x * self.screen_width
    screen_y = y * self.screen_height

    # Apply EWMA smoothing
    # new_pos = alpha * target + (1 - alpha) * current
    self.cursor_x = self.smoothing * screen_x + (1 - self.smoothing) * self.cursor_x
    self.cursor_y = self.smoothing * screen_y + (1 - self.smoothing) * self.cursor_y

    # Apply precision mode damping
    if precision_mode:
        dx = self.cursor_x - self.last_x
        dy = self.cursor_y - self.last_y
        self.cursor_x = self.last_x + dx * self.precision_damping
        self.cursor_y = self.last_y + dy * self.precision_damping

    # Dead zone: skip tiny movements
    if abs(screen_x - self.cursor_x) < self.dead_zone * self.screen_width:
        return

    # Actually move the mouse
    self.mouse.position = (int(self.cursor_x), int(self.cursor_y))
    self.last_x, self.last_y = self.cursor_x, self.cursor_y
```

### Click & Drag

Pinch gesture triggers click with drag detection:

```python
def handle_pinch_gesture(self, pinch_detected: bool):
    now = time.time()

    if pinch_detected:
        if self.pinch_start_time is None:
            # Pinch just started
            self.pinch_start_time = now

            # Check for double click
            if now - self.last_click_time < self.double_click_timeout:
                self.click(double=True)
                self.click_count = 0
            else:
                self.click()
                self.click_count = 1
                self.last_click_time = now

        elif now - self.pinch_start_time > self.drag_hold_duration:
            # Pinch held long enough - start drag
            if not self.is_dragging:
                self.mouse.press(Button.left)
                self.is_dragging = True

    else:  # Pinch released
        if self.is_dragging:
            self.mouse.release(Button.left)
            self.is_dragging = False
        self.pinch_start_time = None
```

---

## Keyboard Actions

### System Zoom

Uses Ctrl+Plus/Minus for system-level zoom:

```python
def zoom(self, zoom_in: bool, velocity_norm: float = 1.0):
    if not self.velocity_sensitivity['zoom'].should_perform_action(velocity_norm):
        return False  # Rate limited

    with self.keyboard.pressed(Key.ctrl):
        if zoom_in:
            self.keyboard.press(Key.equal)  # Plus (=) key
            self.keyboard.release(Key.equal)
        else:
            self.keyboard.press(Key.minus)
            self.keyboard.release(Key.minus)

    return True
```

### Workspace Switch

Uses Ctrl+Alt+Arrow for virtual desktop switching:

```python
def workspace_switch(self, direction: str):
    if direction not in ['left', 'right']:
        return

    with self.keyboard.pressed(Key.ctrl):
        with self.keyboard.pressed(Key.alt):
            if direction == 'left':
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
            else:
                self.keyboard.press(Key.right)
                self.keyboard.release(Key.right)
```

---

## Scroll

Maps swipe gestures to mouse wheel events:

```python
def scroll(self, dx: int, dy: int, velocity_norm: float = 1.0):
    # Determine primary direction for rate limiting
    if abs(dy) >= abs(dx):
        direction = 'swipe_up' if dy > 0 else 'swipe_down'
    else:
        direction = 'swipe_right' if dx > 0 else 'swipe_left'

    if not self.velocity_sensitivity[direction].should_perform_action(velocity_norm):
        return False  # Rate limited

    # Perform scroll
    self.mouse.scroll(dx, dy)
    return True
```

---

## Volume & Brightness

### Volume Control

Uses XF86 media keys with pactl fallback:

```python
def _volume_change(self, delta: int):
    try:
        # Method 1: Media keys
        if delta > 0:
            self.keyboard.press(Key.media_volume_up)
            self.keyboard.release(Key.media_volume_up)
        else:
            self.keyboard.press(Key.media_volume_down)
            self.keyboard.release(Key.media_volume_down)
    except Exception:
        # Method 2: PulseAudio CLI
        import subprocess
        change = '+5%' if delta > 0 else '-5%'
        subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', change])
```

### Brightness Control

Multiple fallback methods for Linux compatibility:

```python
def _brightness_change(self, delta: int):
    import subprocess

    # Method 1: brightnessctl (most common)
    try:
        change = '+5%' if delta > 0 else '5%-'
        result = subprocess.run(['brightnessctl', 'set', change],
                               capture_output=True, timeout=1)
        if result.returncode == 0:
            return
    except FileNotFoundError:
        pass

    # Method 2: xbacklight (legacy X11)
    try:
        arg = '-inc' if delta > 0 else '-dec'
        result = subprocess.run(['xbacklight', arg, '5'],
                               capture_output=True, timeout=1)
        if result.returncode == 0:
            return
    except FileNotFoundError:
        pass

    # Method 3: DBus (GNOME/KDE)
    try:
        import dbus
        bus = dbus.SessionBus()
        brightness = bus.get_object(
            'org.gnome.SettingsDaemon.Power',
            '/org/gnome/SettingsDaemon/Power'
        )
        iface = dbus.Interface(brightness,
            'org.gnome.SettingsDaemon.Power.Screen')
        current = iface.GetPercentage()
        new = max(5, min(100, current + (5 if delta > 0 else -5)))
        iface.SetPercentage(new)
    except:
        pass  # No method available
```

---

## Velocity Sensitivity Per Gesture

Each gesture type has independent velocity sensitivity:

```python
# In SystemController.__init__()
self.velocity_sensitivity = {
    'zoom': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('zoom')
    ),
    'swipe_left': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('swipe_left')
    ),
    'swipe_right': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('swipe_right')
    ),
    'swipe_up': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('swipe_up')
    ),
    'swipe_down': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('swipe_down')
    ),
    'thumbs_up_moving_up': VelocitySensitivity.from_dict(
        get_velocity_sensitivity_config('thumbs_up_moving_up')
    ),
    # ... etc
}
```

Config values from `config.json`:

```json
"swipe": {
  "swipe_up": {
    "sensitivity": 5.0,       // High for scrolling
    "speed_neutral": 1.0,
    "speed_factor": 0.4,
    "base_delay": 0.05        // Low for smooth scroll
  },
  "swipe_left": {
    "sensitivity": 1.0,       // Normal for workspace
    "speed_neutral": 1.0,
    "speed_factor": 0.3,
    "base_delay": 0.3         // Higher to prevent rapid switching
  }
}
```

---

## Unified Action Method

The `perform_velocity_action()` method standardizes velocity-sensitive actions:

```python
def perform_velocity_action(self, gesture_name: str, velocity_norm: float,
                           action_callback: Callable) -> bool:
    """
    Perform an action with velocity-modulated rate limiting.

    Args:
        gesture_name: Name for rate limiting lookup
        velocity_norm: Current gesture velocity
        action_callback: Function to call if not rate-limited

    Returns:
        True if action was performed, False if rate-limited
    """
    sens = self.velocity_sensitivity.get(gesture_name)
    if sens is None:
        # No rate limiting for this gesture
        action_callback()
        return True

    if sens.should_perform_action(velocity_norm):
        action_callback()
        return True

    return False  # Rate limited
```

---

## Class Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SystemController                          │
├─────────────────────────────────────────────────────────────┤
│ - mouse: Controller                                          │
│ - keyboard: Controller                                       │
│ - velocity_sensitivity: Dict[str, VelocitySensitivity]      │
│ - cursor_x, cursor_y: float                                 │
│ - is_dragging: bool                                         │
├─────────────────────────────────────────────────────────────┤
│ + move_cursor(x, y, precision_mode)                         │
│ + handle_pinch_gesture(pinch_detected)                      │
│ + click(double=False)                                       │
│ + zoom(zoom_in, velocity_norm)                              │
│ + scroll(dx, dy, velocity_norm)                             │
│ + workspace_switch(direction)                               │
│ + swipe(direction, velocity_norm)                           │
│ + thumbs_action(gesture_name, velocity_norm)                │
│ + perform_velocity_action(name, velocity, callback)         │
│ - _volume_change(delta)                                     │
│ - _brightness_change(delta)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  VelocitySensitivity                         │
├─────────────────────────────────────────────────────────────┤
│ - config: VelocitySensitivityConfig                         │
│ - last_action_time: float                                   │
├─────────────────────────────────────────────────────────────┤
│ + calculate_effective_sensitivity(velocity_norm)            │
│ + should_perform_action(velocity_norm) -> bool              │
│ + from_dict(config_dict) -> VelocitySensitivity             │
└─────────────────────────────────────────────────────────────┘
```

---

## Platform Considerations

| Feature            | Linux                       | Windows           | macOS         |
| ------------------ | --------------------------- | ----------------- | ------------- |
| Mouse control      | pynput ✅                   | pynput ✅         | pynput ✅     |
| Keyboard shortcuts | pynput ✅                   | pynput ✅         | pynput ✅     |
| System zoom        | Ctrl+/- ✅                  | Ctrl+/- ✅        | Cmd+/- ⚠️     |
| Workspace switch   | Ctrl+Alt+Arrow ✅           | Win+Ctrl+Arrow ⚠️ | Ctrl+Arrow ⚠️ |
| Volume             | Media keys + pactl ✅       | Media keys ✅     | Media keys ✅ |
| Brightness         | brightnessctl/xbacklight ✅ | WMI ⚠️            | IOKit ⚠️      |

⚠️ = May require modification for full support

---

## Related Files

- `source_code/utils/system_controller.py` - System control implementation
- `source_code/config/config_manager.py` - Velocity sensitivity config loading
- `source_code/app/hands_app.py` - Gesture to action routing
