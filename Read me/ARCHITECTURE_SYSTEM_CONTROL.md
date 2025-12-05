# System Controller Architecture

This document explains how HANDS translates gesture detections into system actions.

---

## System Control Architecture

The system control layer is now split into two parts:
1.  **The Body (`SystemController`)**: Handles the raw interaction with the OS (Mouse, Keyboard, Screen).
2.  **The Brain (`ActionDispatcher`)**: Decides *what* the body should do based on User Configuration.

### System Controller (`source_code/utils/system_controller.py`)
*   **Role**: Execution only.
*   **Dependencies**: `pynput`, `screeninfo`.
*   **New Features**:
    *   `execute_key_combo`: Parses strings like "ctrl+c" and presses keys.
    *   `@exposed_action`: Marks methods essentially safe for user binding.

### Action Dispatcher (`source_code/app/action_dispatcher.py`)
*   **Role**: Decision making.
*   **Input**: Gesture Names (Strings) + Metadata.
*   **Config**: Loaded from `config.json` "action_map".

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

| Gesture             | Method                   | Action                      |
| ------------------- | ------------------------ | --------------------------- |
| `pointing`          | `move_cursor()`          | Mouse movement (smooth)     |
| `pinch`             | `handle_pinch_gesture()` | Click/drag                  |
| `zoom_in/out`       | `zoom()`                 | Ctrl+Shift+= / Ctrl+-       |
| `swipe_up/down`     | `scroll()`               | Mouse wheel (velocity-based) |
| `swipe_left/right`  | `workspace_switch()`     | Ctrl+Alt+Arrow (velocity-based) |
| `thumbs_up_moving_*`| `increase/decrease_volume()` | Volume control (velocity-based) |
| `thumbs_down_moving_*` | `increase/decrease_brightness()` | Brightness control (velocity-based) |

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

Uses Ctrl+Plus/Minus for system-level zoom with velocity-modulated rate limiting:

```python
def zoom(self, zoom_in: bool = True, velocity_norm: float = 1.0):
    if self.paused:
        return

    try:
        # Use the VelocitySensitivity calculator for rate limiting
        if not self.velocity_sensitivity['zoom'].try_act(velocity_norm):
            return

        with self.keyboard.pressed(Key.ctrl):
            if zoom_in:
                # Press Ctrl + + (Shift + =)
                self.keyboard.press(Key.shift)
                self.keyboard.press('=')  # Shift+= is +
                self.keyboard.release('=')
                self.keyboard.release(Key.shift)
            else:
                self.keyboard.press('-')
                self.keyboard.release('-')
    except Exception as e:
        print(f"⚠ Error zooming: {e}")
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

### Volume & Brightness

Thumbs gestures with movement control volume and brightness:

| Gesture | Action |
| --- | --- |
| `thumbs_up_moving_up` | Increase volume |
| `thumbs_up_moving_down` | Decrease volume |
| `thumbs_down_moving_up` | Increase brightness |
| `thumbs_down_moving_down` | Decrease brightness |

All use **velocity-modulated rate limiting** - faster movements = faster volume/brightness changes.

### Volume Control

Uses exposed atomic actions:

```python
@exposed_action
def increase_volume(self, velocity_norm: float = 1.0):
    """Increase System Volume."""
    self.perform_velocity_action('thumbs_up_moving_up', velocity_norm, 
                               lambda: self._volume_change(+5))

@exposed_action
def decrease_volume(self, velocity_norm: float = 1.0):
    """Decrease System Volume."""
    self.perform_velocity_action('thumbs_up_moving_down', velocity_norm, 
                               lambda: self._volume_change(-5))
```

The underlying `_volume_change` method uses XF86 media keys with pactl fallback.

### Brightness Control

Uses exposed atomic actions:

```python
@exposed_action
def increase_brightness(self, velocity_norm: float = 1.0):
    """Increase Screen Brightness."""
    self.perform_velocity_action('thumbs_down_moving_up', velocity_norm, 
                               lambda: self._brightness_change(+5))

@exposed_action
def decrease_brightness(self, velocity_norm: float = 1.0):
    """Decrease Screen Brightness."""
    self.perform_velocity_action('thumbs_down_moving_down', velocity_norm, 
                               lambda: self._brightness_change(-5))
```

The underlying `_brightness_change` method uses multiple fallback methods for Linux compatibility:
1. **brightnessctl** (most common on modern Linux)
2. **xbacklight** (legacy X11 method)
3. **DBus** (GNOME/KDE brightness interface)

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

The `try_act()` method standardizes velocity-sensitive actions:

```python
def try_act(self, velocity_norm: float) -> bool:
    """
    Check if action is allowed and record it if so.
    
    Combines should_act() and record_action() for convenience.
    
    Args:
        velocity_norm: Normalized gesture velocity
    
    Returns:
        True if action was allowed (and recorded), False if rate-limited
    """
    if self.should_act(velocity_norm):
        self.record_action()
        return True
    return False
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
│ CORE METHODS:                                               │
│ + move_cursor(x, y, precision_mode)                         │
│ + handle_pinch_gesture(pinch_detected)                      │
│ + click(button, double=False)                               │
│ + zoom(zoom_in, velocity_norm)                              │
│ + scroll(dx, dy, velocity_norm)                             │
│ + workspace_switch(direction)                               │
│ + swipe(direction, velocity_norm)                           │
│ + thumbs_action(gesture_name, velocity_norm)                │
│ + perform_velocity_action(name, velocity, callback)         │
│                                                              │
│ EXPOSED ATOMIC ACTIONS (@exposed_action):                   │
│ + left_click()                                              │
│ + right_click()                                             │
│ + double_click()                                            │
│ + scroll_up/down/left/right(velocity_norm)                  │
│ + zoom_in/out(velocity_norm)                                │
│ + next_workspace/previous_workspace(velocity_norm)          │
│ + increase/decrease_volume(velocity_norm)                   │
│ + increase/decrease_brightness(velocity_norm)               │
│                                                              │
│ PRIVATE METHODS:                                            │
│ - _volume_change(delta)                                     │
│ - _brightness_change(delta)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  VelocitySensitivity                         │
├─────────────────────────────────────────────────────────────┤
│ - config: VelocitySensitivityConfig                         │
│ - _last_action_time: float                                  │
├─────────────────────────────────────────────────────────────┤
│ + calculate_effective_sensitivity(velocity_norm)            │
│ + calculate_min_delay(velocity_norm)                        │
│ + should_act(velocity_norm) -> bool                         │
│ + record_action()                                           │
│ + try_act(velocity_norm) -> bool                            │
│ + reset()                                                   │
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
