# My design philosophy
**Gesture Detection Module for HANDS (Hand Assisted Navigation and Device System)**

**This module provides modular gesture detectors that work with MediaPipe hand landmarks.**
**All detectors operate on normalized coordinates (0..1) to be resolution-independent.**

**Design principles:**
- **Reuse existing utilities from math_utils.py (EWMA, euclidean, landmarks_to_array)**
- **Keep detectors stateless where possible; use small state objects for temporal logic**
- **Compute hand metrics once per frame and pass to all detectors**
- **Operate in normalized space; convert to pixels only for visualization**



# New Refractoring I will be adding:
You should implement a configuration‑driven gesture→action system with a dynamic GUI dispatcher, so users can map any 0–528 combinations to functions or key binds without touching core code.[1][2][3]

## High-level goals

- Keep all platform-specific logic isolated in SystemController (volume, brightness, scroll, zoom, workspaces, etc.).[1]
- Make gesture→action mapping fully data‑driven via config.json, but compact (a list of mappings, not 528 keys).[2][3]
- Use config_gui.py as the primary tool to:
  - Discover available gestures and actions automatically.
  - Let users create mappings with dropdowns and text inputs.
  - Enforce uniqueness (no duplicate gesture combos; optional rules for key binds).[3][2]
- Use a dispatcher layer in the runtime that:
  - Prioritizes bimanual combos over single-hand actions.
  - Supports both “function” and “key” actions.
  - Scales to any number of configured combos in O(1) time via hash maps.[2][1]

## Config design (action_map)

Add an action_map list to config.json:

```json
"action_map": [
  {
    "left": "pinch",         // "none" means no gesture on that hand
    "right": "openhand",
    "type": "key",
    "keys": "ctrl+shift+p"
  },
  {
    "left": "none",
    "right": "openhand",
    "type": "function",
    "name": "toggle_left_click"
  }
]
```

- Semantics:
  - Bimanual combo: left != "none" and right != "none".
  - Single right: left == "none", right != "none".
  - Single left: left != "none", right == "none".
- This keeps JSON small and readable; the “528” space only exists conceptually, not as 528 fields.[3][2]

## SystemController actions

- Keep existing methods as modular actions:[1]
  - movecursor(x, y, precisionmode)
  - handlepinchgesture(...)
  - zoom(zoomin, velocitynorm)
  - scroll(dx, dy, velocitynorm)
  - workspaceswitch(direction)
  - volumechange(delta)
  - brightnesschange(delta)
- Add a generic key combo action:
  - execute_key_combo(keys: str | list[str])
    - Parse strings like "ctrl+shift+p" into pynput keys.
    - Press/release modifiers then main key.
- Optional new named actions for higher‑level behaviors:
  - start_draw_mode, toggle_left_click, quick_menu, etc., implemented with the same velocity‑sensitivity pattern (performvelocityaction) where needed.[2][1]

## Dispatcher data structures

On config load (or reload), build lookup maps from the action_map list:

```python
bimanual_map = {}  # (left, right) -> entry
left_map = {}      # left -> entry
right_map = {}     # right -> entry

for entry in action_map_list:
    left = entry["left"]
    right = entry["right"]
    if left != "none" and right != "none":
        bimanual_map[(left, right)] = entry
    elif left != "none":
        left_map[left] = entry
    elif right != "none":
        right_map[right] = entry
```

- Optionally also compute a composite string ID for debugging:
  - combo_id = f"left_{left}&right_{right}"
  - Use only for dup checks and logging, not as JSON keys.[3][2]

## Dispatcher runtime logic

Create a central dispatch function (in a new module like action_dispatcher.py or inside HANDSApp):

```python
def execute_entry(systemctrl, entry, metadata):
    if entry["type"] == "key":
        systemctrl.execute_key_combo(entry["keys"])
    elif entry["type"] == "function":
        func = getattr(systemctrl, entry["name"], None)
        if func is None:
            return
        # Decide what metadata to pass (e.g. velocitynorm, cursor position)
        func(**build_params_for(func, metadata))
```

Then the gesture dispatcher with tiered priority:

```python
def dispatch(systemctrl, left_gesture, right_gesture, metadata):
    # Tier 1: bimanual
    if left_gesture and right_gesture:
        entry = bimanual_map.get((left_gesture, right_gesture))
        if entry:
            execute_entry(systemctrl, entry, metadata)
            return

    # Tier 2: left-only
    if left_gesture:
        entry = left_map.get(left_gesture)
        if entry:
            execute_entry(systemctrl, entry, metadata)

    # Tier 3: right-only
    if right_gesture:
        entry = right_map.get(right_gesture)
        if entry:
            execute_entry(systemctrl, entry, metadata)
```

- build_params_for(...) is where you map metadata to function parameters:
  - movecursor gets cursor position + precision flag.
  - zoom / scroll / thumbs actions get velocitynorm.[1][2]

## HANDSApp integration

In handsapp.py, you already have the pipeline:

- Capture frame → detect hands → ComprehensiveGestureManager.process → per-hand and bimanual gesture results.[2]
- Currently, processgestures has hard-coded logic like “if pointing in rightgestures: movecursor(...)”.[2]
- Replace that with:
  - Choose “primary” gesture name per hand:
    - right_name = highest priority detected right gesture (e.g., pointing, zoomin, swipeup).
    - left_name = same for left.
  - Respect gesturesenabled:
    - If a gesture is disabled in config, treat it as absent for dispatching, but still send its name + disabled flag into the status indicator.[4][3]
  - Build metadata (positions, velocitynorm, etc.) from gesture metadata fields you already pass to SystemController today.[1][2]
  - Call dispatch(systemctrl, left_name_or_None, right_name_or_None, metadata).

This removes all hard-coded gesture→action wiring from the main loop and delegates decisions to the config + dispatcher.[3][1][2]

## Gesture name discovery

To keep things future-proof and avoid hard-coding “22 gestures”:

- In ConfigManager or config_gui.py:
  - Source gesture names from:
    - gesturesenabled keys in config.json: pointing, pinch, zoomin, zoomout, swipeleft, swiperight, swipeup, swipedown, thumbsup, thumbsdown, thumbsupmovingup, thumbsupmovingdown, etc.[3]
    - Or optionally from a small registry in gesturedetectors.py later (if you add one).
- Build a sorted list of gesture names and then append "none".
- Use this list to populate left/right dropdowns in GUI.[2][3]

## config_gui.py UI design

Extend config_gui.py to manage action_map as a dynamic table-like editor:[3][2]
We will add a new tab in config_gui.py to manage action_map.

- **Row structure**
  - Left gesture dropdown: all gestures + "none".
  - Right gesture dropdown: all gestures + "none".
  - Type dropdown: "key" | "function".
  - Dynamic field:
    - If "key": QLineEdit for "ctrl+shift+p".
    - If "function": dropdown for available SystemController actions.

- **Available function list**
  - decorate SystemController methods with @exposed_action and introspect. And need to make it extendable and flexible so any new modular functions can be added easily.

- **Dynamic rows behaviour**
  - Start with 1 empty row.
  - When user completes a valid row (both gestures chosen, type set, and key or function selected):
    - Add it to in-memory action_map list.
    - Auto-add a new empty row below.
  - Allow deleting rows to remove mappings.

- **Duplicate checks**
  - For each row, build combo_id = f"left_{left}&right_{right}".
  - Maintain a set of combo_ids; if a new row tries to reuse an existing combo_id:
    - Show error / highlight row, do not allow Save.
  - Optional rules:
    - If "type == key": optionally prevent duplicate keys across rows (or just warn).
    - Functions can be reused across multiple combos; usually no restriction needed.

- **Save / load**
  - On load:
    - Read action_map from config.json (list of mappings).
    - Populate the UI rows from these entries + one extra empty row.
  - On Save:
    - Validate all rows (no duplicate combos, required fields filled).
    - Write the list back under "action_map" in config.json.
    - Rely on existing auto-reload every fpsupdateinterval frames.[3]

## Bimanual priority and scaling

- Priority:
  - Bimanual mapping always wins if a match exists.
  - If no bimanual match, left-only then right-only.
- Scaling:
  - You never generate all combinations; you just look up the ones defined by the user.
  - The maximum logical space is:
    - G gestures per hand → up to G×G bimanual combos + 2G single-hand mappings.
  - Runtime remains O(1) per frame due to dict lookups.[2][3]

## Visual feedback and status indicator

- Keep existing indicator behavior:[5][4]
  - It receives gesture name and disabled flag per hand.
  - It displays the current gesture, with a red dot when disabled.
- The dispatcher integration does not change this:
  - HANDSApp still builds handsdata from the same gesture names and gesturesenabled flags.
  - Only the “what action to execute” logic moves into the dispatcher, driven by action_map.[4][2][3]

## Implementation order (practical steps)

1. **Add execute_key_combo and any missing modular actions in SystemController.**[1]
2. **Define action_map list in config.json and update config_documentation.md to describe it.**[2][3]
3. **Implement an ActionMapManager / dispatcher that:**
   - Reads action_map from ConfigManager.
   - Builds bimanual_map, left_map, right_map.
   - Provides dispatch(systemctrl, left, right, metadata).[1][3][2]
4. **Refactor HANDSApp.processgestures to use the dispatcher instead of hard-coded calls.**[2]
5. **Extend config_gui.py with the dynamic rows UI and duplicate-check logic for action_map.**[3][2]
6. **Test step-by-step:**
   - Single-hand mappings only (cursor, scroll, zoom).
   - Then add a couple of bimanual combos.
   - Test both function and key actions.
   - Verify disabled gestures still just show in UI without triggering actions.[5][4][1][3][2]

This plan matches your goal: you write a small, generic dispatcher and GUI once, and users can fill anywhere from 0 to 528+(or any number of combinations) combinations without you touching the core Python code again.