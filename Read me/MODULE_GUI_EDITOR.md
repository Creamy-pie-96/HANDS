# GUI Editor Module

## Overview
The **Config GUI** (`config_gui.py`) allows users to modify the system's `config.json` without touching text files. It has been extended with an "Action Map" editor.

## Key Features
*   **Dynamic Tabs**: Uses `ttk.Notebook` to separate General Settings from Action Mappings.
*   **Introspection**: It scans `SystemController` for methods decorated with `@exposed_action` and automatically lists them in the dropdown.
*   **Key Recording**: Flattens the learning curve by allowing users to press keys to record binding strings (e.g., "Ctrl+S").

## Code Highlight: Introspection
```python
actions = []
for name in dir(SystemController):
    attr = getattr(SystemController, name)
    if getattr(attr, "_is_exposed_action", False):
        actions.append(name)
```
This loop "looks inside" the controller class to find valid actions. This means if a developer adds a new function `do_backflip` and decorates it, the GUI *immediately* supports it without any GUI code changes!

## Educational Concept: Decorators
We use a custom decorator `@exposed_action` to mark safe functions.
1.  **Decorator**: A function that wraps another function.
2.  **Metadata**: We attach `_is_exposed_action = True` to the wrapper.
3.  **Discovery**: Other parts of the code (like the GUI) check for this flag.
