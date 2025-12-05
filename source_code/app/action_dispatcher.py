"""
Action Dispatcher for HANDS

This module is the "Brain" of the operation. It decouples gesture detection from system control.
It receives detected gestures (Left + Right), looks up the user's preferred action in the
Configured Action Map, and tells the SystemController to execute it.

Centralizing this logic allows for:
1. Fully user-configurable keybindings.
2. Complex Bimanual combinations (Left Fist + Right Open = Action).
3. O(1) performance using Hash Maps.
"""

from typing import List, Dict, Any, Optional, Tuple
import time

class ActionDispatcher:
    def __init__(self, system_controller):
        """
        Initialize the dispatcher.
        
        Args:
            system_controller: Instance of SystemController to execute actions.
        """
        self.sys_ctrl = system_controller
        
        # O(1) Lookup Maps
        # Keys are (left_gesture, right_gesture) tuple
        self.bimanual_map: Dict[Tuple[str, str], Dict] = {}
        
        # Keys are gesture_name string
        self.left_map: Dict[str, Dict] = {}
        self.right_map: Dict[str, Dict] = {}
        
        # Cache for map-building statistics
        self.mapping_stats = {"bimanual": 0, "left": 0, "right": 0}

    def load_map(self, action_map_list: List[Dict]):
        """
        Build optimization maps from the raw configuration list.
        
        Args:
            action_map_list: List of dicts, each containing:
                             {"left": "...", "right": "...", "type": "...", ...}
        """
        self.bimanual_map.clear()
        self.left_map.clear()
        self.right_map.clear()
        
        if not action_map_list:
            return

        for entry in action_map_list:
            left = entry.get("left", "none")
            right = entry.get("right", "none")
            
            # Skip invalid entries
            if left == "none" and right == "none":
                continue
                
            # Bimanual Entry
            if left != "none" and right != "none":
                self.bimanual_map[(left, right)] = entry
                
            # Single Hand Entry
            elif left != "none":
                self.left_map[left] = entry
            elif right != "none":
                self.right_map[right] = entry
                
        self.mapping_stats = {
            "bimanual": len(self.bimanual_map),
            "left": len(self.left_map),
            "right": len(self.right_map)
        }
        print(f"✓ Action Dispatcher loaded: {self.mapping_stats['bimanual']} bimanual, "
              f"{self.mapping_stats['left']} left, {self.mapping_stats['right']} right mappings.")

    def dispatch(self, left_gesture: str, right_gesture: str, metadata: Dict[str, Any]):
        
        """
        Decide which action to take based on current gestures.
        
        Priority:
        1. Bimanual Combo (Left + Right)
        2. Single Hand (Left or Right) - usually mutually exclusive in usage, 
           but if both present, we can execute both or prioritize one.
           Current design: Independent execution if no bimanual match.
        
        Args:
            left_gesture: Name of left gesture (e.g. "fist"), or "none"
            right_gesture: Name of right gesture, or "none"
            metadata: Shared metadata (cursor pos, velocity, etc.)
        """
        # 1. Check Bimanual (Highest Priority)
        # Only if both hands are present
        if left_gesture != "none" and right_gesture != "none":
            combo_key = (left_gesture, right_gesture)
            if combo_key in self.bimanual_map:
                print(f"  > Bimanual Match: {combo_key} -> {self.bimanual_map[combo_key]['name']}")
                try:
                    self._execute_entry(self.bimanual_map[combo_key], metadata)
                except Exception as e:
                    print(f"⚠ Error executing bimanual action: {e}")
                return # Exclusive: Don't do single hand actions if bimanual matched

        # 2. Check Single Hands (Independent)
        # We can execute both left and right actions if they don't conflict
        
        if left_gesture != "none" and left_gesture in self.left_map:
            # print(f"  > Left Match: {left_gesture} -> {self.left_map[left_gesture]['name']}") # Noise reduction
            try:
                self._execute_entry(self.left_map[left_gesture], metadata)
            except Exception as e:
                print(f"⚠ Error executing left action: {e}")
                
        if right_gesture != "none" and right_gesture in self.right_map:
            print(f"  > Right Match: {right_gesture} -> {self.right_map[right_gesture]['name']}")
            try:
                self._execute_entry(self.right_map[right_gesture], metadata)
            except Exception as e:
                print(f"⚠ Error executing right action: {e}")

    def _execute_entry(self, entry: Dict, metadata: Dict):
        """Execute a single action entry."""
        action_type = entry.get("type")
        
        if action_type == "key":
            keys = entry.get("keys", "")
            if keys:
                self.sys_ctrl.execute_key_combo(keys)
                
        elif action_type == "function":
            func_name = entry.get("name")
            if not func_name:
                return
                
            func = getattr(self.sys_ctrl, func_name, None)
            if not func:
                # Silent fail for optional features or log warning
                # print(f"⚠ Unknown function: {func_name}")
                return
                
            # Build arguments dynamically based on what the function accepts
            import inspect
            try:
                sig = inspect.signature(func)
                kwargs = {}
                
                # 1. Velocity (for scroll, zoom, volume, etc.)
                if "velocity_norm" in sig.parameters:
                    kwargs["velocity_norm"] = metadata.get("velocity_norm", 1.0)
                    
                # 2. Cursor Position (for move_cursor)
                if "norm_x" in sig.parameters and "norm_y" in sig.parameters:
                    pos = metadata.get("cursor_pos", (0.5, 0.5))
                    kwargs["norm_x"] = pos[0]
                    kwargs["norm_y"] = pos[1]
                
                # 3. Precision Mode
                if "precision_mode" in sig.parameters and "precision" in entry:
                    kwargs["precision_mode"] = entry["precision"]
                    
                # 4. Explicit Args from Config (if any)
                if "args" in entry and isinstance(entry["args"], dict):
                    kwargs.update(entry["args"])
                    
                # Call the function
                func(**kwargs)
                
            except Exception as e:
                print(f"⚠ Error executing {func_name}: {e}")
