#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import ast

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from source_code.utils.system_controller import SystemController

class ActionMapEditor(ttk.Frame):
    def __init__(self, parent, config_data):
        super().__init__(parent)
        self.config_data = config_data
        self.rows = []
        self.gesture_options = sorted(self._get_gestures())
        self.action_options = self._get_actions()
        
        self.create_widgets()
        self.populate_table()

    def _get_gestures(self):
        # Get gestures from config or hardcoded fallback
        gestures = ["none", "pointing", "pinch", "fist", "open_hand"]
        enabled = self.config_data.get("gestures_enabled", {})
        if enabled:
            gestures = list(enabled.keys())
            if "none" not in gestures:
                gestures.append("none")
        return gestures

    def _get_actions(self):
        # Introspect SystemController for @exposed_action
        actions = []
        for name in dir(SystemController):
            attr = getattr(SystemController, name)
            if callable(attr) and getattr(attr, "_is_exposed_action", False):
                actions.append(name)
        return sorted(actions)

    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(header_frame, text="Left Hand", width=20, font='bold').grid(row=0, column=0, padx=2)
        ttk.Label(header_frame, text="Right Hand", width=20, font='bold').grid(row=0, column=1, padx=2)
        ttk.Label(header_frame, text="Type", width=12, font='bold').grid(row=0, column=2, padx=2)
        ttk.Label(header_frame, text="Action / Key", width=30, font='bold').grid(row=0, column=3, padx=2)
        ttk.Label(header_frame, text="", width=5).grid(row=0, column=4, padx=2) # Spacer for delete button

        # Scrollable container for rows
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.row_container = ttk.Frame(canvas)

        self.row_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.row_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")

        # Add Button
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="+ Add Mapping", command=self.add_row).pack(side=tk.LEFT)

    def populate_table(self):
        action_map = self.config_data.get("action_map", [])
        for entry in action_map:
            self.add_row(entry)
        
        # Add one empty row if empty
        if not action_map:
            self.add_row()

    def add_row(self, entry=None):
        if entry is None:
            entry = {"left": "none", "right": "none", "type": "function", "name": ""}
            
        row_frame = ttk.Frame(self.row_container)
        row_frame.pack(fill=tk.X, pady=2)
        # Let the action column expand so the Action/Key widget can grow
        row_frame.columnconfigure(0, weight=0)
        row_frame.columnconfigure(1, weight=0)
        row_frame.columnconfigure(2, weight=0)
        row_frame.columnconfigure(3, weight=1)
        row_frame.columnconfigure(4, weight=0)
        
        # Left Gesture
        left_var = tk.StringVar(value=entry.get("left", "none"))
        left_cb = ttk.Combobox(row_frame, textvariable=left_var, values=self.gesture_options, width=18)
        left_cb.grid(row=0, column=0, padx=2)
        
        # Right Gesture
        right_var = tk.StringVar(value=entry.get("right", "none"))
        right_cb = ttk.Combobox(row_frame, textvariable=right_var, values=self.gesture_options, width=18)
        right_cb.grid(row=0, column=1, padx=2)
        
        # Type
        type_var = tk.StringVar(value=entry.get("type", "function"))
        type_cb = ttk.Combobox(row_frame, textvariable=type_var, values=["function", "key"], width=10)
        type_cb.grid(row=0, column=2, padx=2)
        
        # Action Input (Dynamic based on type)
        # Action Input (Dynamic based on type)
        action_frame = ttk.Frame(row_frame, width=250) # Increased width
        action_frame.grid(row=0, column=3, padx=5, sticky="we")
        action_frame.grid_propagate(False) # Ensure fixed size when using grid
        
        action_var = tk.StringVar()
        action_widget_ref = [None] # Mutable ref

        def on_key_record(event, entry_widget):
            # Ignore modifier keys alone
            if event.keysym in ('Control_L', 'Control_R', 'Shift_L', 'Shift_R', 'Alt_L', 'Alt_R', 'Super_L', 'Super_R'):
                return "break"
                
            # Build key string
            parts = []
            if event.state & 4: parts.append("ctrl") # Control
            if event.state & 1: parts.append("shift") # Shift
            if event.state & 8: parts.append("alt") # Alt
            if event.state & 64: parts.append("cmd") # Super/Command (Linux)
            
            # Map keysym to pynput style strings where possible
            key = event.keysym.lower()
            if key == "return": key = "enter"
            if key == "iso_left_tab": key = "tab" # Fix for Shift+Tab on Linux
            
            if key not in parts: 
                parts.append(key)
                
            combo = "+".join(parts)
            action_var.set(combo)
            return "break" # Stop propagation

        def update_action_widget(*args):
             # Clear old
             if action_widget_ref[0]:
                 action_widget_ref[0].destroy()
             
             t = type_var.get()
             if t == "function":
                 # Function Dropdown
                 current_val = entry.get("name") if entry.get("type") == "function" else ""
                 if not action_var.get() or action_var.get() not in self.action_options:
                      # If switching types, try to keep value if valid, else default
                      action_var.set(current_val if current_val in self.action_options else (self.action_options[0] if self.action_options else ""))
                 
                 w = ttk.Combobox(action_frame, textvariable=action_var, values=self.action_options, width=30)
                 w.pack(fill=tk.BOTH, expand=True) # Fill frame
                 action_widget_ref[0] = w
             else:
                 # Key Recorder
                 current_val = entry.get("keys") if entry.get("type") == "key" else ""
                 if not action_var.get():
                     action_var.set(current_val)
                     
                 w = ttk.Entry(action_frame, textvariable=action_var, width=30)
                 w.pack(fill=tk.BOTH, expand=True)
                 
                 # visual cue
                 if not current_val:
                     w.insert(0, "Click to record keys...") 
                 
                 # Bind events for recording
                 w.bind("<FocusIn>", lambda e: w.selection_range(0, tk.END))
                 w.bind("<KeyPress>", lambda e: on_key_record(e, w))
                 
                 action_widget_ref[0] = w
        
        # Initial set
        t_init = entry.get("type", "function")
        if t_init == "key":
             action_var.set(entry.get("keys", ""))
        else:
             action_var.set(entry.get("name", ""))
             
        update_action_widget()
        type_var.trace("w", update_action_widget)

        # Delete Button
        def delete_row():
            row_frame.destroy()
            self.rows.remove(row_data)
            
        del_btn = ttk.Button(row_frame, text="X", width=3, command=delete_row)
        del_btn.grid(row=0, column=4, padx=5)

        # Store row data for saving
        row_data = {
            "left": left_var,
            "right": right_var,
            "type": type_var,
            "action": action_var
        }
        self.rows.append(row_data)
    
    def get_data(self):
        """Collect list of mappings from UI."""
        result = []
        seen_combos = set()
        seen_actions = set()
        seen_keys = set()

        for row in self.rows:
            left = row["left"].get()
            right = row["right"].get()
            t = row["type"].get()
            act = row["action"].get()

            # Skip incomplete
            if not act or act == "Click to record keys...":
                continue
            
            # Clean up placeholder artifacts if present
            if act.startswith("Click to record keys..."):
                act = act.replace("Click to record keys...", "")
                if not act:
                    continue
            
            if left == "none" and right == "none":
                continue

            # Duplicate gesture combo check
            combo = (left, right)
            if combo in seen_combos:
                messagebox.showwarning("Duplicate Mapping", f"Ignored duplicate mapping for Left:{left} + Right:{right}")
                continue
            seen_combos.add(combo)

            # Duplicate action/key check
            if t == "function":
                if act in seen_actions:
                    messagebox.showwarning("Duplicate Action", f"Ignored duplicate action mapping for function: {act}")
                    continue
                seen_actions.add(act)
            else:
                if act in seen_keys:
                    messagebox.showwarning("Duplicate Key", f"Ignored duplicate key mapping for key: {act}")
                    continue
                seen_keys.add(act)

            entry = {
                "left": left,
                "right": right,
                "type": t
            }
            if t == "function":
                entry["name"] = act
            else:
                entry["keys"] = act

            result.append(entry)
        return result

class ConfigEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HANDS Config Editor")
        self.geometry("600x800")
        
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.config_data = {}
        self.entries = {}
        
        self.create_widgets()
        self.load_config()
        
    def create_widgets(self):
        # Notebook for Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: General Config
        self.general_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.general_frame, text="General Settings")
        
        # Setup Scrollbar for General Frame
        canvas = tk.Canvas(self.general_frame)
        scrollbar = ttk.Scrollbar(self.general_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tab 2 placeholder (will be initialized in load_config)
        self.action_tab = None

        # Bottom Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Save & Apply", command=self.save_config).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Reload from File", command=self.load_config).pack(side=tk.RIGHT, padx=5)
        
    def load_config(self):
        try:
            if not os.path.exists(self.config_path):
                messagebox.showerror("Error", f"Config file not found: {self.config_path}")
                return

            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            self.build_ui(self.config_data, self.scrollable_frame)
            
            # Init Action Map Tab
            if self.action_tab:
                self.action_tab.destroy()
            self.action_tab = ActionMapEditor(self.notebook, self.config_data)
            self.notebook.add(self.action_tab, text="Action Map")
            
            print(f"Loaded config from {self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")

    def build_ui(self, data, parent, path=""):
        if path == "":
            for widget in parent.winfo_children():
                widget.destroy()
            self.entries = {}
            
        for key, value in data.items():
            if key in ["description", "action_map"]:
                continue
                
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                frame = ttk.LabelFrame(parent, text=key, padding=5)
                frame.pack(fill=tk.X, padx=5, pady=5, anchor="nw")
                self.build_ui(value, frame, current_path)
            else:
                # Handle [value, description] format
                actual_value = value
                description = ""
                
                if isinstance(value, list) and len(value) >= 1:
                    actual_value = value[0]
                    if len(value) >= 2:
                        description = value[1]
                
                row = ttk.Frame(parent)
                row.pack(fill=tk.X, pady=2)
                
                ttk.Label(row, text=key, width=25).pack(side=tk.LEFT)
                
                if isinstance(actual_value, bool):
                    var = tk.BooleanVar(value=actual_value)
                    ttk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
                    self.entries[current_path] = (var, bool, description)
                else:
                    var = tk.StringVar(value=str(actual_value))
                    entry = ttk.Entry(row, textvariable=var)
                    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    dtype = type(actual_value)
                    self.entries[current_path] = (var, dtype, description)
                
                # Show description as tooltip/label if available
                if description:
                    desc_label = ttk.Label(row, text="ℹ️", foreground="blue", cursor="hand2")
                    desc_label.pack(side=tk.LEFT, padx=5)
                    # Create tooltip
                    self._create_tooltip(desc_label, description)

    def save_config(self):
        try:
            new_data = self.config_data.copy()
            for path, entry_tuple in self.entries.items():
                var, dtype = entry_tuple[0], entry_tuple[1]
                description = entry_tuple[2] if len(entry_tuple) > 2 else ""
                
                keys = path.split('.')
                current = new_data
                for key in keys[:-1]:
                    current = current[key]
                
                val = var.get()
                target_key = keys[-1]
                
                # Convert value to correct type
                if dtype == bool:
                    converted_val = bool(val)
                elif dtype == int:
                    converted_val = int(val)
                elif dtype == float:
                    converted_val = float(val)
                elif dtype == list:
                     if isinstance(val, str) and val.strip().startswith('['):
                         try:
                             converted_val = ast.literal_eval(val)
                         except:
                             converted_val = val # Fallback
                     else:
                         converted_val = val
                else:
                    converted_val = val
                
                # Save in [value, description] format if description exists
                if description:
                    current[target_key] = [converted_val, description]
                else:
                    # Check if original was in [value, desc] format
                    original = current.get(target_key)
                    if isinstance(original, list) and len(original) >= 2:
                        current[target_key] = [converted_val, original[1]]
                    else:
                        current[target_key] = converted_val
            
            # Update action_map from tab
            if self.action_tab:
                new_data["action_map"] = self.action_tab.get_data()

            
            with open(self.config_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            
            print("Config saved")
            messagebox.showinfo("Success", "Configuration saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
            print(e)
    
    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(tooltip, text=text, background="lightyellow", 
                           relief="solid", borderwidth=1, wraplength=300,
                           justify="left", padx=5, pady=5)
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

if __name__ == "__main__":
    app = ConfigEditor()
    app.mainloop()
