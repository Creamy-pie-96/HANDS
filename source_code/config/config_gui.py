#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import ast

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
        # Main container with scrollbar
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
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
            print(f"Loaded config from {self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")

    def build_ui(self, data, parent, path=""):
        if path == "":
            for widget in parent.winfo_children():
                widget.destroy()
            self.entries = {}
            
        for key, value in data.items():
            if key == "description":
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
