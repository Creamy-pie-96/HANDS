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
        # Clear previous UI if at root
        if path == "":
            for widget in parent.winfo_children():
                widget.destroy()
            self.entries = {}
            
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                # Section header
                frame = ttk.LabelFrame(parent, text=key, padding=5)
                frame.pack(fill=tk.X, padx=5, pady=5, anchor="nw")
                self.build_ui(value, frame, current_path)
            else:
                # Value entry
                row = ttk.Frame(parent)
                row.pack(fill=tk.X, pady=2)
                
                ttk.Label(row, text=key, width=25).pack(side=tk.LEFT)
                
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    ttk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
                    self.entries[current_path] = (var, bool)
                else:
                    var = tk.StringVar(value=str(value))
                    ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
                    # Infer type
                    dtype = type(value)
                    self.entries[current_path] = (var, dtype)

    def save_config(self):
        try:
            new_data = self.config_data.copy()
            for path, (var, dtype) in self.entries.items():
                keys = path.split('.')
                current = new_data
                for key in keys[:-1]:
                    current = current[key]
                
                val = var.get()
                target_key = keys[-1]
                
                if dtype == bool:
                    current[target_key] = bool(val)
                elif dtype == int:
                    current[target_key] = int(val)
                elif dtype == float:
                    current[target_key] = float(val)
                elif dtype == list:
                     if isinstance(val, str) and val.strip().startswith('['):
                         try:
                             current[target_key] = ast.literal_eval(val)
                         except:
                             current[target_key] = val # Fallback
                     else:
                         current[target_key] = val
                else:
                    current[target_key] = val
            
            with open(self.config_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            
            print("Config saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
            print(e)

if __name__ == "__main__":
    app = ConfigEditor()
    app.mainloop()
