#!/usr/bin/env python3
import sys
import os
import json
import ast
import inspect

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                               QPushButton, QCheckBox, QTabWidget, QScrollArea, 
                               QMessageBox, QFrame, QSizePolicy, QGroupBox)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QAction

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from source_code.utils.system_controller import SystemController

class KeySequenceRecorder(QLineEdit):
    """Custom QLineEdit that captures key combinations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Click to record keys...")
        self.setReadOnly(True)  # Prevent manual typing
        self.recording = False

    def focusInEvent(self, event):
        self.recording = True
        self.selectAll()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.recording = False
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if not self.recording:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()
        
        # Ignore standalone modifier presses
        if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            return

        parts = []
        if modifiers & Qt.ControlModifier: parts.append("ctrl")
        if modifiers & Qt.ShiftModifier: parts.append("shift")
        if modifiers & Qt.AltModifier: parts.append("alt")
        if modifiers & Qt.MetaModifier: parts.append("cmd") # Meta is Command on macOS/Super on Linux

        # Mapping some Qt keys to the format expected by the backend
        # Use a more robust way to get key name
        key_name = ""
        
        # 1. Check manual map
        key_map = {
            Qt.Key_Return: "enter",
            Qt.Key_Enter: "enter",
            Qt.Key_Tab: "tab",
            Qt.Key_Backtab: "tab",
            Qt.Key_Space: "space",
            Qt.Key_Backspace: "backspace",
            Qt.Key_Delete: "delete",
            Qt.Key_Escape: "esc",
            Qt.Key_Left: "left",
            Qt.Key_Right: "right",
            Qt.Key_Up: "up",
            Qt.Key_Down: "down",
            Qt.Key_Home: "home",
            Qt.Key_End: "end",
            Qt.Key_PageUp: "pageup",
            Qt.Key_PageDown: "pagedown",
            Qt.Key_F1: "f1", Qt.Key_F2: "f2", Qt.Key_F3: "f3", Qt.Key_F4: "f4",
            Qt.Key_F5: "f5", Qt.Key_F6: "f6", Qt.Key_F7: "f7", Qt.Key_F8: "f8",
            Qt.Key_F9: "f9", Qt.Key_F10: "f10", Qt.Key_F11: "f11", Qt.Key_F12: "f12",
        }
        
        if key in key_map:
            key_name = key_map[key]
        else:
            # 2. Try QKeySequence for standard keys
            # event.text() is unreliable with modifiers (e.g. Ctrl+A = \x01)
            # So we prefer the code from QKeySequence which maps Key_A -> "A"
            ks = QKeySequence(key).toString()
            if ks:
                key_name = ks.lower()
            else:
                # 3. Fallback to text if QKeySequence fails (unlikely for standard keys but possible)
                t = event.text().strip()
                if t:
                    key_name = t.lower()
        
        if key_name and key_name not in parts:
             # Filter out duplicates if logic above somehow produced a modifier name (e.g. if key was Key_Control)
             # But we returned early for standalone modifiers, so this is just for safety
             if key_name not in ("ctrl", "shift", "alt", "cmd", "meta"):
                parts.append(key_name)
        
        if not parts:
             return # Nothing to record
                  
        combo = "+".join(parts)
        self.setText(combo)
        # Clear focus to stop recording
        self.clearFocus()

from PySide6.QtGui import QKeySequence # Import here to ensure availability

class ActionMapRow(QWidget):
    """Represents a single row in the Action Map editor."""
    def __init__(self, parent, gesture_options, action_options, entry=None, delete_callback=None):
        super().__init__(parent)
        self.delete_callback = delete_callback
        self.action_options = action_options
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Left Gesture
        self.left_cb = QComboBox()
        self.left_cb.addItems(gesture_options)
        self.left_cb.setCurrentText(entry.get("left", "none") if entry else "none")
        layout.addWidget(self.left_cb, 1) # Stretch 1
        
        # Right Gesture
        self.right_cb = QComboBox()
        self.right_cb.addItems(gesture_options)
        self.right_cb.setCurrentText(entry.get("right", "none") if entry else "none")
        layout.addWidget(self.right_cb, 1)

        # Type
        self.type_cb = QComboBox()
        self.type_cb.addItems(["function", "key"])
        current_type = entry.get("type", "function") if entry else "function"
        self.type_cb.setCurrentText(current_type)
        layout.addWidget(self.type_cb, 1)
        
        # Action Stack (We switch between ComboBox and LineEdit)
        self.action_container = QWidget()
        self.action_layout = QVBoxLayout(self.action_container)
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.action_container, 2) # Stretch 2 (wider)

        self.function_cb = QComboBox()
        self.function_cb.addItems(action_options)
        
        self.key_input = KeySequenceRecorder()
        
        # Initialize based on current type
        if current_type == "function":
             self.function_cb.setCurrentText(entry.get("name", "") if entry else "")
             self.action_layout.addWidget(self.function_cb)
             self.key_input.setVisible(False)
        else:
             self.key_input.setText(entry.get("keys", "") if entry else "")
             self.action_layout.addWidget(self.key_input)
             self.function_cb.setVisible(False)

        # Connect Type Change
        self.type_cb.currentTextChanged.connect(self.on_type_changed)

        # Delete Button
        self.del_btn = QPushButton("X")
        self.del_btn.setFixedWidth(30)
        self.del_btn.setStyleSheet("color: red; font-weight: bold;")
        self.del_btn.clicked.connect(self.delete_me)
        layout.addWidget(self.del_btn, 0)

    def on_type_changed(self, new_type):
        # Remove current widget
        # Note: We don't delete them, just hide/show or replace
        # Simpler to clear layout and add the right one
        
        # Clear layout items
        while self.action_layout.count():
            item = self.action_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None) # Detach but don't destroy yet if we own it
                # Actually, better to keep ref and just hide/add
        
        if new_type == "function":
            self.action_layout.addWidget(self.function_cb)
            self.function_cb.show()
            self.key_input.hide()
        else:
            self.action_layout.addWidget(self.key_input)
            self.key_input.show()
            self.function_cb.hide()

    def delete_me(self):
        if self.delete_callback:
            self.delete_callback(self)

    def get_data(self):
        left = self.left_cb.currentText()
        right = self.right_cb.currentText()
        t = self.type_cb.currentText()
        
        if t == "function":
            act = self.function_cb.currentText()
        else:
            act = self.key_input.text()
            
        return left, right, t, act

class ActionMapEditor(QWidget):
    def __init__(self, parent, config_data):
        super().__init__(parent)
        self.config_data = config_data
        self.rows = []
        self.gesture_options = sorted(self._get_gestures())
        self.action_options = self._get_actions()

        self.layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Left Hand</b>"), 1)
        header_layout.addWidget(QLabel("<b>Right Hand</b>"), 1)
        header_layout.addWidget(QLabel("<b>Type</b>"), 1)
        header_layout.addWidget(QLabel("<b>Action / Key</b>"), 2)
        header_layout.addWidget(QLabel(""), 0) # Spacer for delete btn
        header_layout.addSpacing(30)
        self.layout.addLayout(header_layout)

        # Scroll Area for Rows
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.row_container = QWidget()
        self.row_layout = QVBoxLayout(self.row_container)
        self.row_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.row_container)
        self.layout.addWidget(self.scroll)

        # Add Button
        self.add_btn = QPushButton("+ Add Mapping")
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)

        self.populate_table()

    def _get_gestures(self):
        gestures = ["none", "pointing", "pinch", "fist", "open_hand"]
        enabled = self.config_data.get("gestures_enabled", {})
        if enabled:
            gestures = list(enabled.keys())
            if "none" not in gestures:
                gestures.append("none")
        return gestures

    def _get_actions(self):
        actions = []
        # Introspect SystemController
        for name in dir(SystemController):
            attr = getattr(SystemController, name)
            if callable(attr) and getattr(attr, "_is_exposed_action", False):
                actions.append(name)
        return sorted(actions)

    def populate_table(self):
        action_map = self.config_data.get("action_map", [])
        for entry in action_map:
            self.add_row(entry)
        
        if not action_map:
            self.add_row()

    def add_row(self, entry=None):
        if entry is None:
            entry = {"left": "none", "right": "none", "type": "function", "name": ""}
        
        row = ActionMapRow(self.row_container, self.gesture_options, self.action_options, entry, self.delete_row)
        self.row_layout.addWidget(row)
        self.rows.append(row)

    def delete_row(self, row_widget):
        self.row_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        if row_widget in self.rows:
            self.rows.remove(row_widget)

    def get_data(self):
        result = []
        seen_combos = set()
        seen_actions = set() # Optional: if we want to enforce unique actions, but user might want same action for different gestures
        seen_keys = set()
        
        for row in self.rows:
            left, right, t, act = row.get_data()
            
            # Validation
            if not act or act == "Click to record keys...":
                continue
                
            if act.startswith("Click to record keys..."):
                 act = act.replace("Click to record keys...", "")
                 if not act: continue
            
            if left == "none" and right == "none":
                continue

            # Duplicate check
            combo = (left, right)
            if combo in seen_combos:
                QMessageBox.warning(self, "Duplicate Mapping", f"Ignored duplicate mapping for Left:{left} + Right:{right}")
                continue
            seen_combos.add(combo)
            
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

class ConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HANDS Config Editor")
        self.resize(800, 800)
        
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.config_data = {}
        # Store references to widgets to retrieve values: {path: (widget, type, description)}
        self.entries = {} 

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # General Settings Tab
        self.general_scroll = QScrollArea()
        self.general_scroll.setWidgetResizable(True)
        self.general_container = QWidget()
        self.general_layout = QVBoxLayout(self.general_container)
        self.general_layout.setAlignment(Qt.AlignTop)
        self.general_scroll.setWidget(self.general_container)
        
        self.tabs.addTab(self.general_scroll, "General Settings")

        # Buttons
        btn_layout = QHBoxLayout()
        reload_btn = QPushButton("Reload from File")
        reload_btn.clicked.connect(self.load_config)
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self.save_config)
        
        btn_layout.addStretch()
        btn_layout.addWidget(reload_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

        self.action_tab = None
        
        # Load
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            QMessageBox.critical(self, "Error", f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            
            # Rebuild General Tab
            self.build_ui(self.config_data, self.general_layout)
            
            # Rebuild Action Tab
            if self.action_tab:
                # Remove old tab
                idx = self.tabs.indexOf(self.action_tab)
                if idx >= 0:
                    self.tabs.removeTab(idx)
                    self.action_tab.deleteLater()
            
            self.action_tab = ActionMapEditor(self, self.config_data)
            self.tabs.addTab(self.action_tab, "Action Map")
            
            print(f"Loaded config from {self.config_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            print(e)

    def build_ui(self, data, parent_layout, path=""):
        if path == "":
            # Clear layout
             while parent_layout.count():
                item = parent_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                # If it's a layout, we might need to be more recursive, but we use widgets mostly
             self.entries = {}

        for key, value in data.items():
            if key in ["description", "action_map"]:
                continue
            
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                group = QGroupBox(key)
                group_layout = QVBoxLayout(group)
                parent_layout.addWidget(group)
                self.build_ui(value, group_layout, current_path)
            else:
                self.add_field(parent_layout, key, value, current_path)
    
    def add_field(self, layout, key, value, path):
        row_layout = QHBoxLayout()
        
        label = QLabel(key)
        label.setFixedWidth(200)
        row_layout.addWidget(label)
        
        # Handle [value, description] format
        actual_value = value
        description = ""
        if isinstance(value, list) and len(value) >= 1:
             # Check if it looks like [val, desc]
             # But be careful, it might just be a list config
             # Heuristic: if len is 2 and second element is string and it's not a list of coordinates or something
             # The previous code assumed if it's a list, check [0] as val and [1] as desc if present
             # BUT what if the config IS a list of values?
             # The previous code had: if isinstance(value, list) ... actual_value = value[0]
             # This implies ALL lists are treated as [val, desc] wrappers?
             # Let's check the previous code logic:
             # if isinstance(value, list) and len(value) >= 1: 
             #    actual_value = value[0] 
             #    if len(value) >= 2: description = value[1]
             # This seems to FORCE lists to be [val, desc]. This might be a limitation of the previous tool.
             # I should respect it to maintain compatibility.
             actual_value = value[0]
             if len(value) >= 2:
                 description = value[1]
        
        widget = None
        dtype = type(actual_value)
        
        if isinstance(actual_value, bool):
            widget = QCheckBox()
            widget.setChecked(actual_value)
            self.entries[path] = (widget, bool, description)
        else:
            widget = QLineEdit()
            widget.setText(str(actual_value))
            self.entries[path] = (widget, dtype, description)
            
        row_layout.addWidget(widget)
        
        if description:
            # Info icon / Tooltip
            info_lbl = QLabel("ℹ️")
            info_lbl.setToolTip(description)
            info_lbl.setStyleSheet("color: blue; cursor: pointer;")
            row_layout.addWidget(info_lbl)
            label.setToolTip(description) # Also set on label
            
        layout.addLayout(row_layout)

    def save_config(self):
        try:
            new_data = self.config_data.copy()
            
            # Update General Settings
            for path, (widget, dtype, desc) in self.entries.items():
                keys = path.split('.')
                current = new_data
                for k in keys[:-1]:
                    current = current[k]
                
                target_key = keys[-1]
                
                # Get value
                if isinstance(widget, QCheckBox):
                    val = widget.isChecked()
                else:
                    val = widget.text()
                
                # Convert
                try:
                    if dtype == bool:
                        converted_val = bool(val)
                    elif dtype == int:
                        converted_val = int(val)
                    elif dtype == float:
                        converted_val = float(val)
                    elif dtype == list:
                         if isinstance(val, str) and val.strip().startswith('['):
                             converted_val = ast.literal_eval(val)
                         else:
                             converted_val = val
                    else:
                        converted_val = val
                except:
                    # Fallback
                    converted_val = val
                
                # Restore [val, desc] format if needed
                # If we had a description, we force the format
                if desc:
                    current[target_key] = [converted_val, desc]
                else:
                    # Check original
                    original = current.get(target_key)
                    if isinstance(original, list) and len(original) >= 2 and isinstance(original[1], str):
                         # It was likely [val, desc], but maybe I didn't detect desc properly or it was empty?
                         # Safest to just write value if no desc detected
                         current[target_key] = converted_val
                    else:
                         current[target_key] = converted_val

            # Update Action Map
            if self.action_tab:
                new_data["action_map"] = self.action_tab.get_data()
            
            # Write
            with open(self.config_path, 'w') as f:
                json.dump(new_data, f, indent=2)
                
            QMessageBox.information(self, "Success", "Configuration saved successfully!")
            print("Config saved.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigEditor()
    window.show()
    sys.exit(app.exec())
