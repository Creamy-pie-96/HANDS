import unittest
from unittest.mock import MagicMock, create_autospec
import sys
import os

# Add path to source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source_code.app.action_dispatcher import ActionDispatcher
from source_code.utils.system_controller import SystemController

class TestActionDispatcher(unittest.TestCase):
    def setUp(self):
        # Mock SystemController with autospec to support introspection
        self.mock_sys_ctrl = create_autospec(SystemController, instance=True)
        self.dispatcher = ActionDispatcher(self.mock_sys_ctrl)

    def test_load_map(self):
        map_list = [
            {"left": "fist", "right": "none", "type": "key", "keys": "a"},
            {"left": "none", "right": "open", "type": "function", "name": "left_click"},
            {"left": "fist", "right": "open", "type": "key", "keys": "b"}
        ]
        self.dispatcher.load_map(map_list)
        
        self.assertIn("fist", self.dispatcher.left_map)
        self.assertIn("open", self.dispatcher.right_map)
        self.assertIn(("fist", "open"), self.dispatcher.bimanual_map)
        
    def test_dispatch_single_hand(self):
        map_list = [
            {"left": "swipe_left", "right": "none", "type": "key", "keys": "left_arrow"},
            {"left": "none", "right": "pinch", "type": "function", "name": "left_click"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Test Left
        self.dispatcher.dispatch("swipe_left", "none", {})
        self.mock_sys_ctrl.execute_key_combo.assert_called_with("left_arrow")
        
        # Test Right
        self.dispatcher.dispatch("none", "pinch", {})
        self.mock_sys_ctrl.left_click.assert_called()

    def test_dispatch_bimanual_override(self):
        map_list = [
            {"left": "fist", "right": "none", "type": "key", "keys": "L"},
            {"left": "none", "right": "fist", "type": "key", "keys": "R"},
            {"left": "fist", "right": "fist", "type": "key", "keys": "BOTH"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Dispatch Both
        self.dispatcher.dispatch("fist", "fist", {})
        
        # Should only call BOTH, not L or R
        self.mock_sys_ctrl.execute_key_combo.assert_called_once_with("BOTH")

    def test_dispatch_metadata_passing(self):
        map_list = [
            {"left": "none", "right": "swipe_up", "type": "function", "name": "scroll_up"}
        ]
        self.dispatcher.load_map(map_list)
        
        meta = {"velocity_norm": 0.8, "direction": "up"}
        self.dispatcher.dispatch("none", "swipe_up", meta)
        
        # Expect scroll_up to be called with velocity_norm (introspected)
        self.mock_sys_ctrl.scroll_up.assert_called_with(velocity_norm=0.8)

    def test_dispatch_move_cursor(self):
        map_list = [
            {"left": "index", "right": "none", "type": "function", "name": "move_cursor"}
        ]
        self.dispatcher.load_map(map_list)
        
        meta = {"cursor_pos": (0.2, 0.3)}
        self.dispatcher.dispatch("index", "none", meta)
        
        self.mock_sys_ctrl.move_cursor.assert_called_with(norm_x=0.2, norm_y=0.3)


if __name__ == '__main__':
    unittest.main()
