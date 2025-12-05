import unittest
from unittest.mock import MagicMock
import sys
import os

# Add path to source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source_code.app.action_dispatcher import ActionDispatcher

class TestActionDispatcher(unittest.TestCase):
    def setUp(self):
        # Mock SystemController
        self.mock_sys_ctrl = MagicMock()
        self.dispatcher = ActionDispatcher(self.mock_sys_ctrl)

    def test_load_map(self):
        map_list = [
            {"left": "fist", "right": "none", "type": "key", "keys": "a"},
            {"left": "none", "right": "open", "type": "function", "name": "foo"},
            {"left": "fist", "right": "open", "type": "key", "keys": "b"}
        ]
        self.dispatcher.load_map(map_list)
        
        self.assertIn("fist", self.dispatcher.left_map)
        self.assertIn("open", self.dispatcher.right_map)
        self.assertIn(("fist", "open"), self.dispatcher.bimanual_map)
        
    def test_dispatch_single_hand(self):
        map_list = [
            {"left": "swipe_left", "right": "none", "type": "key", "keys": "left_arrow"},
            {"left": "none", "right": "pinch", "type": "function", "name": "click"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Test Left
        self.dispatcher.dispatch("swipe_left", "none", {})
        self.mock_sys_ctrl.execute_key_combo.assert_called_with("left_arrow")
        
        # Test Right
        self.dispatcher.dispatch("none", "pinch", {})
        self.mock_sys_ctrl.click.assert_called()

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
            {"left": "none", "right": "swipe_up", "type": "function", "name": "scroll"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Configure Mock sensitivity
        self.mock_sys_ctrl.get_base_sensitivity.return_value = 10
        
        meta = {"velocity_norm": 0.8, "direction": "up"}
        self.dispatcher.dispatch("none", "swipe_up", meta)
        
        # Expect conversion to dy (up -> positive dy in legacy implementation I copied)
        self.mock_sys_ctrl.scroll.assert_called_with(velocity_norm=0.8, dy=10)

if __name__ == '__main__':
    unittest.main()
