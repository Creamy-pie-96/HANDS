import unittest
from unittest.mock import MagicMock, create_autospec
import sys
import os

# Add path to source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source_code.app.action_dispatcher import ActionDispatcher
from source_code.utils.system_controller import SystemController

class TestAtomicIntegration(unittest.TestCase):
    def setUp(self):
        # We want to test the REAL SystemController signature, not just a random mock.
        # So we use create_autospec on the class.
        self.mock_sys_ctrl = create_autospec(SystemController, instance=True)
        # Ensure the attribute exists for inspection (create_autospec does this)
        
        self.dispatcher = ActionDispatcher(self.mock_sys_ctrl)

    def test_next_workspace_trigger(self):
        """Verify next_workspace is called with velocity_norm."""
        # Setup map
        map_list = [
            {"left": "none", "right": "swipe_right", "type": "function", "name": "next_workspace"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Dispatch
        meta = {"velocity_norm": 2.5}
        self.dispatcher.dispatch("none", "swipe_right", meta)
        
        # Check call
        self.mock_sys_ctrl.next_workspace.assert_called_with(velocity_norm=2.5)

    def test_move_cursor_trigger(self):
        """Verify move_cursor is called with coordinates."""
        # Setup map
        map_list = [
            {"left": "none", "right": "pointing", "type": "function", "name": "move_cursor"}
        ]
        self.dispatcher.load_map(map_list)
        
        # Dispatch
        meta = {"cursor_pos": (0.1, 0.9)}
        self.dispatcher.dispatch("none", "pointing", meta)
        
        # Check call
        self.mock_sys_ctrl.move_cursor.assert_called_with(norm_x=0.1, norm_y=0.9)

    def test_left_click_trigger(self):
        """Verify left_click works (no args)."""
        map_list = [
            {"left": "none", "right": "pinch", "type": "function", "name": "left_click"}
        ]
        self.dispatcher.load_map(map_list)
        
        self.dispatcher.dispatch("none", "pinch", {})
        self.mock_sys_ctrl.left_click.assert_called()

if __name__ == '__main__':
    unittest.main()
