#!/usr/bin/env python3
"""
HANDS - Hand Assisted Navigation and Device System
Main Application

Real-world gesture control application with system integration.
Controls your computer using hand gestures captured by webcam.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

# HANDS modules
from bimanual_gestures import ComprehensiveGestureManager
from system_controller import SystemController
from visual_feedback import VisualFeedback
from config_manager import config

# MediaPipe setup
mp_hands = mp.solutions.hands


class HANDSApplication:
    """Main HANDS application controller."""
    
    def __init__(self, camera_idx=0, enable_system_control=True):
        """
        Initialize HANDS application.
        
        Args:
            camera_idx: Camera device index
            enable_system_control: If True, actually control the system. If False, dry-run mode.
        """
        print("\n" + "="*60)
        print("HANDS - Hand Assisted Navigation and Device System")
        print("="*60 + "\n")
        
        # Load configuration
        print("Loading configuration...")
        self.config = config
        self.config_path = config._config_path
        try:
            self.last_config_mtime = os.path.getmtime(self.config_path)
        except Exception:
            self.last_config_mtime = 0

        
        # Initialize camera
        camera_width = config.get('camera', 'width', default=640)
        camera_height = config.get('camera', 'height', default=480)
        camera_fps = config.get('camera', 'fps', default=60)
        
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {camera_idx}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        # Initialize MediaPipe
        detection_conf = config.get('performance', 'min_detection_confidence', default=0.7)
        tracking_conf = config.get('performance', 'min_tracking_confidence', default=0.3)
        max_hands = config.get('performance', 'max_hands', default=2)
        
        self.hands = mp_hands.Hands(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            max_num_hands=max_hands
        )
        print(f"✓ MediaPipe Hands initialized (max_hands={max_hands})")
        
        # Initialize gesture manager
        self.gesture_mgr = ComprehensiveGestureManager()
        print(f"✓ Gesture manager initialized")
        
        # Initialize system controller
        self.enable_system_control = enable_system_control
        if enable_system_control:
            try:
                self.system_ctrl = SystemController(config)
                print(f"✓ System controller initialized")
            except Exception as e:
                print(f"⚠ System controller failed: {e}")
                print("  Running in visualization-only mode")
                self.enable_system_control = False
                self.system_ctrl = None
        else:
            print("  Running in DRY-RUN mode (no system control)")
            self.system_ctrl = None
        
        # Initialize visual feedback
        self.visual = VisualFeedback(config)
        print(f"✓ Visual feedback initialized")
        
        # Application state
        self.running = True
        self.paused = False
        self.show_debug = config.get('performance', 'show_debug_info', default=False)
        self.show_fps = config.get('performance', 'show_fps', default=True)
        
        # Statistics
        self.frame_count = 0
        self.fps_time = time.time()
        self.fps = 0.0
        
        # Current cursor position (from pointing gesture)
        self.cursor_pos = None
        
        print("\n✓ HANDS application ready!\n")
    
    def process_gestures(self, all_gestures):
        """
        Process detected gestures and trigger system actions.
        
        Args:
            all_gestures: Dict with 'left', 'right', 'bimanual' gesture results
        """
        if self.paused or not self.enable_system_control:
            return
        
        # Get right hand gestures (primary control)
        right_gestures = all_gestures.get('right', {})
        left_gestures = all_gestures.get('left', {})
        bimanual = all_gestures.get('bimanual', {})
        
        # Priority: Bimanual > Right hand > Left hand
        
        # BIMANUAL GESTURES (highest priority)
        if 'precision_cursor' in bimanual:
            # Precision cursor mode with damping
            data = bimanual['precision_cursor'].metadata
            cursor_pos = data.get('cursor_pos')
            if cursor_pos:
                self.cursor_pos = cursor_pos
                self.system_ctrl.move_cursor(cursor_pos[0], cursor_pos[1], precision_mode=True)
            return
        
        if 'pan' in bimanual:
            # Pan/scroll gesture
            data = bimanual['pan'].metadata
            velocity = data.get('velocity', (0, 0))
            # Convert to scroll
            scroll_x = int(velocity[0] * self.system_ctrl.scroll_sensitivity)
            scroll_y = int(velocity[1] * self.system_ctrl.scroll_sensitivity)
            self.system_ctrl.scroll(scroll_x, -scroll_y)  # Invert Y for natural scrolling
            return
        
        # SINGLE HAND GESTURES
        
        # Pointing: Move cursor
        if 'pointing' in right_gestures:
            data = right_gestures['pointing'].metadata
            cursor_pos = data.get('tip_position')
            if cursor_pos:
                self.cursor_pos = cursor_pos
                self.system_ctrl.move_cursor(cursor_pos[0], cursor_pos[1])
        
        # Pinch: Click/drag
        if 'pinch' in right_gestures:
            self.system_ctrl.handle_pinch_gesture(True)
        else:
            self.system_ctrl.handle_pinch_gesture(False)
        
        # Zoom: System zoom
        if 'zoom' in right_gestures:
            data = right_gestures['zoom'].metadata
            zoom_type = data.get('zoom_type')
            if zoom_type == 'in':
                self.system_ctrl.zoom(zoom_in=True)
            elif zoom_type == 'out':
                self.system_ctrl.zoom(zoom_in=False)
        
        # Swipe: Scroll or workspace switch
        if 'swipe' in right_gestures:
            data = right_gestures['swipe'].metadata
            direction = data.get('direction')
            
            if direction in ['up', 'down']:
                # Scroll
                scroll_amount = self.system_ctrl.scroll_sensitivity
                if direction == 'up':
                    self.system_ctrl.scroll(0, scroll_amount)
                else:
                    self.system_ctrl.scroll(0, -scroll_amount)
            elif direction in ['left', 'right']:
                # Workspace switch
                self.system_ctrl.workspace_switch(direction)
        
        # Open hand: Pause/unpause
        if 'open_hand' in right_gestures or 'open_hand' in left_gestures:
            # Toggle pause on open hand
            # Note: This is checked once per detection to avoid rapid toggling
            pass  # Handled in keyboard input 'p'
    
    def run(self):
        """Main application loop."""
        
        self.print_controls()
        
        cv2.namedWindow("HANDS Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HANDS Control", 1280, 720)
        
        try:
            while self.running:
                self.frame_count += 1
                ret, frame_bgr = self.cap.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip for mirror effect
                frame_bgr = cv2.flip(frame_bgr, 1)
                h, w = frame_bgr.shape[:2]
                
                # Compute FPS
                if self.frame_count % 30 == 0:
                    now = time.time()
                    self.fps = 30.0 / (now - self.fps_time)
                    self.fps_time = now
                
                    self.fps_time = now
                
                # Check for config updates (every 30 frames approx)
                if self.frame_count % 30 == 0:
                    try:
                        current_mtime = os.path.getmtime(self.config_path)
                        if current_mtime != self.last_config_mtime:
                            print("\n🔄 Config change detected, reloading...")
                            self.last_config_mtime = current_mtime
                            self.config.reload()
                            
                            # Re-initialize components with new config
                            self.gesture_mgr = ComprehensiveGestureManager()
                            if self.enable_system_control:
                                self.system_ctrl = SystemController(self.config)
                            self.visual = VisualFeedback(self.config)
                            print("✓ Components reloaded with new configuration\n")
                    except Exception as e:
                        print(f"⚠ Error checking config update: {e}")

                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                # Separate hands
                left_landmarks = None
                right_landmarks = None
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label.lower()
                        if hand_label == 'left':
                            left_landmarks = hand_landmarks
                        else:
                            right_landmarks = hand_landmarks
                
                # Detect gestures
                all_gestures = self.gesture_mgr.process_hands(
                    left_landmarks,
                    right_landmarks,
                    frame_bgr.shape
                )
                
                # Process gestures for system control
                self.process_gestures(all_gestures)
                
                # Get hand metrics for visualization
                left_metrics = None
                right_metrics = None
                
                if left_landmarks and self.gesture_mgr.single_hand_mgr.history['left']:
                    left_metrics = self.gesture_mgr.single_hand_mgr.history['left'][-1]
                
                if right_landmarks and self.gesture_mgr.single_hand_mgr.history['right']:
                    right_metrics = self.gesture_mgr.single_hand_mgr.history['right'][-1]
                
                # VISUAL FEEDBACK
                
                # Draw hand overlays
                if left_metrics:
                    self.visual.draw_hand_overlay(frame_bgr, left_metrics, 'left', all_gestures.get('left'))
                
                if right_metrics:
                    self.visual.draw_hand_overlay(frame_bgr, right_metrics, 'right', all_gestures.get('right'))
                
                # Draw cursor preview
                if self.cursor_pos:
                    self.visual.draw_cursor_preview(frame_bgr, self.cursor_pos[0], self.cursor_pos[1], 
                                                   active=not self.paused)
                
                # Draw gesture panel
                status = "PAUSED" if self.paused else ("DRY-RUN" if not self.enable_system_control else "ACTIVE")
                self.visual.draw_gesture_panel(frame_bgr, all_gestures, status_text=status)
                
                # Draw FPS and debug info
                if self.show_fps:
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(frame_bgr, fps_text, (w - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Debug: Print detected gestures to terminal
                if self.show_debug:
                    for category, gestures in all_gestures.items():
                        for gesture_name, result in gestures.items():
                            meta_str = ""
                            if 'zoom_type' in result.metadata:
                                meta_str = f" ({result.metadata['zoom_type']})"
                            elif 'direction' in result.metadata:
                                meta_str = f" ({result.metadata['direction']})"
                            print(f"[{category}] {gesture_name.upper()}{meta_str}")
                
                # Display frame
                cv2.imshow("HANDS Control", frame_bgr)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('p'):
                    self.paused = not self.paused
                    if self.system_ctrl:
                        self.system_ctrl.toggle_pause()
                    print(f"{'⏸ PAUSED' if self.paused else '▶ RESUMED'}")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('h'):
                    self.print_controls()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        if self.hands:
            self.hands.close()
        
        cv2.destroyAllWindows()
        
        print("✓ HANDS application stopped\n")
    
    def print_controls(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("  Q - Quit application")
        print("  P - Pause/Resume gesture control")
        print("  D - Toggle debug info")
        print("  F - Toggle FPS display")
        print("  H - Show this help")
        print("\n" + "="*60)
        print("GESTURE CONTROLS")
        print("="*60)
        print("  👆 Pointing (index finger) - Move cursor")
        print("  🤏 Pinch (thumb+index) - Click / Drag")
        print("  🤌 Zoom (3 fingers) - System zoom in/out")
        print("  👋 Swipe (4 fingers) - Scroll / Switch workspace")
        print("  ✋ Open hand (5 fingers) - (Reserved)")
        print("\n  TWO-HAND GESTURES:")
        print("  Left still + Right move - Pan/Scroll")
        print("  Left still + Right point - Precision cursor")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HANDS - Hand Assisted Navigation and Device System"
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run without system control (visualization only)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config.json (default: ./config.json)'
    )
    
    args = parser.parse_args()
    
    # Load custom config if specified
    if args.config:
        from config_manager import Config
        Config(args.config)
    
    # Create and run application
    try:
        app = HANDSApplication(
            camera_idx=args.camera,
            enable_system_control=not args.dry_run
        )
        app.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
