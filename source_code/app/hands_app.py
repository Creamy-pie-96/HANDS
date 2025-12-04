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
import urllib.request
import threading
import queue
from pathlib import Path

# HANDS modules
from source_code.detectors.bimanual_gestures import ComprehensiveGestureManager
from source_code.utils.system_controller import SystemController
from source_code.utils.visual_feedback import VisualFeedback
from source_code.config.config_manager import config, is_gesture_enabled

# MediaPipe setup
mp_hands = mp.solutions.hands

# MediaPipe Tasks API for GPU support
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Model URL and local path
HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
HAND_LANDMARKER_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'hand_landmarker.task'


def ensure_model_downloaded():
    """Download the hand landmarker model if not present."""
    model_path = HAND_LANDMARKER_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"📥 Downloading hand landmarker model...")
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, str(model_path))
            print(f"✓ Model downloaded to {model_path}")
        except Exception as e:
            print(f"⚠ Failed to download model: {e}")
            return None
    
    return str(model_path)


class TasksLandmarkPoint:
    """Wrapper to make Tasks API landmark compatible with legacy API."""
    def __init__(self, landmark):
        self.x = landmark.x
        self.y = landmark.y
        self.z = landmark.z if hasattr(landmark, 'z') else 0.0


class TasksLandmarksWrapper:
    """Wrapper to make Tasks API hand_landmarks compatible with legacy API."""
    def __init__(self, hand_landmarks):
        self.landmark = [TasksLandmarkPoint(lm) for lm in hand_landmarks]


class HANDSApplication:
    """Main HANDS application controller."""
    
    def __init__(self, camera_idx=0, enable_system_control=True, status_queue=None):
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
        use_gpu = config.get('performance', 'use_gpu', default=True)
        
        # Store for later reference
        self.use_gpu = False
        self.hands = None
        self.hand_landmarker = None
        self.use_tasks_api = False
        
        # Try to use Tasks API with GPU if enabled
        if use_gpu:
            model_path = ensure_model_downloaded()
            if model_path:
                try:
                    delegate = BaseOptions.Delegate.GPU
                    options = mp_vision.HandLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=model_path,
                            delegate=delegate
                        ),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_hands=max_hands,
                        min_hand_detection_confidence=detection_conf,
                        min_tracking_confidence=tracking_conf
                    )
                    self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                    self.use_tasks_api = True
                    self.use_gpu = True
                    print(f"✓ MediaPipe HandLandmarker initialized with GPU (max_hands={max_hands})")
                except Exception as e:
                    print(f"⚠ GPU initialization failed: {e}")
                    print("  Falling back to CPU...")
                    self.use_gpu = False
        
        # Fallback to CPU (either GPU disabled or failed)
        if not self.use_tasks_api:
            if use_gpu:
                # Try CPU with Tasks API first (still faster than legacy)
                model_path = ensure_model_downloaded()
                if model_path:
                    try:
                        options = mp_vision.HandLandmarkerOptions(
                            base_options=BaseOptions(
                                model_asset_path=model_path,
                                delegate=BaseOptions.Delegate.CPU
                            ),
                            running_mode=mp_vision.RunningMode.IMAGE,
                            num_hands=max_hands,
                            min_hand_detection_confidence=detection_conf,
                            min_tracking_confidence=tracking_conf
                        )
                        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                        self.use_tasks_api = True
                        print(f"✓ MediaPipe HandLandmarker initialized with CPU (max_hands={max_hands})")
                    except Exception as e:
                        print(f"⚠ Tasks API CPU fallback failed: {e}")
            
            # Final fallback to legacy API
            if not self.use_tasks_api:
                self.hands = mp_hands.Hands(
                    min_detection_confidence=detection_conf,
                    min_tracking_confidence=tracking_conf,
                    max_num_hands=max_hands
                )
                print(f"✓ MediaPipe Hands (legacy) initialized (max_hands={max_hands})")
        
        # Initialize gesture manager
        self.gesture_mgr = ComprehensiveGestureManager(config)
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
        
        # Quit gesture state
        self.quit_gesture_start_time = 0
        self.quit_hold_duration = 3.0  # Seconds to hold gesture to quit
        
        # Status Queue for GUI
        self.status_queue = status_queue
        
        # Frame Queue for GUI (Camera View)
        self.frame_queue = None
        if hasattr(self, 'frame_queue_init'):
             self.frame_queue = self.frame_queue_init
        
        # Key Queue for keyboard input from GUI
        self.key_queue = None
        
    def set_frame_queue(self, queue):
        self.frame_queue = queue
    
    def set_key_queue(self, queue):
        self.key_queue = queue
        
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
        if 'precision_cursor' in bimanual and is_gesture_enabled('precision_cursor'):
            # Precision cursor mode with damping
            data = bimanual['precision_cursor'].metadata
            cursor_pos = data.get('cursor_pos')
            if cursor_pos:
                self.cursor_pos = cursor_pos
                self.system_ctrl.move_cursor(cursor_pos[0], cursor_pos[1], precision_mode=True)
            return
        
        if 'pan' in bimanual and is_gesture_enabled('pan'):
            # Pan/scroll gesture
            data = bimanual['pan'].metadata
            velocity = data.get('velocity', (0, 0))
            # Use correct swipe sensitivity based on direction
            # Horizontal: swipe_right for +X, swipe_left for -X
            # Vertical: swipe_down for +Y, swipe_up for -Y
            sens_x = self.system_ctrl.get_base_sensitivity(
                'swipe_right' if velocity[0] >= 0 else 'swipe_left', 3.0)
            sens_y = self.system_ctrl.get_base_sensitivity(
                'swipe_down' if velocity[1] >= 0 else 'swipe_up', 3.0)
            scroll_x = int(velocity[0] * sens_x)
            scroll_y = int(velocity[1] * sens_y)
            # Compute velocity magnitude for modulation
            velocity_norm = float(np.hypot(velocity[0], velocity[1]))
            self.system_ctrl.scroll(scroll_x, -scroll_y, velocity_norm=velocity_norm)  # Invert Y for natural scrolling
            return
        
        # SINGLE HAND GESTURES
        
        # Pointing: Move cursor
        if 'pointing' in right_gestures and is_gesture_enabled('pointing'):
            data = right_gestures['pointing'].metadata
            cursor_pos = data.get('tip_position')
            if cursor_pos:
                self.cursor_pos = cursor_pos
                self.system_ctrl.move_cursor(cursor_pos[0], cursor_pos[1])
        
        # Pinch: Click/drag
        if 'pinch' in right_gestures and is_gesture_enabled('pinch'):
            self.system_ctrl.handle_pinch_gesture(True)
        else:
            self.system_ctrl.handle_pinch_gesture(False)
        
        # Zoom: System zoom
        # Handle both zoom_in and zoom_out
        for gesture_name, gesture_result in right_gestures.items():
            if gesture_name.startswith('zoom_'):
                # Check if this specific zoom direction is enabled
                if not is_gesture_enabled(gesture_name):
                    continue
                data = gesture_result.metadata
                direction = data.get('direction')
                # Get EWMA velocity for modulation
                ewma_vel = abs(data.get('ewma_velocity', 0.0))
                if direction == 'in':
                    self.system_ctrl.zoom(zoom_in=True, velocity_norm=ewma_vel)
                elif direction == 'out':
                    self.system_ctrl.zoom(zoom_in=False, velocity_norm=ewma_vel)
                break
        
        # Swipe: Scroll or workspace switch
        # Handle all swipe directions
        for gesture_name, gesture_result in right_gestures.items():
            if gesture_name.startswith('swipe_'):
                # Check if this specific swipe direction is enabled
                if not is_gesture_enabled(gesture_name):
                    continue
                data = gesture_result.metadata
                direction = data.get('direction')
                # Get EWMA velocity for modulation
                ewma_velocity = data.get('ewma_velocity', (0.0, 0.0))
                velocity_norm = float(np.hypot(ewma_velocity[0], ewma_velocity[1]))
                
                if direction in ['up', 'down']:
                    # Scroll - use swipe_up/swipe_down sensitivity as scroll amount
                    gesture_key = f'swipe_{direction}'
                    scroll_amount = int(self.system_ctrl.get_base_sensitivity(gesture_key, 3.0))
                    if direction == 'up':
                        self.system_ctrl.scroll(0, scroll_amount, velocity_norm=velocity_norm)
                    else:
                        self.system_ctrl.scroll(0, -scroll_amount, velocity_norm=velocity_norm)
                elif direction in ['left', 'right']:
                    # Workspace switch
                    self.system_ctrl.swipe(direction, velocity_norm)
                break
        
        # Thumbs gestures: Volume and Brightness control
        # Handle moving thumbs gestures
        for gesture_name, gesture_result in right_gestures.items():
            if gesture_name.startswith('thumbs_') and 'moving' in gesture_name:
                # Check if this specific thumbs gesture is enabled
                if not is_gesture_enabled(gesture_name):
                    continue
                data = gesture_result.metadata
                # Get velocity for modulation
                velocity = abs(data.get('ewma_velocity', 0.5))
                self.system_ctrl.thumbs_action(gesture_name, velocity_norm=velocity)
                break
        
        # Open hand: Pause/unpause
        if ('open_hand' in right_gestures or 'open_hand' in left_gestures) and is_gesture_enabled('open_hand'):
            # Toggle pause on open hand
            # Note: This is checked once per detection to avoid rapid toggling
            pass  # Handled in keyboard input 'p'
    
    def run(self):
        """Main application loop."""
        
        self.print_controls()
        
        window_width = config.get('display', 'window_width', default=1280)
        window_height = config.get('display', 'window_height', default=720)
        
        visual_mode = config.get('display', 'visual_mode', default='full')
        show_camera = visual_mode in ['full', 'debug']
        
        # Note: We no longer use cv2.imshow directly. 
        # Frames are sent to the GUI thread via frame_queue if show_camera is True.
        
        try:
            while self.running:
                self.frame_count += 1
                ret, frame_bgr = self.cap.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                flip_horizontal = config.get('display', 'flip_horizontal', default=True)
                if flip_horizontal:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                h, w = frame_bgr.shape[:2]
                
                fps_update_interval = config.get('display', 'fps_update_interval', default=30)
                if self.frame_count % fps_update_interval == 0:
                    now = time.time()
                    self.fps = float(fps_update_interval) / (now - self.fps_time)
                    self.fps_time = now
                
                    self.fps_time = now
                
                if self.frame_count % fps_update_interval == 0:
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
                        
                        # Check app_control flags (hot-reloaded)
                        config_pause = self.config.get('app_control', 'pause', default=False)
                        config_exit = self.config.get('app_control', 'exit', default=False)
                        
                        # Handle pause via config
                        if config_pause != self.paused:
                            self.paused = config_pause
                            if self.system_ctrl:
                                self.system_ctrl.toggle_pause()
                            print(f"{'⏸ PAUSED (via config)' if self.paused else '▶ RESUMED (via config)'}")
                        
                        # Handle exit via config
                        if config_exit:
                            print("\n🛑 Exit signal received via config")
                            self.running = False
                            
                    except Exception as e:
                        print(f"⚠ Error checking config update: {e}")

                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Separate hands
                left_landmarks = None
                right_landmarks = None
                
                if self.use_tasks_api:
                    # Use Tasks API (supports GPU)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    results = self.hand_landmarker.detect(mp_image)
                    
                    if results.hand_landmarks and results.handedness:
                        for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                            # Convert Tasks API landmarks to legacy format wrapper
                            hand_label = handedness[0].category_name.lower()
                            # IMPORTANT: Tasks API doesn't account for flipped images
                            # When image is flipped horizontally, we need to swap handedness
                            # to match user's perspective (mirror mode)
                            if flip_horizontal:
                                hand_label = 'right' if hand_label == 'left' else 'left'
                            landmarks_wrapper = TasksLandmarksWrapper(hand_landmarks)
                            if hand_label == 'left':
                                left_landmarks = landmarks_wrapper
                            else:
                                right_landmarks = landmarks_wrapper
                else:
                    # Use legacy API
                    results = self.hands.process(frame_rgb)
                    
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
                    mode_str = "GPU" if self.use_gpu else "CPU"
                    fps_text = f"FPS: {self.fps:.1f} [{mode_str}]"
                    cv2.putText(frame_bgr, fps_text, (w - 200, 30),
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
                
                # Update Status Indicator - Now sends per-hand data
                # Each indicator shows what THAT hand is doing, not bimanual combinations
                if self.status_queue:
                    # Helper function to get gesture name with direction
                    def get_gesture_with_direction(gestures_dict):
                        """Get gesture name with direction suffix if available."""
                        if not gestures_dict:
                            return ""
                        # Skip internal metadata gestures (start with __)
                        for name in gestures_dict:
                            if name.startswith('__'):
                                continue
                            result = gestures_dict[name]
                            # Check for direction in metadata to create specific gesture name
                            if hasattr(result, 'metadata') and result.metadata:
                                direction = result.metadata.get('direction', '')
                                # Only use direction if it's a simple string
                                if isinstance(direction, str) and direction in ('in', 'out', 'up', 'down', 'left', 'right'):
                                    if not name.endswith(f'_{direction}'):
                                        return f"{name}_{direction}"
                            return name
                        return ""
                    
                    def check_gesture_disabled(gesture_name: str) -> bool:
                        """Check if a gesture is disabled in config."""
                        if not gesture_name:
                            return False
                        # For directional gestures like zoom_in, swipe_up, etc.
                        return not is_gesture_enabled(gesture_name)
                    
                    # Build per-hand status data
                    hands_data = {
                        'left': {'detected': False, 'state': 'hidden', 'gesture': '', 'disabled': False},
                        'right': {'detected': False, 'state': 'hidden', 'gesture': '', 'disabled': False}
                    }
                    
                    # Check for special states (paused, dry_run)
                    if not self.enable_system_control:
                        # Dry-run mode - show on right indicator only
                        hands_data['right'] = {'detected': True, 'state': 'red', 'gesture': 'dry_run', 'disabled': False}
                    elif self.paused:
                        # Paused - show on right indicator only
                        hands_data['right'] = {'detected': True, 'state': 'red', 'gesture': 'paused', 'disabled': False}
                    else:
                        # Process left hand - always show what LEFT hand is doing
                        if left_landmarks:
                            hands_data['left']['detected'] = True
                            hands_data['left']['state'] = 'blue'
                            
                            # Get left hand gesture (individual, not bimanual)
                            l_gests = all_gestures.get('left', {})
                            if l_gests:
                                gesture_name = get_gesture_with_direction(l_gests)
                                hands_data['left']['gesture'] = gesture_name
                                hands_data['left']['disabled'] = check_gesture_disabled(gesture_name)
                            
                            # Check for thumbs down quit gesture on left
                            if 'thumbs_down' in l_gests:
                                if self.quit_gesture_start_time == 0:
                                    self.quit_gesture_start_time = time.time()
                                elapsed = time.time() - self.quit_gesture_start_time
                                if elapsed > self.quit_hold_duration:
                                    print("\n🛑 Quit gesture detected! Exiting...")
                                    self.running = False
                                else:
                                    remaining = int(self.quit_hold_duration - elapsed + 0.9)
                                    hands_data['left']['gesture'] = f"exit_{remaining}"
                                    hands_data['left']['state'] = 'red'
                                    hands_data['left']['disabled'] = False  # Quit gesture is always enabled
                        
                        # Process right hand - always show what RIGHT hand is doing
                        if right_landmarks:
                            hands_data['right']['detected'] = True
                            hands_data['right']['state'] = 'blue'
                            
                            # Get right hand gesture (individual, not bimanual)
                            r_gests = all_gestures.get('right', {})
                            if r_gests:
                                gesture_name = get_gesture_with_direction(r_gests)
                                hands_data['right']['gesture'] = gesture_name
                                hands_data['right']['disabled'] = check_gesture_disabled(gesture_name)
                            
                            # Check for thumbs down quit gesture on right
                            if 'thumbs_down' in r_gests:
                                if self.quit_gesture_start_time == 0:
                                    self.quit_gesture_start_time = time.time()
                                elapsed = time.time() - self.quit_gesture_start_time
                                if elapsed > self.quit_hold_duration:
                                    print("\n🛑 Quit gesture detected! Exiting...")
                                    self.running = False
                                else:
                                    remaining = int(self.quit_hold_duration - elapsed + 0.9)
                                    hands_data['right']['gesture'] = f"exit_{remaining}"
                                    hands_data['right']['state'] = 'red'
                                    hands_data['right']['disabled'] = False  # Quit gesture is always enabled
                        
                        # NOTE: Bimanual gestures are detected and used for system control,
                        # but the status indicators show individual hand gestures.
                        # This way: pan = left shows open_hand, right shows swipe_direction
                        
                        # Reset quit timer if no thumbs down
                        is_thumbs_down = 'thumbs_down' in all_gestures.get('right', {}) or \
                                         'thumbs_down' in all_gestures.get('left', {})
                        if not is_thumbs_down:
                            self.quit_gesture_start_time = 0

                    self.status_queue.put(hands_data)

                if show_camera and self.frame_queue:
                    # Resize frame to match display window size
                    display_frame = cv2.resize(frame_bgr, (window_width, window_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Send to GUI thread
                    try:
                        # Keep queue size small by removing old frames if full
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put_nowait(display_frame)
                    except queue.Full:
                        pass
                    
                    # No cv2.waitKey needed here as GUI handles events
                    time.sleep(0.001) # Small sleep to yield
                else:
                    # No window, just sleep briefly to avoid busy loop
                    time.sleep(0.01)
                
                # Process keyboard input from PyQt
                if self.key_queue:
                    try:
                        from PyQt6.QtCore import Qt
                        while True:
                            key = self.key_queue.get_nowait()
                            # Map Qt keys to actions
                            if key == Qt.Key.Key_Q:
                                self.running = False
                            elif key == Qt.Key.Key_P:
                                self.paused = not self.paused
                                if self.system_ctrl:
                                    self.system_ctrl.toggle_pause()
                                print(f"{'⏸ PAUSED' if self.paused else '▶ RESUMED'}")
                            elif key == Qt.Key.Key_D:
                                self.show_debug = not self.show_debug
                            elif key == Qt.Key.Key_F:
                                self.show_fps = not self.show_fps
                            elif key == Qt.Key.Key_H:
                                self.print_controls()
                            elif key == Qt.Key.Key_Z:
                                self.visual.show_gesture_debug['zoom'] = not self.visual.show_gesture_debug['zoom']
                                print(f"Zoom debug: {'ON' if self.visual.show_gesture_debug['zoom'] else 'OFF'}")
                            elif key == Qt.Key.Key_X:
                                self.visual.show_gesture_debug['pinch'] = not self.visual.show_gesture_debug['pinch']
                                print(f"Pinch debug: {'ON' if self.visual.show_gesture_debug['pinch'] else 'OFF'}")
                            elif key == Qt.Key.Key_I:
                                self.visual.show_gesture_debug['pointing'] = not self.visual.show_gesture_debug['pointing']
                                print(f"Pointing debug: {'ON' if self.visual.show_gesture_debug['pointing'] else 'OFF'}")
                            elif key == Qt.Key.Key_S:
                                self.visual.show_gesture_debug['swipe'] = not self.visual.show_gesture_debug['swipe']
                                print(f"Swipe debug: {'ON' if self.visual.show_gesture_debug['swipe'] else 'OFF'}")
                            elif key == Qt.Key.Key_O:
                                self.visual.show_gesture_debug['open_hand'] = not self.visual.show_gesture_debug['open_hand']
                                print(f"Open hand debug: {'ON' if self.visual.show_gesture_debug['open_hand'] else 'OFF'}")
                            elif key == Qt.Key.Key_T:
                                self.visual.show_gesture_debug['thumbs'] = not self.visual.show_gesture_debug['thumbs']
                                print(f"Thumbs debug: {'ON' if self.visual.show_gesture_debug['thumbs'] else 'OFF'}")
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"⚠ Keyboard processing error: {e}")
                
                # Old cv2.waitKey code removed - keyboard handled via PyQt now
                key = 255 # Default to no key pressed
                # Only process if a key was pressed
                if key != 255:
                    # Normalize to lowercase character when possible
                    try:
                        k = chr(key).lower()
                    except Exception:
                        k = None

                    if k == 'q':
                        self.running = False
                    elif k == 'p':
                        self.paused = not self.paused
                        if self.system_ctrl:
                            self.system_ctrl.toggle_pause()
                        print(f"{'⏸ PAUSED' if self.paused else '▶ RESUMED'}")
                    elif k == 'd':
                        self.show_debug = not self.show_debug
                    elif k == 'f':
                        self.show_fps = not self.show_fps
                    elif k == 'h':
                        self.print_controls()
                    # Gesture debug toggles
                    elif k == 'z':
                        self.visual.show_gesture_debug['zoom'] = not self.visual.show_gesture_debug['zoom']
                        print(f"Zoom debug: {'ON' if self.visual.show_gesture_debug['zoom'] else 'OFF'}")
                    elif k == 'x':
                        self.visual.show_gesture_debug['pinch'] = not self.visual.show_gesture_debug['pinch']
                        print(f"Pinch debug: {'ON' if self.visual.show_gesture_debug['pinch'] else 'OFF'}")
                    elif k == 'i':
                        self.visual.show_gesture_debug['pointing'] = not self.visual.show_gesture_debug['pointing']
                        print(f"Pointing debug: {'ON' if self.visual.show_gesture_debug['pointing'] else 'OFF'}")
                    elif k == 's':
                        self.visual.show_gesture_debug['swipe'] = not self.visual.show_gesture_debug['swipe']
                        print(f"Swipe debug: {'ON' if self.visual.show_gesture_debug['swipe'] else 'OFF'}")
                    elif k == 'o':
                        self.visual.show_gesture_debug['open_hand'] = not self.visual.show_gesture_debug['open_hand']
                        print(f"Open hand debug: {'ON' if self.visual.show_gesture_debug['open_hand'] else 'OFF'}")
                    elif k == 't':
                        self.visual.show_gesture_debug['thumbs'] = not self.visual.show_gesture_debug['thumbs']
                        print(f"Thumbs debug: {'ON' if self.visual.show_gesture_debug['thumbs'] else 'OFF'}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Signal GUI to quit if present
        if self.status_queue:
            try:
                # Send shutdown sentinel
                self.status_queue.put(('shutdown', 'shutdown'), timeout=0.5)
            except Exception as e:
                print(f"⚠ Could not send shutdown signal: {e}")
        
        # Small delay to let GUI process shutdown
        try:
            time.sleep(0.2)
        except:
            pass
        
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"⚠ Error releasing camera: {e}")
        
        if self.hand_landmarker:
            try:
                self.hand_landmarker.close()
            except Exception as e:
                print(f"⚠ Error closing hand landmarker: {e}")
        
        if self.hands:
            try:
                self.hands.close()
            except Exception as e:
                print(f"⚠ Error closing hands: {e}")
        
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
        print("\n  GESTURE DEBUG OVERLAYS:")
        print("  Z - Toggle Zoom debug")
        print("  X - Toggle Pinch debug")
        print("  I - Toggle Pointing debug")
        print("  S - Toggle Swipe debug")
        print("  O - Toggle Open hand debug")
        print("  T - Toggle Thumbs debug")
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
        from source_code.config.config_manager import Config
        Config(args.config)
    
    # Create and run application
    # Check display settings
    # show_camera_window: explicit control over camera preview window
    # status_indicator.enabled: explicit control over floating status indicator
    # visual_mode: legacy/fallback (full/minimal/debug)
    
    show_camera = config.get('display', 'show_camera_window', default=True)
    visual_mode = config.get('display', 'visual_mode', default='full')
    status_enabled = config.get('display', 'status_indicator', 'enabled', default=True)
    
    # Fallback: if show_camera_window not set, use visual_mode
    if show_camera is None:
        show_camera = visual_mode in ['full', 'debug']
    
    status_queue = None
    frame_queue = None
    key_queue = None
    
    # Status indicator always runs if enabled (independent of camera window)
    if status_enabled:
        status_queue = queue.Queue()
    
    # Camera window only if show_camera is True
    if show_camera:
        frame_queue = queue.Queue(maxsize=2)
        key_queue = queue.Queue()  # For keyboard input from PyQt
    
    # Print display mode info
    print(f"📺 Display mode: camera_window={show_camera}, status_indicator={status_enabled}")
    
    # Create application
    try:
        app = HANDSApplication(
            camera_idx=args.camera,
            enable_system_control=not args.dry_run,
            status_queue=status_queue
        )
        
        if frame_queue:
            app.set_frame_queue(frame_queue)
        
        if key_queue:
            app.set_key_queue(key_queue)
        
        # Decide how to run based on what GUI elements are needed
        if status_enabled or show_camera:
            # Need PyQt GUI - run app in thread, GUI in main thread
            app_thread = threading.Thread(target=app.run)
            app_thread.daemon = True
            app_thread.start()
            
            # Run GUI in main thread (required for PyQt on some platforms)
            from source_code.gui.status_indicator import run_gui
            run_gui(config, status_queue, frame_queue, key_queue)
        else:
            # No GUI needed - run app directly (headless mode)
            print("🖥️ Running in headless mode (no camera window, no status indicator)")
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
