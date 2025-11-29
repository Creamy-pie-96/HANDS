#!/usr/bin/env python3
"""
Comprehensive Gesture Control Demo
Tests all single-hand and two-hand gestures with live camera feed.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import os

from bimanual_gestures import ComprehensiveGestureManager
from gesture_detectors import visualize_hand_metrics

# MediaPipe setup with optimizations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3,
    max_num_hands=2  # Essential for two-hand gestures
)
mp_draw = mp.solutions.drawing_utils


def draw_gesture_info(frame, gestures_dict, y_start=50):
    """
    Draw gesture information on frame.
    Returns next available y position.
    """
    y = y_start
    
    for category, gestures in gestures_dict.items():
        if not gestures:
            continue
            
        # Category header
        color = {
            'left': (0, 255, 255),    # Yellow
            'right': (255, 128, 0),   # Orange
            'bimanual': (255, 0, 255)  # Magenta
        }.get(category, (255, 255, 255))
        
        cv2.putText(frame, f"{category.upper()}:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
        
        # Individual gestures
        for gesture_name, result in gestures.items():
            text = f"  {gesture_name.upper()}"
            cv2.putText(frame, text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 25
            
            # Key metadata on same line or next line
            if result.metadata:
                meta_items = []
                for key, val in result.metadata.items():
                    if isinstance(val, float):
                        meta_items.append(f"{key}={val:.3f}")
                    elif isinstance(val, tuple) and len(val) == 2:
                        meta_items.append(f"{key}=({val[0]:.2f},{val[1]:.2f})")
                    elif key in ['zoom_type', 'resize_type', 'direction']:
                        meta_items.append(f"{key}={val}")
                
                if meta_items:
                    meta_text = "    " + ", ".join(meta_items[:3])  # Max 3 items
                    cv2.putText(frame, meta_text, (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y += 18
    
    return y


def draw_tuning_panel(frame, left_metrics, right_metrics):
    """
    Draw real-time tuning values panel on the right side.
    """
    w = frame.shape[1]
    x = w - 280
    y = 50
    
    # Panel title
    cv2.putText(frame, "TUNING VALUES", (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 25
    
    # Left hand values
    if left_metrics:
        cv2.putText(frame, "LEFT HAND:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 18
        
        # Pinch distance
        pinch_dist = left_metrics.tip_distances.get('index_thumb', 0.0)
        cv2.putText(frame, f"  Pinch: {pinch_dist:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Zoom spread
        zoom_spread = left_metrics.tip_distances.get('index_thumb', 0.0)
        finger_gap = left_metrics.tip_distances.get('index_middle', 0.0)
        cv2.putText(frame, f"  Zoom: {zoom_spread:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        cv2.putText(frame, f"  FingerGap: {finger_gap:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Velocity
        vel = np.hypot(*left_metrics.velocity) if left_metrics.velocity else 0.0
        cv2.putText(frame, f"  Vel: {vel:.3f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Fingers extended
        fingers = sum(left_metrics.fingers_extended.values())
        cv2.putText(frame, f"  Fingers: {fingers}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20
    
    # Right hand values
    if right_metrics:
        cv2.putText(frame, "RIGHT HAND:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
        y += 18
        
        # Pinch distance
        pinch_dist = right_metrics.tip_distances.get('index_thumb', 0.0)
        cv2.putText(frame, f"  Pinch: {pinch_dist:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Zoom spread
        zoom_spread = right_metrics.tip_distances.get('index_thumb', 0.0)
        finger_gap = right_metrics.tip_distances.get('index_middle', 0.0)
        cv2.putText(frame, f"  Zoom: {zoom_spread:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        cv2.putText(frame, f"  FingerGap: {finger_gap:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Velocity
        vel = np.hypot(*right_metrics.velocity) if right_metrics.velocity else 0.0
        cv2.putText(frame, f"  Vel: {vel:.3f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 15
        
        # Fingers extended
        fingers = sum(right_metrics.fingers_extended.values())
        cv2.putText(frame, f"  Fingers: {fingers}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20
    
    # Inter-hand distance
    if left_metrics and right_metrics:
        from math_utils import euclidean
        inter_dist = euclidean(left_metrics.centroid, right_metrics.centroid)
        cv2.putText(frame, f"Inter-hand: {inter_dist:.4f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)


def main(camera_idx=0, width=640, height=480, show_metrics=True):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    print("✓ Camera opened")
    print("DISPLAY=", os.environ.get("DISPLAY"), "WAYLAND_DISPLAY=", os.environ.get("WAYLAND_DISPLAY"))

    cv2.namedWindow("Comprehensive Gesture Demo", cv2.WINDOW_NORMAL)
    
    # Performance optimizations
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Verify settings
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    # Initialize comprehensive gesture manager
    gesture_mgr = ComprehensiveGestureManager()
    
    # Stats
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    debug = False
    
    print("\n🎮 Comprehensive Gesture Control Demo")
    print("=" * 60)
    print("Controls:")
    print("  q - quit")
    print("  m - toggle metrics visualization")
    print("  d - toggle debug info")
    print("\nSingle-Hand Gestures:")
    print("  • Pinch: Thumb + Index close together")
    print("  • Zoom: Index + Middle together, spread/pinch with thumb")
    print("  • Pointing: Index only extended")
    print("  • Swipe: Fast hand movement")
    print("  • Open Hand: All 5 fingers extended")
    print("\nTwo-Hand Gestures:")
    print("  • Pan: Left still + Right move")
    print("  • Rotate: Left still + Right zoom")
    print("  • Two-Hand Resize: Both pinch + change distance")
    print("  • Precision Cursor: Left still + Right point")
    print("  • Draw Mode: Left pinch + Right point & move")
    print("  • Undo: Left still + Right swipe left")
    print("  • Quick Menu: Left zoom + Right pinch")
    print("  • Warp: Both hands pointing (far apart)")
    print("=" * 60)

    try:
        running = True
        while running:
            frame_count += 1
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            
            # Compute FPS
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - fps_time)
                fps_time = now
            
            # Status bar
            status = f"FPS:{fps:.1f} Frame:{frame_count} Metrics:{'ON' if show_metrics else 'OFF'} Debug:{'ON' if debug else 'OFF'}"
            cv2.putText(frame_bgr, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            left_landmarks = None
            right_landmarks = None
            left_metrics = None
            right_metrics = None

            if results.multi_hand_landmarks and results.multi_handedness:
                # Separate left and right hands
                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    hand_label = handedness.classification[0].label.lower()
                    
                    # Draw MediaPipe landmarks
                    mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Store landmarks by hand
                    if hand_label == 'left':
                        left_landmarks = hand_landmarks
                    else:
                        right_landmarks = hand_landmarks
            
            # Process gestures
            all_gestures = gesture_mgr.process_hands(
                left_landmarks,
                right_landmarks,
                frame_bgr.shape
            )
            
            # Get metrics for visualization
            if left_landmarks and gesture_mgr.single_hand_mgr.history['left']:
                left_metrics = gesture_mgr.single_hand_mgr.history['left'][-1]
                if show_metrics:
                    visualize_hand_metrics(frame_bgr, left_metrics, color=(0, 255, 255))
            
            if right_landmarks and gesture_mgr.single_hand_mgr.history['right']:
                right_metrics = gesture_mgr.single_hand_mgr.history['right'][-1]
                if show_metrics:
                    visualize_hand_metrics(frame_bgr, right_metrics, color=(255, 128, 0))
            
            # Draw gesture information
            draw_gesture_info(frame_bgr, all_gestures)
            
            # Draw tuning panel
            if show_metrics:
                draw_tuning_panel(frame_bgr, left_metrics, right_metrics)
            
            # Terminal output for detected gestures
            for category, gestures in all_gestures.items():
                for gesture_name, result in gestures.items():
                    if debug or category == 'bimanual':  # Always print bimanual
                        print(f"[{category}] {gesture_name.upper()}: {result.metadata}")
            
            # Display frame
            cv2.imshow("Comprehensive Gesture Demo", frame_bgr)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('m'):
                show_metrics = not show_metrics
                print(f"Metrics visualization: {'ON' if show_metrics else 'OFF'}")
            elif key == ord('d'):
                debug = not debug
                print(f"Debug info: {'ON' if debug else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Demo completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive gesture control demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics visualization")
    args = parser.parse_args()
    
    main(args.camera, args.width, args.height, not args.no_metrics)
