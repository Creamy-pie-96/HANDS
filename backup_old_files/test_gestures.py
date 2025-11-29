#!/usr/bin/env python3
"""
Test script for gesture detection system.
Tests all detectors with live camera feed.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import os

from gesture_detectors import (
    GestureManager,
    visualize_hand_metrics,
    PinchDetector,
    PointingDetector,
    SwipeDetector,
    ZoomDetector,
    OpenHandDetector
)

# MediaPipe setup with optimizations for fast motion tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,  # Lower for better fast motion tracking
    max_num_hands=2  # Track both hands for two-hand gestures
)
mp_draw = mp.solutions.drawing_utils


def main(camera_idx=0, width=640, height=400, show_metrics=True):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    print("✓ Camera opened")
    print("DISPLAY=", os.environ.get("DISPLAY"), "WAYLAND_DISPLAY=", os.environ.get("WAYLAND_DISPLAY"))

    cv2.namedWindow("Gesture Test", cv2.WINDOW_NORMAL)
    
    # Performance optimizations
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS (hardware dependent)
    
    # Verify actual camera settings
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    # Initialize gesture manager
    gesture_mgr = GestureManager()
    
    # Stats
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    
    print("\n🎮 Gesture Detection Test")
    print("=" * 50)
    print("Controls:")
    print("  q - quit")
    print("  m - toggle metrics visualization")
    print("  d - toggle debug info")
    print("\nTest each gesture:")
    print("  1. Open hand (5 fingers)")
    print("  2. Pointing (index only)")
    print("  3. Pinch (thumb + index close)")
    print("  4. Swipe (fast hand movement)")
    print("  5. Zoom (3 fingers spread/pinch)")
    print("=" * 50)

    debug = False

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

            if results.multi_hand_landmarks:
                # Process each detected hand
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine hand label
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label.lower()
                    else:
                        hand_label = 'right' if idx == 0 else 'left'
                    
                    # Draw MediaPipe landmarks
                    mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Process hand through gesture manager
                    gestures = gesture_mgr.process_hand(
                        hand_landmarks,
                        frame_bgr.shape,
                        hand_label
                    )
                    
                    # Get metrics for visualization
                    if show_metrics and gesture_mgr.history[hand_label]:
                        metrics = gesture_mgr.history[hand_label][-1]
                        visualize_hand_metrics(frame_bgr, metrics)
                        
                        # Real-time tuning display: show key detection values
                        tune_y = 50 + (idx * 200) + 120
                        cv2.putText(frame_bgr, f"TUNING VALUES ({hand_label}):", (400, tune_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        tune_y += 20
                        
                        # Pinch distance
                        pinch_dist = metrics.tip_distances.get(('thumb', 'index'), 0.0)
                        cv2.putText(frame_bgr, f"Pinch: {pinch_dist:.4f}", (400, tune_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        tune_y += 15
                        
                        # Zoom spread (3-finger weighted)
                        t_i = metrics.tip_distances.get(('thumb', 'index'), 0.0)
                        t_m = metrics.tip_distances.get(('thumb', 'middle'), 0.0)
                        i_m = metrics.tip_distances.get(('index', 'middle'), 0.0)
                        zoom_spread = (2*t_i + 2*t_m + i_m) / 5.0
                        cv2.putText(frame_bgr, f"Zoom: {zoom_spread:.4f}", (400, tune_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        tune_y += 15
                        
                        # Velocity
                        vel_mag = np.hypot(*metrics.velocity) if metrics.velocity else 0.0
                        cv2.putText(frame_bgr, f"Vel: {vel_mag:.3f}", (400, tune_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        tune_y += 15
                        
                        # Extension ratio (for pointing)
                        if metrics.fingers_extended['index']:
                            ext_dist = metrics.tip_distances.get(('index', 'middle'), 0.0)
                            cv2.putText(frame_bgr, f"Ext: {ext_dist:.4f}", (400, tune_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Display detected gestures
                    y_offset = 50 + (idx * 150)  # Offset for multiple hands
                    
                    # Hand label
                    cv2.putText(frame_bgr, f"{hand_label.upper()} HAND", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 30
                    
                    if gestures:
                        for gesture_name, result in gestures.items():
                            # Color based on gesture type
                            colors = {
                                'pinch': (0, 255, 255),      # Yellow
                                'pointing': (0, 255, 0),     # Green
                                'swipe': (255, 128, 0),      # Orange
                                'zoom': (255, 0, 255),       # Magenta
                                'open_hand': (0, 128, 255),  # Light blue
                            }
                            color = colors.get(gesture_name, (255, 255, 255))
                            
                            # Main gesture text
                            text = f"✓ {gesture_name.upper()}"
                            cv2.putText(frame_bgr, text, (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            y_offset += 25
                            
                            # Metadata (if debug)
                            if debug and result.metadata:
                                for key, val in result.metadata.items():
                                    if isinstance(val, float):
                                        meta_text = f"  {key}={val:.3f}"
                                    elif isinstance(val, tuple) and len(val) == 2:
                                        meta_text = f"  {key}=({val[0]:.2f},{val[1]:.2f})"
                                    else:
                                        meta_text = f"  {key}={val}"
                                    cv2.putText(frame_bgr, meta_text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                                    y_offset += 15
                            
                            # Terminal output for all detected gestures
                            print(f"[{hand_label}] {gesture_name.upper()}: confidence={result.confidence:.2f}, metadata={result.metadata}")
                    else:
                        cv2.putText(frame_bgr, "No gesture", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            else:
                # No hands detected
                cv2.putText(frame_bgr, "No hands detected", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Display frame
            cv2.imshow("Gesture Test", frame_bgr)
            
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
        print("\n✓ Test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gesture detection system")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=400, help="Frame height")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics visualization")
    args = parser.parse_args()
    
    main(args.camera, args.width, args.height, not args.no_metrics)
