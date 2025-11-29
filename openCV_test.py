import argparse
import cv2
import mediapipe as mp
import time
import os

# pyautogui is optional; wrap import in case running headless or on systems without display
try:
    import pyautogui
except Exception:
    pyautogui = None
from math_utils import(
    landmarks_to_array,
    normalized_to_pixels,
    EWMA,
    ClickDetector,
    euclidean
) 

# Media Pipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Smoothers & detectors
idx_ewma = EWMA(alpha=0.25) # will tweak this alpha
thumb_ewma = EWMA(alpha=0.25)
# Use safer defaults to reduce false positives during early testing
click_detector = ClickDetector(thresh_px=15, hold_frames=5, cooldown_s=0.6)

ENABLE_MOUSE = False


def main(camera_idx=0, width=640, height=400):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Could not open camera! Try a different --camera index or check permissions.")
        return

    # Debug: show what display environment variables are set
    print("DISPLAY=", os.environ.get("DISPLAY"), "WAYLAND_DISPLAY=", os.environ.get("WAYLAND_DISPLAY"))

    # Create a named window (helps some backends) and allow resize
    cv2.namedWindow("web demo", cv2.WINDOW_NORMAL)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        running = True
        enable_mouse = ENABLE_MOUSE
        debug = False
        prev_time = None
        prev_pos = None
        vx, vy = 0.0, 0.0
        frame_count = 0
        # velocity threshold (pixels per second) for swipe detection
        SWIPE_VX_THRESH = 1000.0
        SWIPE_VY_THRESH = 1000.0
        while running:
            frame_count += 1
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            
            # Add status bar overlay
            status = f"Frame:{frame_count} Mouse:{'ON' if enable_mouse else 'OFF'} Debug:{'ON' if debug else 'OFF'}"
            cv2.putText(frame_bgr, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # debug: print basic frame stats when debugging is enabled (but only every 30 frames to reduce spam)
            if debug and frame_count % 30 == 0:
                try:
                    print(
                        f"[Frame {frame_count}] shape={frame_bgr.shape} min={frame_bgr.min()} max={frame_bgr.max()} mean={frame_bgr.mean():.1f}"
                    )
                except Exception:
                    pass
            frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # for now use only the first detected hand for testing
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Add visual indicator that hand is detected
                cv2.putText(frame_bgr, "HAND DETECTED", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # draw MP landmarks for visual debugging
                mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # convert to Nx2 normalized array
                norm = landmarks_to_array(hand_landmarks.landmark) # shape [21,2]
                # map normalized coords to pixel coords (returns (N,2) ints)
                pts_px = normalized_to_pixels(norm, frame_bgr.shape)
                #index fingertip = landmark 8, thump tips = landmark 4
                idx_px = pts_px[8]
                thumb_px = pts_px[4]

                # sanity check: landmarks must lie inside the frame
                h, w = frame_bgr.shape[:2]
                if not (
                    0 <= idx_px[0] < w
                    and 0 <= idx_px[1] < h
                    and 0 <= thumb_px[0] < w
                    and 0 <= thumb_px[1] < h
                ):
                    # Skip unreliable frame
                    if debug:
                        print("Landmarks out of frame bounds — skipping frame")
                    continue

                # smoothing 
                sm_idx = idx_ewma.update(idx_px)
                sm_thumb = thumb_ewma.update(thumb_px)

                #compute the distance (pixel units)
                dist = euclidean(sm_idx,sm_thumb)
                # compute velocities (px/s) using previous smoothed position
                now = time.time()
                if prev_time is not None and prev_pos is not None:
                    dt = max(1e-6, now - prev_time)
                    vx = (sm_idx[0] - prev_pos[0]) / dt
                    vy = (sm_idx[1] - prev_pos[1]) / dt
                else:
                    vx, vy = 0.0, 0.0
                prev_time = now
                prev_pos = sm_idx.copy()

                # debug prints: distance and velocity (reduced spam: only every 10 frames)
                if debug and frame_count % 10 == 0:
                    print(f"[Frame {frame_count}] dist={dist:.1f}, vx={vx:.1f}, vy={vy:.1f}")

                # swipe detection based on velocity thresholds
                if abs(vx) > SWIPE_VX_THRESH:
                    direction = "right" if vx > 0 else "left"
                    cv2.putText(frame_bgr, f"SWIPE {direction}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)
                    print(f"Swipe {direction} detected (vx={vx:.1f})")
                if abs(vy) > SWIPE_VY_THRESH:
                    direction = "down" if vy > 0 else "up"
                    cv2.putText(frame_bgr, f"SWIPE {direction}", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)
                    print(f"Swipe {direction} detected (vy={vy:.1f})")

                # visualization: draw raw and smoothed points
                cv2.circle(frame_bgr, tuple(idx_px), 6, (255, 0, 0), 2)          # raw index
                cv2.circle(frame_bgr, tuple(thumb_px), 6, (0, 0, 255), 2)        # raw thumb
                cv2.circle(frame_bgr, tuple(sm_idx.astype(int)), 8, (0, 255, 0), -1)  # smoothed index

                # show distance
                cv2.putText(frame_bgr, f"d={int(dist)}", (10, frame_bgr.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # gesture detection
                if click_detector.pinched(dist):
                    cv2.putText(frame_bgr, "PINCH!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    if debug:
                        print(f"[Frame {frame_count}] PINCH detected (dist={dist:.1f})")
                    # only move/click the OS mouse if enabled
                    if enable_mouse and pyautogui is not None:
                        # Map smoothed point to screen coordinates before moving:
                        screen_w, screen_h = pyautogui.size()
                        fx = sm_idx[0] / frame_bgr.shape[1]   # normalized x
                        fy = sm_idx[1] / frame_bgr.shape[0]   # normalized y
                        px = int(fx * screen_w)
                        py = int(fy * screen_h)
                        pyautogui.click(px, py)
                    elif enable_mouse and pyautogui is None:
                        print("ENABLE_MOUSE requested but pyautogui is not available on this system")
            else:
                # No hand detected - show status
                cv2.putText(frame_bgr, "No hand detected", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            
            # CRITICAL: Actually display the frame!
            cv2.imshow("web demo", frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('m'):
                enable_mouse = not enable_mouse
                print("ENABLE_MOUSE =", enable_mouse)
            elif key == ord('d'):
                debug = not debug
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    args = parser.parse_args()
    main(args.camera, args.width, args.height)

