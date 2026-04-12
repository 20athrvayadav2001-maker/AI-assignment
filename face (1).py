import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
import queue
import pyttsx3

# ================= AUDIO ALERT SETUP =================
is_speaking = False

def trigger_warning(msg):
    global is_speaking
    if is_speaking:
        return
        
    def tts_task():
        global is_speaking
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
            
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print("Audio error:", e)
        finally:
            is_speaking = False

    is_speaking = True
    threading.Thread(target=tts_task, daemon=True).start()

# ================= CONSTANTS & THRESHOLDS =================
EAR_THRESH = 0.22      # Eye Aspect Ratio threshold (below this means eyes are closed)
MAR_THRESH = 0.5       # Mouth Aspect Ratio threshold (above means yawning)
TILT_THRESH = 70       # Degrees of head tilt to be considered "tilting"
CLOSE_TIME_THRESH = 3.0 # Seconds before eye closure triggers alert
TILT_TIME_THRESH = 3.0  # Seconds before head tilt triggers alert
SLOW_BLINK_THRESH = 0.5 # Seconds before slow blink triggers alert
BLINK_WINDOW_SEC = 30.0 # Time window for blink count
MAX_BLINKS = 20         # Maximum blinks in the time window
WARNING_DURATION_SEC = 10.0 # Time to keep warning on screen

# ================= MEDIAPIPE INDICES =================
# Left eye
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Mouth
MOUTH_TOP_BOTTOM = (13, 14)
MOUTH_LEFT_RIGHT = (78, 308)
# Head Pose reference points
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip 1
    (0.0, -330.0, -65.0),        # Chin 152
    (-225.0, 170.0, -135.0),     # Left eye left corner 33
    (225.0, 170.0, -135.0),      # Right eye right corner 263
    (-150.0, -150.0, -125.0),    # Left Mouth corner 61
    (150.0, -150.0, -125.0)      # Right mouth corner 291
])
HEAD_POSE_LANDMARK_INX = [1, 152, 33, 263, 61, 291]

# ================= HELPER FUNCTIONS =================
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(eye_points, landmarks, img_w, img_h):
    # Retrieve pixel coords
    pts = [ (landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in eye_points ]
    # vertical dists
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    # horizontal dist
    h = euclidean_dist(pts[0], pts[3])
    # Avoid div by 0 just in case
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear

def get_mar(landmarks, img_w, img_h):
    p_top = (landmarks[MOUTH_TOP_BOTTOM[0]].x * img_w, landmarks[MOUTH_TOP_BOTTOM[0]].y * img_h)
    p_bot = (landmarks[MOUTH_TOP_BOTTOM[1]].x * img_w, landmarks[MOUTH_TOP_BOTTOM[1]].y * img_h)
    p_left = (landmarks[MOUTH_LEFT_RIGHT[0]].x * img_w, landmarks[MOUTH_LEFT_RIGHT[0]].y * img_h)
    p_right = (landmarks[MOUTH_LEFT_RIGHT[1]].x * img_w, landmarks[MOUTH_LEFT_RIGHT[1]].y * img_h)
    v = euclidean_dist(p_top, p_bot)
    h = euclidean_dist(p_left, p_right)
    return v / (h + 1e-6)

def get_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
        for idx in HEAD_POSE_LANDMARK_INX
    ], dtype="double")
    
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, translation_vec = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )
    
    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360 # Usually up/down
    yaw = angles[1] * 360   # Usually left/right
    roll = angles[2] * 360  # Usually tilt shoulder to shoulder
    
    return pitch, yaw, roll

# ================= MAIN LOOP =================
def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Monitoring")
    parser.add_argument('--video', type=str, default=None, help="Path to a video file. Defaults to webcam if not provided.")
    args = parser.parse_args()

    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    eye_closed = False
    eye_closed_start_time = None
    
    head_tilted = False
    head_tilted_start_time = None
    
    yawn_count = 0
    yawn_active = False

    blink_timestamps = []

    warning_active = False
    warning_start_time = 0
    current_warning_msg = ""

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
            
        frame = cv2.flip(frame, 1) # Mirror image
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        current_time = time.time()
        
        display_warning = ""

        if warning_active:
            if current_time - warning_start_time < WARNING_DURATION_SEC:
                display_warning = current_warning_msg
            else:
                warning_active = False
                eye_closed = False
                eye_closed_start_time = None
                head_tilted = False
                head_tilted_start_time = None
                blink_timestamps = []

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Calculate EAR
                left_ear = get_ear(LEFT_EYE_IDX, face_landmarks, w, h)
                right_ear = get_ear(RIGHT_EYE_IDX, face_landmarks, w, h)
                ear = (left_ear + right_ear) / 2.0
                
                # Calculate MAR
                mar = get_mar(face_landmarks, w, h)
                
                # Calculate Head Pose
                pitch, yaw, roll = get_head_pose(face_landmarks, w, h)
                
                if not warning_active:
                    # ================= 1. EYE LOGIC (Blinks & Closure) =================
                    if ear < EAR_THRESH:
                        if not eye_closed:
                            eye_closed = True
                            eye_closed_start_time = current_time
                    else:
                        if eye_closed:
                            eye_closed = False
                            # If eye was closed for less than the threshold, it was just a blink
                            if eye_closed_start_time is not None:
                                close_dur = current_time - eye_closed_start_time
                                if close_dur < SLOW_BLINK_THRESH:
                                    blink_timestamps.append(current_time)
                    
                    # Check closed eyes durations
                    if eye_closed and eye_closed_start_time is not None:
                        closed_duration = current_time - eye_closed_start_time
                        if closed_duration >= CLOSE_TIME_THRESH:
                            display_warning = "DRIVER IS TIRED TAKE A BREAK [Eyes Closed]"
                            trigger_warning("take a break you are tired")
                        elif closed_duration >= SLOW_BLINK_THRESH:
                            display_warning = "DRIVER IS DROWSY [Slow Blink]"
                            trigger_warning("wake up you are drowsy")
                    
                    # Clean up old blinks
                    blink_timestamps = [t for t in blink_timestamps if (current_time - t) <= BLINK_WINDOW_SEC]
                    
                    # Check frequent blinks
                    if len(blink_timestamps) > MAX_BLINKS:
                        display_warning = "DRIVER IS TIRED TAKE A BREAK [Frequent Blinks]"
                        trigger_warning("take a break you are tired")
                    
                    # ================= 2. HEAD LOGIC (Tilting) =================
                    # absolute angle check (depends on coordinate frame, let's assume > TILT_THRESH is bad)
                    if abs(pitch) > TILT_THRESH or abs(roll) > TILT_THRESH or abs(yaw) > TILT_THRESH + 10:
                        if not head_tilted:
                            head_tilted = True
                            head_tilted_start_time = current_time
                        elif (current_time - head_tilted_start_time) >= TILT_TIME_THRESH:
                            display_warning = "DRIVER IS TIRED TAKE A BREAK [Head Tilted]"
                            trigger_warning("take a break you are tired")
                    else:
                        head_tilted = False
                    
                    # ================= 3. YAWN LOGIC =================
                    if mar > MAR_THRESH:
                        if not yawn_active:
                            yawn_active = True
                    else:
                        if yawn_active:
                            yawn_active = False
                            yawn_count += 1

                    if display_warning:
                        if "[Eyes Closed]" in display_warning:
                            # Display updates continuously until eyes open, bypassing the 10s cooldown
                            pass
                        else:
                            warning_active = True
                            warning_start_time = current_time
                            current_warning_msg = display_warning

                # Debug Info on screen
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks (30s): {len(blink_timestamps)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR (Yawn): {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch:{pitch:.1f} Roll:{roll:.1f} Yaw:{yaw:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Warning overlay
        if display_warning:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            text_size = cv2.getTextSize(display_warning, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 50
            cv2.putText(frame, display_warning, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Driver Drowsiness Monitoring', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Cleanup audio (handled automatically as threads are daemonized)

if __name__ == "__main__":
    main()