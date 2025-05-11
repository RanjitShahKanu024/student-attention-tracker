import cv2
import dlib
import face_recognition
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import os
import matplotlib
matplotlib.use('macosx')  # Or use 'Agg' if you're running headless (on server)
import matplotlib.pyplot as plt
from collections import deque
import signal
import sys
import time
from utils import setup_csv, log_blink
import pandas as pd
from datetime import datetime

# Constants
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 2
RECOGNITION_THRESHOLD = 0.45
FRAME_SKIP = 2  # Process every 3rd frame for face recognition
DISPLAY_SCALE = 0.8  # Slightly reduced display size
MOUTH_AR_THRESH = 0.75
(mStart, mEnd) = (60, 68)  # Mouth landmarks (60-67 in 0-based indexing)
ALERT_DURATION = 5  # seconds for alert trigger
REPORT_INTERVAL = 300  # 5 minutes in seconds

# EAR Calculation Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Calculate the Mouth Aspect Ratio to detect yawning"""
    # Horizontal distance between mouth corners
    mouth_width = dist.euclidean(mouth[0], mouth[4])
    
    # Vertical distances between top and bottom lips
    inner_lip_height = dist.euclidean(mouth[2], mouth[6])
    outer_lip_height = dist.euclidean(mouth[1], mouth[7])
    
    # Average the vertical distances and divide by horizontal width
    mar = (inner_lip_height + outer_lip_height) / (2.0 * mouth_width)
    return mar

def estimate_head_pose(shape):
    # 2D image points from facial landmarks
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype="double")
    
    # 3D model points (approximate)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])
    
    # Camera internals (approximate)
    focal_length = frame.shape[1]
    center = (frame.shape[1]/2, frame.shape[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    return rotation_vector, translation_vector, camera_matrix, dist_coeffs

def generate_report(student_data):
    report_data = []
    for name, data in student_data.items():
        avg_ear = np.mean(data["ear_values"]) if data["ear_values"] else 0
        blink_rate = len(data["blink_times"]) / (len(data["ear_values"]) / cap.get(cv2.CAP_PROP_FPS) * 60) if data["ear_values"] else 0
        
        report_data.append({
            "Student": name,
            "Total Blinks": data["total_blinks"],
            "Avg EAR": avg_ear,
            "Blink Rate (bpm)": blink_rate,
            "Attention Score": data.get("attention_score", "N/A"),
            "Status": data["status"],
            "Last Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    df = pd.DataFrame(report_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"student_report_{timestamp}.csv"
    df.to_csv(report_filename, index=False)
    print(f"Report saved as {report_filename}")
    return report_filename

# Load known student encodings
print("Loading known faces...")
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(name)

# Initialize dlib and CSV
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
csv_file, csv_writer = setup_csv()

(lStart, lEnd) = (42, 48)  # Left eye landmarks
(rStart, rEnd) = (36, 42)  # Right eye landmarks

# Graceful Exit Handler
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Video Capture Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

student_data = {}
frame_count = 0
current_student = None
last_face_locations = []
last_face_encodings = []
last_names = []
last_report_time = time.time()

# Setup live plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="EAR")
ax.set_ylim(0, 0.4)
ax.set_xlim(0, 100)
ax.set_title("Live EAR Graph")
ax.set_xlabel("Frame")
ax.set_ylabel("EAR")
ax.legend()
plt.grid(True)

# Main Loop
print("Starting video capture...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    process_this_frame = (frame_count % FRAME_SKIP == 0)
    
    # Convert to RGB and resize for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if process_this_frame:
        # Perform face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        last_face_locations = face_locations
        last_face_encodings = face_encodings
        last_names = []
        
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            
            if distances[best_match_index] < RECOGNITION_THRESHOLD:
                last_names.append(known_names[best_match_index])
            else:
                last_names.append("Unknown")
    else:
        # Reuse previous frame's data
        face_locations = last_face_locations
        face_encodings = last_face_encodings
    
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if process_this_frame:
            name = last_names[len(last_names) - 1] if last_names else "Unknown"
            
            if name != "Unknown":
                current_student = name
                
                if name not in student_data:
                    student_data[name] = {
                        "counter": 0,
                        "total_blinks": 0,
                        "ear_values": deque(maxlen=100),
                        "blink_times": [],
                        "last_blink_time": None,
                        "status": "Focused",
                        "attention_score": 100,
                        "distraction_start": None,
                        "alert_start": None
                    }
        
        if name != "Unknown":
            # Facial landmark detection
            rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)
            
            # Calculate EAR
            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            data = student_data[name]
            data["ear_values"].append(ear)
            
            # Head pose estimation
            rotation_vector, translation_vector, camera_matrix, dist_coeffs = estimate_head_pose(shape_np)
            
            # Mouth detection for yawning
            mouth = shape_np[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "YAWNING", (left, bottom + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                data["status"] = "Drowsy"
            
            # Gaze detection
            left_pupil = np.mean(shape_np[36:42], axis=0)
            right_pupil = np.mean(shape_np[42:48], axis=0)
            pupil_center = ((left_pupil + right_pupil) / 2).astype(int)
            eye_center = shape_np[27]  # Top of nose
            gaze_direction = pupil_center - eye_center
            cv2.arrowedLine(frame, tuple(eye_center), tuple(pupil_center), (255,0,0), 2)
            
            # Draw eye landmarks
            for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Blink detection
            if ear < EYE_AR_THRESH:
                data["counter"] += 1
            else:
                if data["counter"] >= EYE_AR_CONSEC_FRAMES:
                    current_time = time.time()
                    data["total_blinks"] += 1
                    data["blink_times"].append(current_time)
                    data["last_blink_time"] = current_time
                    
                    # Calculate blinks per minute
                    data["blink_times"] = [t for t in data["blink_times"] if current_time - t <= 60]
                    blinks_per_min = len(data["blink_times"])
                    
                    # Log to CSV
                    log_blink(csv_writer, data["total_blinks"], name)
                    
                    # Update status
                    if blinks_per_min < 5:
                        data["status"] = "Drowsy"
                    else:
                        data["status"] = "Focused"
                
                data["counter"] = 0
            
            # Attention score calculation
            # Deduct points for drowsiness
            if data["status"] == "Drowsy":
                data["attention_score"] = max(0, data["attention_score"] - 0.5)
            
            # Check if looking away
            if np.abs(gaze_direction[0]) > 30:  # Threshold for looking away
                if data["distraction_start"] is None:
                    data["distraction_start"] = time.time()
                else:
                    duration = time.time() - data["distraction_start"]
                    if duration > 3:  # 3 seconds of looking away
                        data["attention_score"] = max(0, data["attention_score"] - 1)
            else:
                data["distraction_start"] = None
                # Small reward for paying attention
                data["attention_score"] = min(100, data["attention_score"] + 0.1)
            
            # Real-time alerts
            if data["status"] == "Drowsy":
                current_time = time.time()
                if data["alert_start"] is None:
                    data["alert_start"] = current_time
                elif current_time - data["alert_start"] > ALERT_DURATION:
                    cv2.putText(frame, "ALERT: PAY ATTENTION!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                data["alert_start"] = None
            
            # Draw face box and info
            status_color = (0, 255, 0) if data["status"] == "Focused" else (0, 165, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), status_color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            cv2.putText(frame, f"Blinks: {data['total_blinks']}", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(frame, f"Status: {data['status']}", (left, bottom + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(frame, f"Attention: {int(data['attention_score'])}%", 
                        (left, bottom + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        else:
            # Unknown face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Periodic reporting
    current_time = time.time()
    if current_time - last_report_time > REPORT_INTERVAL:
        generate_report(student_data)
        last_report_time = current_time
    
    # Update the EAR graph live
    if current_student:
        ydata = list(student_data[current_student]["ear_values"])
        xdata = list(range(len(ydata)))
        line.set_ydata(ydata)
        line.set_xdata(xdata)
        ax.set_xlim(0, max(100, len(ydata)))
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Display the frame
    display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Student Attention Tracker", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Save EAR Graphs
print("Saving graphs...")
for name, data in student_data.items():
    fig, ax = plt.subplots()
    ax.plot(list(data["ear_values"]))
    ax.set_title(f"{name} - Final EAR Graph")
    ax.set_xlabel("Frame")
    ax.set_ylabel("EAR")
    ax.grid(True)
    fig.savefig(f"{name}_ear_graph.png")
    plt.close(fig)

# Generate final report and cleanup
print("\n--- Student Summary ---")
for name, data in student_data.items():
    print(f"{name}: {data['total_blinks']} blinks, Attention: {data.get('attention_score', 'N/A')}%, Final status: {data['status']}")

report_file = generate_report(student_data)
csv_file.close()
cap.release()
cv2.destroyAllWindows()