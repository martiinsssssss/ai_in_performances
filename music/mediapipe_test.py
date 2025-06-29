import cv2
import mediapipe as mp
import time
import csv
import os
from pythonosc import udp_client

# --- OSC Configuration ---
# These MUST match your OSC In CHOP settings in TouchDesigner.
OSC_IP = "127.0.0.1"
OSC_PORT = 8000 
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
print(f"OSC client initialized to send to {OSC_IP}:{OSC_PORT}")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5   
)

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("CRITICAL ERROR: Could not open webcam. Please check:")
    print("  1. If webcam is connected.")
    print("  2. If another application is using the webcam (close all other apps).")
    print("  3. If you have proper camera permissions for your terminal/IDE.")
    print("  4. Try a different camera index (e.g., cv2.VideoCapture(1)).")
    exit() 

# *** IMPORTANT: Change these values for EACH test run! ***
session_name = "td_test_intermitent_light" 
duration_seconds = 20               # Duration of data collection for this single run

output_dir = "collected_data_osc" # Folder to save data
os.makedirs(output_dir, exist_ok=True)

# Generate a unique filename with session name and timestamp
timestamp = int(time.time())
output_filename = os.path.join(output_dir, f"{session_name}_{timestamp}.csv")

csv_file = None
csv_writer = None

try:
    csv_file = open(output_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # CSV Headers for comprehensive data logging
    csv_writer.writerow([
        'global_timestamp',             # Timestamp when the frame was processed
        'hand_id',                      # 0 for first hand, 1 for second
        'landmark_id',                  # 0 to 20 for each hand landmark
        'x_normalized', 'y_normalized', 'z_relative', # Normalized coordinates [0.0, 1.0]
        'hand_confidence_score',        # Confidence score for the hand detection
        'frame_processing_time_ms',     # Time taken to process this specific frame
        'frame_width_px',               # Original frame width in pixels
        'frame_height_px'               # Original frame height in pixels
    ])

    print(f"Starting data collection and OSC sending for session: '{session_name}'")
    print(f"This session will run for {duration_seconds} seconds.")
    print("No OpenCV window will be displayed.")
    print("Press 'Ctrl+C' in the console to stop manually at any time.")

    start_session_time = time.time()
    
    while True:
        frame_start_time = time.time() 

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001) 
            continue 
        
        frame_height, frame_width, _ = frame.shape 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        results = hands.process(rgb_frame) 

        frame_processing_end_time = time.time()
        frame_processing_time_ms = (frame_processing_end_time - frame_start_time) * 1000 

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_confidence = -1.0 
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    if hasattr(results.multi_handedness[hand_idx], 'score'):
                        hand_confidence = results.multi_handedness[hand_idx].score
                
                for i, lm in enumerate(hand_landmarks.landmark):
                    if i in [8, 12, 16]:  
                        client.send_message(f"/finger{i}", [lm.x, lm.y]) 

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                client.send_message(f"/hand/{hand_idx}/wrist_pos", [wrist_x, wrist_y])
                
                if hand_idx == 0: 
                     client.send_message("/hand/main_y", wrist_y)


                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    csv_writer.writerow([
                        time.time(), 
                        hand_idx,
                        landmark_id, 
                        landmark.x, landmark.y, landmark.z, 
                        hand_confidence, 
                        frame_processing_time_ms, 
                        frame_width, 
                        frame_height 
                    ])
        else:
            client.send_message("/hand/none_detected", 1) 
            csv_writer.writerow([
                time.time(), -1, -1, -1.0, -1.0, -1.0,
                0.0, 
                frame_processing_time_ms, 
                frame_width, 
                frame_height
            ])
        

        if (time.time() - start_session_time) > duration_seconds:
            print(f"Session '{session_name}' finished by duration ({duration_seconds}s).")
            break 

except KeyboardInterrupt:
    print(f"\nSession '{session_name}' stopped manually by user (Ctrl+C).")
except Exception as e:
    print(f"An unexpected error occurred during session '{session_name}': {e}")
finally:
    if csv_file:
        csv_file.close() 
    hands.close()      
    cap.release()     
    print(f"Data for '{session_name}' saved to: {output_filename}")
    print("--- Script Finished ---")