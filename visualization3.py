import sys
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Handling command line arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]  # Get video path from command line argument

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create mask to divide road into two compartments
def create_mask(width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts1 = np.array([[0, height//2], [width, height//2], [width, height], [0, height]], np.int32)
    pts2 = np.array([[0, 0], [width, 0], [width, height//2], [0, height//2]], np.int32)
    cv2.fillPoly(mask, [pts1], 1)
    cv2.fillPoly(mask, [pts2], 2)
    return mask

mask = create_mask(frame_width, frame_height)

# Store the track history
track_history = defaultdict(lambda: [])

# Store unique car IDs for each compartment
unique_car_ids_down = set()
unique_car_ids_up = set()

# Initialize list to store annotated frames
annotated_frames = []

# Start time for tracking
start_time = time.time()

# Variables for FPS calculation
frame_count = 0
fps = 0
fps_update_interval = 1  # Update FPS every second
last_fps_update = start_time

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - last_fps_update
        if elapsed_time > fps_update_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            last_fps_update = current_time

        results = model.track(frame, persist=True)
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255,20,147), thickness=10)
                
                # Check which compartment the vehicle is in
                compartment = mask[int(y), int(x)]
                if compartment == 1:
                    unique_car_ids_down.add(track_id)
                elif compartment == 2:
                    unique_car_ids_up.add(track_id)
            
            # Display car count and FPS on the frame
            down_count_text = f"Left Lane: {len(unique_car_ids_down)} vehicles"
            up_count_text = f"Right Lane: {len(unique_car_ids_up)} vehicles"
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, down_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (194, 247, 50), 3)
            cv2.putText(annotated_frame, up_count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (194, 247, 50), 3)
            cv2.putText(annotated_frame, fps_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (194, 247, 50), 3)
            
            annotated_frames.append(annotated_frame)
            
            # Display frame in OpenCV GUI
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.imshow('Frame', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break  # Break the loop if 'q' is pressed
            
            # Check if 2 minutes have elapsed
            if time.time() - start_time >= 120:
                break
    else:
        break

# Release the video capture object
cap.release()

# Write annotated frames to output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_rate = 30  # Set frame rate
frame_size = (annotated_frames[0].shape[1], annotated_frames[0].shape[0])  # Use first frame's size
output_video_path = video_path.split('.')[0] + '_tracked.mp4'  # Output file name with '_tracked' suffix
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

for frame in annotated_frames:
    out.write(frame)

out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()