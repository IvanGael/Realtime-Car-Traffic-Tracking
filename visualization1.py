import sys
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Handling command line arguments
if len(sys.argv) < 2:
    print("Usage: py visualization1.py <video_path>")
    sys.exit(1)
video_path = sys.argv[1]  # Get video path from command line argument

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize list to store annotated frames
annotated_frames = []

# Start time for tracking
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if success:
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
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
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
