import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLO model
model_path = 'runs/detect/train12/weights/best.pt'  # Adjust the path if necessary
model = YOLO(model_path)

# Set up webcam
cap = cv2.VideoCapture(0)  # Open the webcam (index 0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Store the track history
track_history = defaultdict(lambda: [])

try:
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # Display the annotated frame
                cv2.imshow("YOLO Tracking", annotated_frame)
            else:
                print("No detections in this frame.")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break
        else:
            # Break the loop if the end of the video is reached
            break

finally:
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
