import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLO model
model_path = 'runs/detect/train12/weights/best.pt'  # Adjust the path if necessary
model = YOLO(model_path)

# Get class names
class_names = model.names  # Mapping from class indices to names

# Set up webcam
cap = cv2.VideoCapture(0)  # Open the webcam (index 0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Store the track history
track_history = defaultdict(lambda: [])
selected_object_id = None  # Store the ID of the selected object
mouse_position = None  # Store mouse click position
mouse_position_timer = 0  # Timer for clearing mouse position
movement_direction = "No Object Locked"  # Default movement direction

# Define the center region as 20% of the frame size (adjustable)
CENTER_REGION_RATIO = 0.2


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to select an object by clicking.
    """
    global selected_object_id, mouse_position, track_history, movement_direction, mouse_position_timer
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        mouse_position = (x, y)  # Update mouse position
        mouse_position_timer = 30  # Display mouse position for 30 frames
        selected_object_id = None  # Reset selection
        for track_id, track in track_history.items():
            if track:  # Check if the track has points
                last_x, last_y = track[-1]  # Get the most recent position
                if abs(last_x - x) < 20 and abs(last_y - y) < 20:  # Check proximity
                    selected_object_id = track_id
                    movement_direction = "Stay"  # Reset direction when a new object is selected
                    break


cv2.namedWindow("YOLO Tracking")
cv2.setMouseCallback("YOLO Tracking", mouse_callback)  # Set the callback

try:
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        annotated_frame = frame.copy()  # Initialize annotated frame

        if success:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            center_x, center_y = frame_width // 2, frame_height // 2
            center_w, center_h = int(frame_width * CENTER_REGION_RATIO), int(frame_height * CENTER_REGION_RATIO)
            center_rect = (center_x - center_w, center_y - center_h, center_x + center_w, center_y + center_h)

            # Draw center region for reference
            cv2.rectangle(annotated_frame, (center_rect[0], center_rect[1]), (center_rect[2], center_rect[3]),
                          (255, 255, 0), 2)

            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Get the boxes, track IDs, and labels
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                labels = results[0].boxes.cls.cpu().tolist()  # Class labels
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id, label in zip(boxes, track_ids, labels):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # Retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    if len(track) > 1:
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                    # Highlight the selected object
                    if track_id == selected_object_id:
                        cv2.rectangle(
                            annotated_frame,
                            (int(x - w / 2), int(y - h / 2)),
                            (int(x + w / 2), int(y + h / 2)),
                            color=(0, 0, 255),  # Red color
                            thickness=3,
                        )

                        # Display object ID and label in the top-left corner
                        label_name = class_names[int(label)]  # Convert label index to name
                        label_text = f"ID: {track_id}, Label: {label_name}"
                        cv2.putText(
                            annotated_frame,
                            label_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        # Check if the object is within the center region
                        dx = x - center_x
                        dy = center_y - y  # Reverse y-axis for correct forward/backward logic
                        threshold = 10  # Dead zone threshold
                        directions = []
                        if abs(dx) > threshold:
                            directions.append("Right" if dx > 0 else "Left")
                        if abs(dy) > threshold:
                            directions.append("Forward" if dy > 0 else "Backward")
                        movement_direction = "+".join(directions) if directions else "Stay"

                        # Draw direction arrow and text
                        arrow_start = (frame_width - 120, 50)
                        arrow_end = (arrow_start[0] + (50 if "Right" in directions else -50 if "Left" in directions else 0),
                                     arrow_start[1] - (50 if "Forward" in directions else -50 if "Backward" in directions else 0))
                        cv2.arrowedLine(annotated_frame, arrow_start, arrow_end, (0, 255, 0), 4, tipLength=0.5)
                        cv2.putText(
                            annotated_frame,
                            movement_direction,
                            (frame_width - 220, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

            # If no object is selected or tracking is lost
            if selected_object_id and selected_object_id not in track_history:
                movement_direction = "No Object Locked"
                selected_object_id = None

            # Display mouse position in the bottom-left corner
            if mouse_position_timer > 0:
                cv2.putText(
                    annotated_frame,
                    f"Mouse: {mouse_position}",
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                mouse_position_timer -= 1  # Decrease timer

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

finally:
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
