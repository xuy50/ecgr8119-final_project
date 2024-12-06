import cv2
from ultralytics import YOLO

def run_webcam():
    # Load the trained YOLO model
    model_path = 'runs/detect/train_0/weights/best.pt'  # Adjust the path if necessary
    model = YOLO(model_path)

    # Set up webcam
    cap = cv2.VideoCapture(0)  # Open the webcam (index 0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Run inference
        results = model.predict(frame)
        # results = model.track(frame, show=True)

        # Get the first result
        result = results[0]

        # Plot the detections on the frame
        annotated_frame = result.plot()

        # Display the frame with detections
        cv2.imshow('YOLO Webcam', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
