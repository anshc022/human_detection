import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Ensure you have this model file in your working directory

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple webcams

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform object detection
        results = model(frame)

        # Process results and render the frame with annotations
        annotated_frame = results[0].plot()  # Use plot() to draw boxes and labels

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the resulting frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
