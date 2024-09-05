import cv2
from ultralytics import YOLO
import os
import numpy as np
import tkinter as tk
from tkinter import Label
import threading
import time

# Global variables for dashboard updates and activity tracking
person_count = 0
activity = ""
prev_person_centers = {}

def process_frame(frame):
    global person_count, activity, prev_person_centers

    # Perform object detection
    results = model(frame)
    
    # Initialize counters
    person_count = 0
    
    # Get detection results
    boxes = results[0].boxes  # Access detection boxes

    # Process results and render the frame with annotations
    annotated_frame = results[0].plot()

    # Track the positions of detected persons
    current_person_centers = {}

    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]  # Convert to integers
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID
        
        label = results[0].names[cls]  # Get the label for the class ID
        
        # Count detections and track positions
        if label == 'person':
            person_count += 1
            # Calculate the center of the bounding box for tracking
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_person_centers[box] = (center_x, center_y)

            # Draw bounding box and label on the frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update activity based on movement (Example logic, adjust as needed)
    activity = "None"
    for box, (center_x, center_y) in current_person_centers.items():
        if box in prev_person_centers:
            prev_center_x, prev_center_y = prev_person_centers[box]
            speed = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
            if speed < 5:
                activity = 'Sitting'
            elif speed < 20:
                activity = 'Walking'
            else:
                activity = 'Running'
    
    # Update previous positions
    prev_person_centers = current_person_centers

    return annotated_frame

def update_dashboard():
    global person_count, activity

    def update():
        person_count_label.config(text=f"Persons Detected: {person_count}")
        activity_label.config(text=f"Activity: {activity}")
        root.after(1000, update)  # Update every second

    update()

def main():
    global model

    # Create output directory if it doesn't exist
    if not os.path.exists('output_videos'):
        os.makedirs('output_videos')

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

    # Initialize the Tkinter dashboard
    global person_count_label, activity_label
    root = tk.Tk()
    root.title("Activity Dashboard")

    person_count_label = Label(root, text="Persons Detected: 0", font=("Arial", 16))
    person_count_label.pack(pady=10)

    activity_label = Label(root, text="Activity: None", font=("Arial", 16))
    activity_label.pack(pady=10)

    # Start dashboard update
    update_dashboard()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process frame and get annotated frame
        annotated_frame = process_frame(frame)

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
    root.mainloop()

if __name__ == "__main__":
    main()
