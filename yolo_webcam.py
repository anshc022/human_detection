import os
import time
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load the YOLOv8 model
model_yolo = YOLO('yolov8n.pt')  # Ensure you have this model file in your working directory

# Load the age and gender prediction models
age_proto = 'age_deploy.prototxt'
age_model = 'age_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'
gender_model = 'gender_net.caffemodel'

age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

def classify_age_gender(face_img):
    # Preprocess the face image for Caffe models
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104.0, 117.0, 123.0))  # Mean values might differ
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    
    # Gender prediction
    gender = 'Male' if gender_preds[0][0] > gender_preds[0][1] else 'Female'
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    
    # Age prediction
    age = int(age_preds[0].argmax())
    
    return gender, age

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('output_videos'):
        os.makedirs('output_videos')

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple webcams

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # To store previous positions for drawing paths
    path_points = []
    last_path_update_time = time.time()

    # Store previous frame's person positions for activity detection
    previous_person_centers = {}

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform object detection
        results = model_yolo(frame)
        
        # Initialize counters
        person_count = 0
        gender_count = {'Male': 0, 'Female': 0}

        # Get detection results
        boxes = results[0].boxes  # Access detection boxes

        # Process results and render the frame with annotations
        annotated_frame = results[0].plot()
        
        # Track the positions of detected persons
        current_person_centers = {}

        # Set to track already processed faces
        processed_faces = set()

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

                # Extract face image
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # Generate a unique key for the face image (e.g., by its bounding box coordinates)
                    face_key = (x1, y1, x2, y2)
                    
                    if face_key not in processed_faces:
                        # Classify age and gender
                        gender, age = classify_age_gender(face_img)
                        gender_count[gender] += 1
                        
                        # Mark face as processed
                        processed_faces.add(face_key)

                # Draw bounding box and label on the frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label}, {gender}, {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Compare current positions with previous positions to detect activity
        activity = "None"  # Default activity
        for box, (prev_x, prev_y) in previous_person_centers.items():
            if box in current_person_centers:
                curr_x, curr_y = current_person_centers[box]
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Estimate speed and classify activity
                speed = distance
                if speed < 5:  # Threshold for low speed
                    activity = 'sitting'
                elif speed < 20:  # Threshold for medium speed
                    activity = 'walking'
                else:  # High speed
                    activity = 'running'
                
                # Draw activity text
                cv2.putText(annotated_frame, activity, (curr_x, curr_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update path points with current positions
        path_points.extend(current_person_centers.values())

        # Draw the path (a simple line for demonstration)
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                cv2.line(annotated_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)
        
        # Display counts on the frame
        cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Male: {gender_count["Male"]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Female: {gender_count["Female"]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the resulting frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Check if 1 second has passed to update path points
        current_time = time.time()
        if current_time - last_path_update_time >= 1.0:
            # Clear path points and update timestamp
            path_points = []
            last_path_update_time = current_time

        # Update previous positions for next frame
        previous_person_centers = current_person_centers

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
