import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model_yolo = YOLO('yolov8n.pt')

# Load the age and gender prediction models
age_proto = 'age_deploy.prototxt'
age_model = 'age_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'
gender_model = 'gender_net.caffemodel'

age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

def classify_age_gender(face_img):
    """Classify the age and gender from the face image."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104.0, 117.0, 123.0))
    
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = 'Male' if gender_preds[0][0] > gender_preds[0][1] else 'Female'
    
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = int(age_preds[0].argmax())
    
    return gender, age

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(0, 255, 0), thickness=2, bg_color=(0, 0, 0), bg_padding=5):
    """Draw text with background on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    top_left = (position[0] - bg_padding, position[1] - text_height - bg_padding)
    bottom_right = (position[0] + text_width + bg_padding, position[1] + baseline + bg_padding)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, cv2.FILLED)
    
    # Draw text on top of the rectangle
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    """Main function to run the video processing."""
    if not os.path.exists('output_videos'):
        os.makedirs('output_videos')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get screen resolution
    screen_res = (1920, 1080)  # Replace with your screen resolution if necessary
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_res[1])

    # Create output video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/output.avi', fourcc, 20.0, screen_res)

    path_points = []
    last_path_update_time = time.time()
    previous_person_centers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        results = model_yolo(frame)
        boxes = results[0].boxes

        person_count = 0
        gender_count = {'Male': 0, 'Female': 0}
        annotated_frame = results[0].plot()
        current_person_centers = {}
        processed_faces = set()

        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            label = results[0].names[int(box.cls[0])]
            
            if label == 'person':
                person_count += 1
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                current_person_centers[box] = (center_x, center_y)

                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    face_key = (x1, y1, x2, y2)
                    
                    if face_key not in processed_faces:
                        gender, age = classify_age_gender(face_img)
                        gender_count[gender] += 1
                        processed_faces.add(face_key)

                        # Draw bounding box around the detected person
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Calculate the position to the right of the bounding box for age and gender information
                        text_x = x2 + 10
                        text_y = y1 + 20

                        draw_text_with_background(annotated_frame, f'Age: {age}', (text_x, text_y), font_scale=0.7, color=(0, 255, 0), bg_color=(0, 0, 0))
                        draw_text_with_background(annotated_frame, f'Gender: {gender}', (text_x, text_y + 30), font_scale=0.7, color=(0, 255, 0), bg_color=(0, 0, 0))

        activity = "None"
        for box, (prev_x, prev_y) in previous_person_centers.items():
            if box in current_person_centers:
                curr_x, curr_y = current_person_centers[box]
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = np.sqrt(dx**2 + dy**2)
                speed = distance
                
                if speed < 5:
                    activity = 'Sitting'
                elif speed < 20:
                    activity = 'Walking'
                else:
                    activity = 'Running'
                
                draw_text_with_background(annotated_frame, activity, (curr_x, curr_y - 20), font_scale=0.7, color=(255, 0, 0), bg_color=(0, 0, 0))

        path_points.extend(current_person_centers.values())
        
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                cv2.line(annotated_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)
        
        draw_text_with_background(annotated_frame, f'Persons: {person_count}', (10, 30), font_scale=1, color=(0, 255, 0), bg_color=(0, 0, 0))
        draw_text_with_background(annotated_frame, f'Male: {gender_count["Male"]}', (10, 70), font_scale=1, color=(0, 255, 0), bg_color=(0, 0, 0))
        draw_text_with_background(annotated_frame, f'Female: {gender_count["Female"]}', (10, 110), font_scale=1, color=(0, 255, 0), bg_color=(0, 0, 0))
        
        out.write(annotated_frame)

        # Display the resulting frame in fullscreen mode
        cv2.namedWindow('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        current_time = time.time()
        if current_time - last_path_update_time >= 1.0:
            path_points = []
            last_path_update_time = current_time

        previous_person_centers = current_person_centers

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
