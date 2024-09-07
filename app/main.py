from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)

# Load YOLOv8 model
model_yolo = YOLO('yolov8n.pt')

# Load the age and gender prediction models
age_proto = 'age_deploy.prototxt'
age_model = 'age_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'
gender_model = 'gender_net.caffemodel'

age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Initialize Mediapipe Pose and Drawing utilities
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

body_parts = [
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index",
    "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
    "Right Heel", "Left Foot Index", "Right Foot Index"
]

body_indices = [11, 12]

# Global variables
person_count = 0
gender_count = {'Male': 0, 'Female': 0}
movement_detected = 'No Movement'
head_bounding_box = 'Not Detected'

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

def generate_frames():
    """Generate frames for video feed."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return

    global person_count, gender_count, movement_detected, head_bounding_box
    prev_body_positions = {i: (None, None) for i in body_indices}
    movement_threshold = 10  # Minimum movement to detect change (tweak as needed)

    detected_person_boxes = set()  # Track detected person bounding boxes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Object Detection
        results = model_yolo(frame)
        boxes = results[0].boxes

        annotated_frame = results[0].plot()
        current_person_centers = {}
        processed_faces = set()

        new_person_boxes = set()

        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            label = results[0].names[int(box.cls[0])]
            
            if label == 'person':
                new_person_box = (x1, y1, x2, y2)
                new_person_boxes.add(new_person_box)

                # Check if this person box is new
                if new_person_box not in detected_person_boxes:
                    # Increment the person count and mark this box as detected
                    person_count += 1
                    detected_person_boxes.add(new_person_box)
                    current_person_centers[new_person_box] = ((x1 + x2) // 2, (y1 + y2) // 2)
                                    
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

        # Update detected_person_boxes to include new person boxes
        detected_person_boxes.update(new_person_boxes)

        # Mediapipe Pose Detection
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(imgRGB)

        if pose_results.pose_landmarks:
            # Draw landmarks and labels for all body parts except face
            mpDraw.draw_landmarks(annotated_frame, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            current_landmarks = {}

            for id, lm in enumerate(pose_results.pose_landmarks.landmark):
                h, w, c = annotated_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                current_landmarks[id] = (cx, cy)
                cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # Label body parts
                if id >= 11:  # Starting index of non-face parts
                    if id < len(body_parts) + 11:  # Adjust according to the list length
                        cv2.putText(annotated_frame, body_parts[id - 11], (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw a bounding box and label for the head region
            h, w, c = annotated_frame.shape
            x_min = int(min(pose_results.pose_landmarks.landmark[i].x for i in body_indices) * w)
            x_max = int(max(pose_results.pose_landmarks.landmark[i].x for i in body_indices) * w)
            y_min = int(min(pose_results.pose_landmarks.landmark[i].y for i in body_indices) * h)
            y_max = int(max(pose_results.pose_landmarks.landmark[i].y for i in body_indices) * h)
            head_bounding_box = (x_min, y_min, x_max, y_max)
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Head", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect significant movement in shoulder positions
            movement_detected = 'No Movement'
            for idx in body_indices:
                if idx in current_landmarks:
                    cx, cy = current_landmarks[idx]
                    prev_cx, prev_cy = prev_body_positions[idx]

                    if prev_cx is not None and prev_cy is not None:
                        dx, dy = cx - prev_cx, cy - prev_cy
                        if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
                            movement_detected = f'Movement Detected ({idx})'
                            # Draw movement detection text on the right side
                            text = f'Movement Detected ({idx})'
                            position = (annotated_frame.shape[1] - 200, 70 + (idx - body_indices[0]) * 20)
                            draw_text_with_background(annotated_frame, text, position, font_scale=0.7, color=(0, 255, 0), bg_color=(0, 0, 0))

                    prev_body_positions[idx] = (cx, cy)

        # Draw person count
        cv2.putText(annotated_frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Gender Count: Male: {gender_count["Male"]}, Female: {gender_count["Female"]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Movement: {movement_detected}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Head Bounding Box: {head_bounding_box}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert frame to JPEG and yield
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_dashboard_data')
def get_dashboard_data():
    global person_count, gender_count, movement_detected, head_bounding_box

    # Ensure the data is valid before returning
    if 'Male' not in gender_count:
        gender_count['Male'] = 0
    if 'Female' not in gender_count:
        gender_count['Female'] = 0
    if not person_count:
        person_count = 0
    if not movement_detected:
        movement_detected = 'No Movement'
    if not head_bounding_box:
        head_bounding_box = 'Not Detected'

    return jsonify({
        'person_count': person_count,
        'gender_count': gender_count,
        'movement_detected': movement_detected,
        'head_bounding_box': head_bounding_box
    })

if __name__ == '__main__':
    app.run(debug=True)
