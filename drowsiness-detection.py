import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import time

from scipy.spatial import distance
from imutils import face_utils
import dlib

# Load model
try:
    model = load_model("drowsiness_eye_cnn_model.h5")
    print("Model drowsiness_eye_cnn_model loaded successfully!")
except:
    try:
        model = load_model("best_model.keras")
        print("Model best_model.keras loaded successfully!")
    except:
        print("Error: Model not found!")
        exit()

# Declare classes
class_names = {0: 'Closed', 1: 'Open'}

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize normalization layer
normalization_layer = layers.Normalization()

# Configure parameters
detection_threshold = 10  # Number of consecutive frames detecting drowsiness to trigger alert
frame_counter = 0
drowsy_status = False

# Variable for FPS tracking
prev_time = time.time()
fps_counter = 0
fps = 0

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
    
def extract_eye_region(eye_points, frame):
    # Create pounding box around the eye points
    (x, y, w, h) = cv2.boundingRect(np.array(eye_points))
    
    # Extend the bounding box to include some padding
    padding = 10
    x_start = max(0, x-padding)
    y_start = max(0, y-padding)
    x_end = min(frame.shape[1], x+w+padding)
    y_end = min(frame.shape[0], y+h+padding)
    
    eye_region = frame[y_start:y_end, x_start:x_end]
    
    # Check if the eye region is valid
    if eye_region.size == 0:
        return None, None, None, None, None
    
    return eye_region, x_start, y_start, x_end-x_start, y_end-y_start

def preprocess_eye(eye_region):
    # Check if eye region is valid
    if eye_region is None or eye_region.size == 0:
        return None
        
    # Change color space from BGR to RGB
    eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    
    # Resize the eye region to the input size of the model (224x224)
    eye_resized = cv2.resize(eye_rgb, (224, 224))
    
    # Normalize the eye region
    normalization_layer.adapt(eye_resized)
    normalized_eye = normalization_layer(eye_resized)
    
    # Add batch dimension
    return np.expand_dims(normalized_eye, axis=0)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = cv2.resize(frame, (224,224))
    normalization_layer.adapt(frame_array)
    frame_array = normalization_layer(frame_array)
    
    return np.expand_dims(frame_array, axis=0)

def get_prediction(preprocessed_frame):
    # Predict on preprocessed frame
    prediction = model.predict(preprocessed_frame, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = class_names[class_idx]
    return class_name, confidence, class_idx

def is_drowsy(class_idx):
    # Determine drowsy state: closed eyes (0) are considered drowsy
    return class_idx == 0

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera")
        break
        
    # Flip frame horizontally for easier front camera usage
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        # Cut and display the eye regions
        left_eye_region, left_x, left_y, left_w, left_h = extract_eye_region(leftEye, frame)
        right_eye_region, right_x, right_y, right_w, right_h = extract_eye_region(rightEye, frame)
        
        # Display eye regions (for debugging)
        if left_eye_region is not None and right_eye_region is not None:
            # Resize
            display_left_eye = cv2.resize(left_eye_region, (100, 100))
            display_right_eye = cv2.resize(right_eye_region, (100, 100))
            
            # Display eye regions on the frame
            frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = display_right_eye
            frame[10:110, frame.shape[1]-220:frame.shape[1]-120] = display_left_eye
            
            # Process and predict for left eye
            processed_left_eye = preprocess_eye(left_eye_region)
            if processed_left_eye is not None:
                left_class_name, left_confidence, left_class_idx = get_prediction(processed_left_eye)
                
                # Display result for left eye
                cv2.putText(frame, f"Left: {left_class_name}", (frame.shape[1]-220, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Process and predict for right eye
            processed_right_eye = preprocess_eye(right_eye_region)
            if processed_right_eye is not None:
                right_class_name, right_confidence, right_class_idx = get_prediction(processed_right_eye)
                
                # Display result for right eye
                cv2.putText(frame, f"Right: {right_class_name}", (frame.shape[1]-110, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Determine drowsiness state based on both eyes
            if processed_left_eye is not None and processed_right_eye is not None:
                # Check if both eyes are closed
                if left_class_idx == 0 or right_class_idx == 0:
                    class_idx = 0
                    class_name = "Closed"
                    confidence = max(left_confidence, right_confidence)
                else:
                    class_idx = 1
                    class_name = "Open" 
                    confidence = (left_confidence + right_confidence) / 2.0
  
    # Calculate FPS
    fps_counter += 1
    if (time.time() - prev_time) > 1.0:
        fps = fps_counter
        fps_counter = 0
        prev_time = time.time()
        
    # Display FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Check drowsy state
    if is_drowsy(class_idx):
        frame_counter += 1
    else:
        frame_counter = max(0, frame_counter - 1)  # Decrease counter gradually
        
    # Display status information
    cv2.putText(frame, f"Status: {class_name} ({confidence:.2f})", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    # Display drowsiness status bar
    drowsiness_bar_length = int((frame_counter / detection_threshold) * 100)
    drowsiness_bar_length = min(100, drowsiness_bar_length)
    cv2.rectangle(frame, (10, 90), (10 + 100, 110), (0, 0, 255), 1)
    cv2.rectangle(frame, (10, 90), (10 + drowsiness_bar_length, 110), (0, 0, 255), -1)
    
    # Trigger alert when drowsiness detected
    if frame_counter >= detection_threshold:
        # Display alert on frame
        cv2.putText(frame, "DROWSINESS ALERT!", (frame.shape[1]//2 - 140, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        drowsy_status = True
    else:
        drowsy_status = False
    
    # Show frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
