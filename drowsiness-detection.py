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

# Load saved normalization stats (mean, variance) produced in [train.py](http://_vscodecontentref_/2)
try:
    norm_mean = np.load("norm_mean.npy")
    norm_var  = np.load("norm_var.npy")
    # clean up shapes: want either (3,) or (1,1,3)
    norm_mean = np.asarray(norm_mean)
    norm_var  = np.asarray(norm_var)

    # remove accidental extra dims
    norm_mean = np.squeeze(norm_mean)
    norm_var  = np.squeeze(norm_var)

    # final reshape to (1,1,3) for broadcasting convenience
    if norm_mean.ndim == 1 and norm_mean.size == 3:
        norm_mean = norm_mean.reshape((1,1,3))
    if norm_var.ndim == 1 and norm_var.size == 3:
        norm_var = norm_var.reshape((1,1,3))

    norm_std = np.sqrt(norm_var + 1e-8)
    print("Loaded normalization stats:", norm_mean.shape)
except Exception as e:
    print("Warning: could not load normalization stats, falling back to /255.0", e)
    norm_mean = None
    norm_std = None

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
    if eye_region is None or eye_region.size == 0:
        return None
    
    eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    eye_resized = cv2.resize(eye_rgb, (224, 224)).astype(np.float32)
    
    # Áp dụng chuẩn hóa đã học, KHÔNG adapt lại
    if norm_mean is not None and norm_std is not None:
        normalized_eye = (eye_resized - norm_mean) / norm_std
    else:
        # Fallback nếu không có file norm
        normalized_eye = eye_resized / 255.0
    
    return np.expand_dims(normalized_eye, axis=0)

def preprocess_frame(frame):
    # HÀM NÀY KHÔNG ĐƯỢC SỬ DỤNG VÀ CÓ LOGIC SAI.
    # CÓ THỂ XÓA ĐI ĐỂ TRÁNH NHẦM LẪN.
    # Hoặc sửa lại cho đúng nếu sau này cần dùng.
    # Tạm thời thầy sẽ comment nó ra.
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_array = cv2.resize(frame, (224,224)).astype(np.float32)
    # if norm_mean is not None and norm_std is not None:
    #     frame_array = (frame_array - norm_mean) / norm_std
    # else:
    #     frame_array = frame_array / 255.0
    # return np.expand_dims(frame_array, axis=0)
    pass # Không làm gì cả

def get_prediction(preprocessed_frame):
    # --- PHẦN SỬA LỖI QUAN TRỌNG ---
    # Đảm bảo dữ liệu đầu vào là một mảng numpy
    arr = np.asarray(preprocessed_frame, dtype=np.float32)

    # Vòng lặp phòng thủ: loại bỏ các chiều không cần thiết (kích thước 1)
    while arr.ndim > 4:
        arr = np.squeeze(arr, axis=1)

    # Đảm bảo có chiều batch: nếu là (224, 224, 3) thì thêm chiều batch
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    # Kiểm tra cuối cùng trước khi dự đoán
    if arr.shape != (1, 224, 224, 3):
        print(f"Error: Invalid shape before prediction: {arr.shape}")
        # Trả về giá trị mặc định an toàn
        return "Error", 0.0, -1, np.array([0.0, 1.0]) 

    # Predict on preprocessed frame
    prediction = model.predict(arr, verbose=0)
    probs = prediction[0] # Lấy vector xác suất
    class_idx = np.argmax(probs)
    confidence = float(probs[class_idx])
    class_name = class_names[class_idx]
    
    # Trả về các biến ĐÚNG đã được tính toán trong hàm này
    return class_name, confidence, class_idx, probs

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

    # Khởi tạo giá trị mặc định cho mỗi frame
    class_idx = 1  # Mặc định là "Open"
    class_name = "Open"
    confidence = 1.0
        
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
            
            # Khởi tạo giá trị mặc định an toàn cho mỗi frame
            left_class_idx, right_class_idx = 1, 1 
            left_confidence, right_confidence = 0.0, 0.0
            left_probs = right_probs = np.array([0.0, 1.0]) # Mặc định là Open

            # Process and predict for left eye
            processed_left_eye = preprocess_eye(left_eye_region)
            if processed_left_eye is not None:
                left_class_name, left_confidence, left_class_idx, left_probs = get_prediction(processed_left_eye)
                cv2.putText(frame, f"Left: {left_class_name} ({left_confidence:.2f})", (frame.shape[1]-220, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Process and predict for right eye
            processed_right_eye = preprocess_eye(right_eye_region)
            if processed_right_eye is not None:
                right_class_name, right_confidence, right_class_idx, right_probs = get_prediction(processed_right_eye)
                cv2.putText(frame, f"Right: {right_class_name} ({right_confidence:.2f})", (frame.shape[1]-110, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # --- LOGIC QUYẾT ĐỊNH MỚI, CHẶT CHẼ HƠN ---
            CONF_THRESHOLD = 0.80 # Ngưỡng tin cậy cao để xác nhận mắt đóng

            # Điều kiện 1: Cả hai mắt đều được dự đoán là "Closed" (AND logic)
            both_closed = (left_class_idx == 0 and right_class_idx == 0)
            
            # Điều kiện 2: Một mắt được dự đoán là "Closed" với độ tin cậy RẤT CAO
            left_strong_closed = (left_class_idx == 0 and left_confidence >= CONF_THRESHOLD)
            right_strong_closed = (right_class_idx == 0 and right_confidence >= CONF_THRESHOLD)

            if both_closed or left_strong_closed or right_strong_closed:
                class_idx = 0
                class_name = "Closed"
                # Lấy confidence của dự đoán mạnh nhất cho trạng thái "Closed"
                confidence = max(left_probs[0], right_probs[0]) 
            else:
                class_idx = 1
                class_name = "Open" 
                # Lấy trung bình confidence cho trạng thái "Open"
                confidence = (left_probs[1] + right_probs[1]) / 2.0
  
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
