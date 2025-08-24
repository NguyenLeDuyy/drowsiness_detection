from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25 # Ngưỡng để xác định nhắm mắt, nếu EAR < thresh thì mắt được coi là nhắm
frame_check = 20 # Số khung hình liên tiếp để xác định nhắm mắt, nếu nhắm liên tục trong frame_check khung hình thì sẽ cảnh báo
detect = dlib.get_frontal_face_detector() # Khởi tạo bộ phát hiện khuôn mặt
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") # Khởi tạo bộ dự đoán các điểm mốc trên khuôn mặt

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] # Lấy chỉ số của các điểm mốc mắt trái
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] # Lấy chỉ số của các điểm mốc mắt phải
cap=cv2.VideoCapture(0) # Mở camera
if not cap.isOpened():
	print("Cannot open camera")
	exit()
flag=0 # Biến đếm số khung hình nhắm mắt liên tiếp
# Biến flag sẽ tăng lên mỗi khi mắt nhắm và reset về 0 khi mắt mở
# Nếu flag >= frame_check thì sẽ cảnh báo nhắm mắt
# Bắt đầu vòng lặp để đọc từng khung hình từ camera
while True:
	ret, frame=cap.read() # Đọc khung hình từ camera
	frame = imutils.resize(frame, width=450) # Thay đổi kích thước khung hình để xử lý nhanh hơn
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Chuyển đổi khung hình sang ảnh xám để phát hiện khuôn mặt
	subjects = detect(gray, 0) 
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye) # Tạo đường bao lồi cho mắt trái
		rightEyeHull = cv2.convexHull(rightEye) # Tạo đường bao lồi cho mắt phải
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # Vẽ đường bao lồi cho mắt trái
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) # Vẽ đường bao lồi cho mắt phải
		if ear < thresh: #
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
