import cv2
import mediapipe as mp
import os
import urllib.request
import time
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
SERVER_URL = "http://<ĐỊA_CHỈ_IP_SERVER_CỦA_BẠN>:8001/api/attendance/process"
MODEL_FILE = 'face_detector.tflite'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

# Điều kiện cấu hình
MIN_FACE_AREA_RATIO = 0.15  # Tỷ lệ khuôn mặt
HOLD_TIME_SECONDS = 1.0    # Giữ mặt trong 1 giây

# --- TU DONG TAI MODEL ---
if not os.path.exists(MODEL_FILE):
    print("--- Dang tai model... Vui long doi ---")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("--- Tai xong! ---")

# --- KHOI TAO DETECTOR ---
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

print("He thong tu dong diem danh da san sang!")
print("- Nhan 'ESC' de thoat.")

# --- CÁC BIẾN QUẢN LÝ TRẠNG THÁI ---
face_start_time = None
face_processed = False

def is_looking_straight(keypoints):
    """Hàm kiểm tra mặt nhìn trực diện dựa vào mắt và mũi"""
    if not keypoints or len(keypoints) < 3:
        return False
    # index 0: Right Eye, 1: Left Eye, 2: Nose Tip
    right_eye = keypoints[0]
    left_eye = keypoints[1]
    nose = keypoints[2]
    
    eye_center_x = (right_eye.x + left_eye.x) / 2.0
    eye_dist = abs(right_eye.x - left_eye.x)
    
    # Nếu mũi nằm giữa 2 mắt (lệch tối đa 25% khoảng cách 2 mắt) là trực diện
    return abs(nose.x - eye_center_x) < (eye_dist * 0.25)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    img_h, img_w, _ = frame.shape
    frame_area = img_w * img_h

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    has_valid_face_in_current_frame = False
    face_to_send = None

    if detection_result.detections:
        # Lấy khuôn mặt to nhất trong khung hình để xét
        largest_detection = max(
            detection_result.detections, 
            key=lambda d: d.bounding_box.width * d.bounding_box.height
        )
        
        bbox = largest_detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        face_area = w * h

        # 1. Kiểm tra điều kiện diện tích (> 70%)
        if (face_area / frame_area) >= MIN_FACE_AREA_RATIO:
            # 2. Kiểm tra điều kiện nhìn trực diện
            if is_looking_straight(largest_detection.keypoints):
                has_valid_face_in_current_frame = True
                
                # Cắt ảnh với padding như cũ
                pad_t, pad_b, pad_s = int(h * 0.5), int(h * 0.1), int(w * 0.2)
                nx, ny = max(0, x - pad_s), max(0, y - pad_t)
                nw, nh = min(img_w - nx, w + 2*pad_s), min(img_h - ny, h + pad_t + pad_b)
                face_to_send = frame[ny:ny+nh, nx:nx+nw]
                
                # Vẽ khung màu xanh dương báo hiệu đạt chuẩn
                cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 2)
                cv2.putText(frame, "Hop le! Giu nguyen...", (nx, ny-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Vui long nhin thang", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Tien lai gan hon", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # --- LOGIC THỜI GIAN & GỬI SERVER ---
    if has_valid_face_in_current_frame:
        if face_start_time is None:
            # Bắt đầu đếm thời gian
            face_start_time = time.time()
        else:
            elapsed_time = time.time() - face_start_time
            if elapsed_time >= HOLD_TIME_SECONDS and not face_processed:
                print(">>> Dieu kien thoa man. Dang gui len server...")
                
                # Mã hóa ảnh để gửi thẳng qua RAM, không cần lưu xuống ổ cứng Ras
                _, img_encoded = cv2.imencode('.jpg', face_to_send)
                files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
                
                try:
                    # Gửi POST request (Code sẽ block/đợi tại đây cho đến khi có phản hồi)
                    res = requests.post(SERVER_URL, files=files, timeout=10)
                    print("<<< Server phan hoi:", res.json())
                except Exception as e:
                    print("Loi ket noi server:", e)
                
                # Đánh dấu là đã xử lý xong khuôn mặt này, không gửi lại nữa
                face_processed = True
    else:
        # Nếu khuôn mặt bị mất, xoay đi chỗ khác, hoặc lùi ra xa -> Reset toàn bộ
        face_start_time = None
        face_processed = False

    cv2.imshow('He thong Diem danh UIT', frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
detector.close()