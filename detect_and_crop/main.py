import cv2
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
MODEL_FILE = 'face_detector.tflite'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
SAVE_DIR = "dataset/23520688" # Thu muc luu anh theo MSSV cua Khang

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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
count = 0 

print("He thong da san sang!")
print("- Nhan 's' de luu anh khuon mat vao dataset.")
print("- Nhan 'ESC' de thoat.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    face_to_save = None

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            # 1. Lấy thông số gốc
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            img_h, img_w, _ = frame.shape

            # 2. Định nghĩa độ mở rộng (Padding)
            # Tăng padding_top để lấy hết tóc, giảm padding_bottom để bớt vai
            padding_top = int(h * 0.5)     # Mở rộng mạnh lên phía trên (tóc)
            padding_bottom = int(h * 0.1)  # Mở rộng rất ít xuống dưới (hạn chế vai)
            padding_side = int(w * 0.2)    # Mở rộng vừa phải sang hai bên

            # 3. Tính toán tọa độ mới
            new_x = x - padding_side
            new_y = y - padding_top
            new_w = w + (2 * padding_side)
            new_h = h + padding_top + padding_bottom

            # 4. Giới hạn không để khung văng ra ngoài ảnh
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(new_w, img_w - new_x)
            new_h = min(new_h, img_h - new_y)

            # 5. Cắt và hiển thị
            face_to_save = frame[new_y:new_y+new_h, new_x:new_x+new_w]
            cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

            
            if face_to_save is not None and face_to_save.size > 0:
                cv2.imshow('Khuon mat dang crop', face_to_save)

    cv2.imshow('He thong Diem danh UIT', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == ord('s'): 
        if face_to_save is not None and face_to_save.size > 0:
            count += 1
            file_name = f"{SAVE_DIR}/khang_{count}.jpg"
            cv2.imwrite(file_name, face_to_save)
            print(f"Da luu anh thu {count} vao: {file_name}")

cap.release()
cv2.destroyAllWindows()
detector.close()