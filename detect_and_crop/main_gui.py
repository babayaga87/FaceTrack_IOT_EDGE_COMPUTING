import cv2
import os
import urllib.request
import time
import requests
import threading
import numpy as np                     # Sử dụng ma trận phẳng numpy cố định
import tkinter as tk                       
from tkinter import simpledialog           
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# ===== USB CAMERA CLASS (BỘ ĐỌC GỐC CHẠY ỔN ĐỊNH CỦA BẠN) =====
# =====================================================================
class UsbCamera:
    def __init__(self):
        self.capture = None
        self.last_error = ""

    def apply_capture_options(self, width, height):
        if self.capture is None or not self.capture.isOpened():
            return False
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def open_device(self, device=0, width=640, height=480):
        self.close_device()
        self.last_error = ""
        try:
            self.capture = cv2.VideoCapture(device)
            if not self.capture.isOpened():
                self.last_error = "Cannot open camera"
                return False
            if not self.apply_capture_options(width, height):
                self.last_error = "Failed to apply options"
                self.close_device()
                return False
            frame = None
            for _ in range(5):
                ret, frame = self.capture.read()
            if frame is None:
                self.last_error = "Warmup frame empty"
                self.close_device()
                return False
        except Exception as e:
            self.last_error = f"Exception: {e}"
            self.close_device()
            return False
        return True

    def close_device(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def is_open(self):
        return self.capture is not None and self.capture.isOpened()

    def read_frame(self):
        if not self.is_open():
            self.last_error = "Camera not opened"
            return False, None
        try:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.last_error = "Empty frame"
                return False, None
            return True, frame
        except Exception as e:
            self.last_error = f"Exception: {e}"
            return False, None

    def get_last_error(self):
        return self.last_error

# =====================================================================
# ===== CẤU HÌNH HỆ THỐNG API =====
# =====================================================================
SERVER_URL = "http://10.152.19.232:8001/api/attendance/process" 
REGISTER_URL = "http://10.152.19.232:8000/api/students"          
MODEL_FILE = 'face_detector.tflite'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

MIN_FACE_AREA_RATIO = 0.05  
HOLD_TIME_SECONDS = 1.5     

face_start_time = None
face_processed = False
is_sending = False  
student_name = ""
student_id = ""

REG_ID = ""                  
REG_NAME = ""                
captured_images_buffer = []  
is_registering = False      
register_message = ""       

def is_looking_straight(keypoints):
    if not keypoints or len(keypoints) < 3: return False
    right_eye, left_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_center_x = (right_eye.x + left_eye.x) / 2.0
    eye_dist = abs(right_eye.x - left_eye.x)
    return abs(nose.x - eye_center_x) < (eye_dist * 0.25)

def send_to_server_async(image_data):
    global face_processed, is_sending, student_name, student_id
    is_sending = True
    _, img_encoded = cv2.imencode('.jpg', image_data)
    files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        res = requests.post(SERVER_URL, files=files, timeout=10)
        data = res.json()
        if "full_name" in data:
            student_name = data.get("full_name")
        elif "message" in data and "Điểm danh thành công: " in data["message"]:
            student_name = data["message"].split("Điểm danh thành công: ")[1]
        else:
            student_name = "Unknown"
        student_id = data.get("student_id") or data.get("studentId") or data.get("id") or "Unknown"
    except:
        student_name = "Server Error"
        student_id = ""
    is_sending = False
    face_processed = True 

def upload_registration_packet_async():
    global is_registering, register_message, captured_images_buffer, REG_ID, REG_NAME
    is_registering = True
    register_message = "Analyzing face..."
    
    payload = {'full_name': REG_NAME, 'student_id': REG_ID}
    files = []
    for i, img_bytes in enumerate(captured_images_buffer):
        files.append(('images', (f"face_{i}.jpg", img_bytes, 'image/jpeg')))
        
    try:
        res = requests.post(REGISTER_URL, data=payload, files=files, timeout=25)
        if res.status_code == 200:
            register_message = "Success!"
            REG_ID = ""
            REG_NAME = ""
            captured_images_buffer = []
        else:
            try: err_detail = res.json().get('detail', 'Tu choi')
            except: err_detail = "DB Error"
            register_message = f"Error: {err_detail}"
            captured_images_buffer = [] 
    except:
        register_message = "Error: Network Error"
        captured_images_buffer = []
    is_registering = False

def draw_beautiful_box(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# =====================================================================
# ===== CHƯƠNG TRÌNH CHÍNH =====
# =====================================================================
def main():
    global face_start_time, face_processed, is_sending, student_name, student_id
    global REG_ID, REG_NAME, captured_images_buffer, is_registering, register_message

    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    cam = UsbCamera()
    if not cam.open_device(0, 640, 480):
        print("Cannot open camera:", cam.get_last_error())
        return

    # 💡 CẤU HÌNH FIX CỨNG THEO ĐỘ PHÂN GIẢI PHẦN CỨNG 800x480
    win_w = 800
    win_h = 480
    cam_w = 480  # Camera dạng vuông chiếm lề trái (480x480)
    panel_w = 320 # Cột thông tin chiếm lề phải (320x480)

    window_name = 'UIT Checkin System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Đẩy tràn viền màn hình Ras

    while True:
        success, frame = cam.read_frame()
        if not success or frame is None: break

        # 💡 FIX LẬT CAMERA TRÁI PHẢI (HIỆU ỨNG GƯƠNG) TẠI ĐÂY
        frame = cv2.flip(frame, 1)

        # 1. TẠO KHUNG NỀN MÀU XÁM TỐI VỪA KHÍT 800x480 PIXELS
        interface = np.zeros((win_h, win_w, 3), dtype=np.uint8) + 28 

        # 2. XỬ LÝ ẢNH CAMERA: Resize và cắt gọn thành hình vuông 480x480 đưa vào bên trái
        frame_resized = cv2.resize(frame, (640, 480))
        frame_square = frame_resized[0:480, 80:560] # Cắt bớt lề thừa 2 bên để giữ tỉ lệ 1:1 không móp hình
        interface[0:win_h, 0:cam_w] = frame_square

        frame_area = cam_w * win_h

        # Nhận diện khuôn mặt trên vùng ảnh vuông camera
        rgb = cv2.cvtColor(frame_square, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        has_valid_face_in_current_frame = False
        face_to_send = None
        progress_ratio = 0.0

        if result.detections:
            largest_detection = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = largest_detection.bounding_box
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            
            padding_top, padding_bottom, padding_side = int(h * 0.45), int(h * 0.1), int(w * 0.18)    
            new_x = max(0, x - padding_side)
            new_y = max(0, y - padding_top)
            new_w = min(w + (2 * padding_side), cam_w - new_x)
            new_h = min(h + padding_top + padding_bottom, win_h - new_y)
            
            # Tính toán scale ngược lại để trích xuất ảnh gốc chất lượng cao gửi Server
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            real_x = int((new_x + 80) * scale_x)
            real_y = int(new_y * scale_y)
            face_to_send = frame[real_y:int((new_y+new_h)*scale_y), real_x:int((new_x+new_w+80)*scale_x)]

            if (w * h / frame_area) >= MIN_FACE_AREA_RATIO:
                if is_looking_straight(largest_detection.keypoints):
                    has_valid_face_in_current_frame = True
                    if len(captured_images_buffer) > 0 or is_registering: color = (255, 0, 255)
                    elif is_sending: color = (0, 165, 255)
                    elif face_processed: color = (0, 255, 0)
                    else: color = (255, 255, 0)
                    draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), color, 2, 12, 8)
                else:
                    draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2, 12, 8)

        # Logic tính tiến trình điểm danh tự động
        if has_valid_face_in_current_frame and len(captured_images_buffer) == 0:
            if not face_processed and not is_sending:
                if face_start_time is None: face_start_time = time.time()
                else:
                    dt = time.time() - face_start_time
                    progress_ratio = min(1.0, dt / HOLD_TIME_SECONDS)
                    if dt >= HOLD_TIME_SECONDS:
                        threading.Thread(target=send_to_server_async, args=(face_to_send,)).start()
        elif not has_valid_face_in_current_frame:
            face_start_time = None
            face_processed = False

        # Vẽ thanh tải Progress Bar dưới đáy khung ảnh camera (y = 468)
        bar_y = win_h - 12
        cv2.rectangle(interface, (15, bar_y), (cam_w - 15, bar_y + 4), (50, 50, 50), -1)
        if progress_ratio > 0:
            cv2.rectangle(interface, (15, bar_y), (15 + int((cam_w - 30) * progress_ratio), bar_y + 4), (0, 255, 0), -1)

        # =====================================================================
        # 💡 THIẾT KẾ CỘT SIDE PANEL BÊN PHẢI (CHIỀU NGANG TỪ 480 ĐẾN 800)
        # =====================================================================
        px = cam_w + 20    # Điểm lề trái bắt đầu của Panel chữ (Pixel 500)
        font_scale = 0.52  # Kích thước font chữ cực kỳ nét cho màn hình 800x480
        spacing_y = 48     # Dãn dòng cách đều 48px

        # Dòng 1: Tiêu đề
        curr_y = 45
        cv2.putText(interface, "UIT Checkin System", (px, curr_y), cv2.FONT_HERSHEY_DUPLEX, font_scale * 1.1, (255, 255, 255), 1)
        cv2.line(interface, (px, curr_y + 12), (win_w - 20, curr_y + 12), (80, 80, 80), 1)
        
        # Dòng 2: Nhật ký điểm danh
        curr_y += spacing_y + 15
        cv2.putText(interface, "STATUS LOG:", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (150, 150, 150), 1)
        curr_y += spacing_y - 10
        if is_sending: cv2.putText(interface, "Processing...", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 165, 255), 1)
        elif face_processed:
            cv2.putText(interface, "Recorded!", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 255, 0), 1)
            curr_y += 30
            cv2.putText(interface, f"SV: {student_name[:12]}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            curr_y += 25
            cv2.putText(interface, f"ID: {student_id}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            curr_y -= 55 # Bù dòng để không lệch spacing bên dưới
        else: cv2.putText(interface, "Scanning...", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (200, 200, 200), 1)

        # Dòng 3: Tiến trình khu vực đăng ký mới
        curr_y = 240
        cv2.line(interface, (px, curr_y - 15), (win_w - 20, curr_y - 15), (60, 60, 60), 1)
        cv2.putText(interface, "REGISTRATION:", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (150, 150, 150), 1)
        curr_y += spacing_y - 10
        if REG_ID and REG_NAME:
            cv2.putText(interface, f"Target: {REG_ID}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            curr_y += 30
            cv2.putText(interface, f"Buf: {len(captured_images_buffer)}/5 Pics", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), 1)
            if register_message:
                curr_y += 30
                cv2.putText(interface, register_message, (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
        else:
            cv2.putText(interface, "IDLE (Press R)", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (110, 110, 110), 1)

        # Dòng cuối cùng: Menu hướng dẫn ghim sát góc đáy lề phải
        cv2.line(interface, (px, win_h - 75), (win_w - 20, win_h - 75), (50, 50, 50), 1)
        cv2.putText(interface, "[R] Register New User", (px, win_h - 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (160, 160, 160), 1)
        cv2.putText(interface, "[ESC] Exit Application", (px, win_h - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (160, 160, 160), 1)

        # Kết xuất toàn bộ ma trận ra màn hình chuẩn 800x480
        cv2.imshow(window_name, interface)
        
        key = cv2.waitKey(15) & 0xFF
        if key == 27: break
        elif key == ord('r') or key == ord('R'):
            if not REG_ID or not REG_NAME:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                input_id = simpledialog.askstring("REGISTRATION", "Enter Student ID:")
                input_name = simpledialog.askstring("REGISTRATION", "Enter Student Full Name:")
                root.destroy()
                
                if not input_id or not input_name:
                    REG_ID = ""
                    REG_NAME = ""
                    register_message = "Missing information!"
                else:
                    REG_ID = input_id.strip()
                    REG_NAME = input_name.strip()
                    register_message = "Press R to capture"
                    captured_images_buffer = []
            else:
                if has_valid_face_in_current_frame and len(captured_images_buffer) < 5 and not is_registering:
                    _, img_encoded = cv2.imencode('.jpg', face_to_send)
                    captured_images_buffer.append(img_encoded.tobytes())
                    register_message = f"Capturing... {len(captured_images_buffer)}/5"
                    if len(captured_images_buffer) == 5:
                        threading.Thread(target=upload_registration_packet_async).start()

    cam.close_device()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()