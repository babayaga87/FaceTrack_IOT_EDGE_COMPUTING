import cv2
import os
import urllib.request
import time
import requests
import threading
import tkinter as tk                       # Thêm thư viện giao diện đồ họa
from tkinter import simpledialog           # Thêm thư viện tạo hộp thoại Popup
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# ===== USB CAMERA CLASS (BỘ ĐỌC CAM MẠNH MẼ CỦA BẠN) =====
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
        # Ép định dạng mã hóa MJPG để không bị lỗi timeout phần cứng trên WSL/Pi
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
            # Warmup camera
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
# ===== HỆ THỐNG ĐIỀU KHIỂN & CẤU HÌNH API ĐỒNG BỘ 100% =====
# =====================================================================
# Khớp cổng 8000 và trỏ chính xác về endpoint /api/students của file main.py
REGISTER_URL = "http://127.0.0.1:8002/api/students"  
SERVER_URL = "http://127.0.0.1:8002/api/students"

MODEL_FILE = 'face_detector.tflite'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

MIN_FACE_AREA_RATIO = 0.05  
HOLD_TIME_SECONDS = 1.5     

# --- CÁC BIẾN QUẢN LÝ TRẠNG THÁI TOÀN CỤC ---
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
        student_name = data.get("full_name", data.get("name", "Unknown"))
        student_id = data.get("student_id", "Unknown")
    except:
        student_name = "Loi Server"
        student_id = ""
    is_sending = False
    face_processed = True 

def upload_registration_packet_async():
    global is_registering, register_message, captured_images_buffer, REG_ID, REG_NAME
    is_registering = True
    register_message = "Dang xu ly AI..."
    
    # Ép kiểu string tường minh để FastAPI Form(...) nhận diện chính xác
    payload = {
        'full_name': str(REG_NAME).strip(), 
        'student_id': str(REG_ID).strip()
    }
    
    # Khóa key gửi file ảnh phải đặt tên chính xác là 'images' trùng khớp với main.py
    files = []
    for i, img_bytes in enumerate(captured_images_buffer):
        files.append(('images', (f"face_{i}.jpg", img_bytes, 'image/jpeg')))
        
    try:
        res = requests.post(REGISTER_URL, data=payload, files=files, timeout=25)
        if res.status_code == 200:
            register_message = "Dang ky THANH CONG!"
            print(f">>>> Đăng ký thành công: {REG_NAME} ({REG_ID})")
            REG_ID = ""
            REG_NAME = ""
            captured_images_buffer = []
        else:
            try: 
                err_detail = res.json().get('detail', 'Tu choi')
            except: 
                err_detail = f"Code {res.status_code}"
            register_message = f"That bai: {err_detail}"
            captured_images_buffer = [] 
    except Exception as e:
        register_message = "Loi ket noi!"
        print(f"Lỗi chi tiết: {str(e)}")
        captured_images_buffer = []
    is_registering = False

def draw_beautiful_box(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# =====================================================================
# ===== MAIN LOOP RUNNER =====
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

    print("Hệ thống UI/UX tích hợp Popup Đăng ký đã sẵn sàng vận hành!")

    while True:
        success, frame = cam.read_frame()
        if not success or frame is None: break

        panel_w = 320
        interface = cv2.copyMakeBorder(frame, 0, 0, 0, panel_w, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        img_h, img_w, _ = frame.shape
        frame_area = img_w * img_h

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        has_valid_face_in_current_frame = False
        face_to_send = None
        progress_ratio = 0.0

        if result.detections:
            largest_detection = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = largest_detection.bounding_box
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            
            padding_top, padding_bottom, padding_side = int(h * 0.5), int(h * 0.1), int(w * 0.2)    
            new_x = max(0, x - padding_side)
            new_y = max(0, y - padding_top)
            new_w = min(w + (2 * padding_side), img_w - new_x)
            new_h = min(h + padding_top + padding_bottom, img_h - new_y)
            face_to_send = frame[new_y:new_y+new_h, new_x:new_x+new_w]

            if (w * h / frame_area) >= MIN_FACE_AREA_RATIO:
                if is_looking_straight(largest_detection.keypoints):
                    has_valid_face_in_current_frame = True
                    
                    if len(captured_images_buffer) > 0 or is_registering: color = (255, 0, 255)
                    elif is_sending: color = (0, 165, 255)
                    elif face_processed: color = (0, 255, 0)
                    else: color = (255, 255, 0)
                    
                    draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), color, 2, 15, 10)
                else:
                    draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2, 15, 10)
                    cv2.putText(interface, "Vui long nhin thang", (new_x, new_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2, 15, 10)
                cv2.putText(interface, "Hay tien lai gan camera", (new_x, new_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

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
            student_name = ""
            student_id = ""
            if register_message and "thành công" in register_message.lower(): register_message = ""

        bar_y = img_h - 20
        cv2.rectangle(interface, (30, bar_y), (img_w - 30, bar_y + 8), (50, 50, 50), -1)
        if progress_ratio > 0:
            cv2.rectangle(interface, (30, bar_y), (30 + int((img_w - 60) * progress_ratio), bar_y + 8), (0, 255, 0), -1)

        px = img_w + 20
        cv2.putText(interface, "MAY DIEM DANH UIT", (px, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(interface, (px, 55), (px + 280, 55), (100, 100, 100), 1)

        cv2.putText(interface, "STATUS LOG:", (px, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if is_sending: cv2.putText(interface, "Processing...", (px, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif face_processed:
            cv2.putText(interface, "DA GHI NHAN!", (px, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(interface, f"SV: {student_name}", (px, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(interface, f"MSSV: {student_id}", (px, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else: cv2.putText(interface, "Scanning...", (px, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.line(interface, (px, 215), (px + 280, 215), (60, 60, 60), 1)

        cv2.putText(interface, "MODE REGISTRATION:", (px, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if REG_ID and REG_NAME:
            cv2.putText(interface, f"Target: {REG_ID}", (px, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(interface, f"Buffer: {len(captured_images_buffer)} / 5 Pics", (px, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
            if register_message: cv2.putText(interface, register_message, (px, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else: cv2.putText(interface, "IDLE (An R de dang ky)", (px, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

        cv2.line(interface, (px, 365), (px + 280, 365), (60, 60, 60), 1)
        cv2.putText(interface, "GUIDE MENU:", (px, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.rectangle(interface, (px, 410), (px + 40, 435), (255, 0, 255), -1)
        cv2.putText(interface, "R", (px + 13, 429), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(interface, "Dang ky nguoi moi", (px + 55, 427), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(interface, (px, 445), (px + 40, 470), (50, 50, 50), -1)
        cv2.putText(interface, "ESC", (px + 4, 464), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(interface, "Thoat phan mem", (px + 55, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('He thong Diem danh UIT', interface)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: break
        elif key == ord('r') or key == ord('R'):
            if not REG_ID or not REG_NAME:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                
                input_id = simpledialog.askstring("ĐĂNG KÝ SINH VIÊN", "Nhập Mã số sinh viên (MSSV):")
                input_name = simpledialog.askstring("ĐĂNG KÝ SINH VIÊN", "Nhập Họ và Tên sinh viên:")
                root.destroy()
                
                if not input_id or not input_name:
                    REG_ID = ""
                    REG_NAME = ""
                    register_message = "Loi: Thong tin trong!"
                else:
                    REG_ID = input_id.strip()
                    REG_NAME = input_name.strip()
                    register_message = "An R de chup anh thu 1..."
                    captured_images_buffer = []
            else:
                # Nhấn R liên tục 5 lần để chụp đủ ảnh đẩy thẳng lên Server
                if has_valid_face_in_current_frame and len(captured_images_buffer) < 5 and not is_registering:
                    _, img_encoded = cv2.imencode('.jpg', face_to_send)
                    captured_images_buffer.append(img_encoded.tobytes())
                    register_message = f"Da chup tam {len(captured_images_buffer)}/5"
                    if len(captured_images_buffer) == 5:
                        threading.Thread(target=upload_registration_packet_async).start()
                elif not has_valid_face_in_current_frame:
                    print("Gương mặt chưa đạt chuẩn, không thể chụp!")

    cam.close_device()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()