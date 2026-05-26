import cv2
import mediapipe as mp
import time
import requests
import threading
import tkinter as tk                       # Thêm thư viện để làm ô nhập popup
from tkinter import simpledialog           
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# --- CẤU HÌNH HỆ THỐNG (CONFIG) ---
# =====================================================================
SERVER_URL = "http://localhost:8001/api/attendance/process"
REGISTER_URL = "http://localhost:8001/api/students"          
MODEL_FILE = 'face_detector.tflite'                          

# Các hằng số điều kiện nhận diện
MIN_FACE_AREA_RATIO = 0.05  
HOLD_TIME_SECONDS = 1.5     

# =====================================================================
# --- KHỞI TẠO KHUNG HÌNH CAMERA & AI ---
# =====================================================================
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Mở camera (Cài đặt độ phân giải chuẩn của mắt cam, không resize quá nhỏ)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Thử cài đặt độ phân giải HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Hệ thống Máy Điểm Danh UIT đã khởi chạy!")
print("- Nhấn phím 'ESC' để thoát phần mềm.")
print("- Nhấn phím 'R' để kích hoạt chế độ đăng ký mới.")

# --- BIẾN TOÀN CỤC QUẢN LÝ ĐIỂM DANH TỰ ĐỘNG ---
face_start_time = None
face_processed = False
is_sending = False  
student_name = ""
student_id = ""

# --- BIẾN TOÀN CỤC QUẢN LÝ ĐĂNG KÝ (LƯU TẠM TRÊN RAM) ---
REG_ID = ""                  # MSSV người cần đăng ký mới
REG_NAME = ""                # Họ tên người cần đăng ký mới
captured_images_buffer = []  # Mảng lưu trữ 5 ảnh bytes trên RAM
is_registering = False      # Cờ báo mạng bận upload dữ liệu
register_message = ""       # Văn bản hiển thị trạng thái lên màn hình

# =====================================================================
# --- CÁC HÀM XỬ LÝ PHỤ TRỢ (FUNCTIONS) ---
# =====================================================================
def is_looking_straight(keypoints):
    """Hàm AI kiểm tra xem mặt có nhìn thẳng vào camera không"""
    if not keypoints or len(keypoints) < 3: return False
    right_eye, left_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_center_x = (right_eye.x + left_eye.x) / 2.0
    eye_dist = abs(right_eye.x - left_eye.x)
    # Nếu mũi nằm chính giữa 2 mắt (lệch tối đa 25% khoảng cách 2 mắt) là nhìn thẳng
    return abs(nose.x - eye_center_x) < (eye_dist * 0.25)


def send_to_server_async(image_data):
    """Luồng ngầm tự động gửi dữ liệu ĐIỂM DANH"""
    global face_processed, is_sending, student_name, student_id
    is_sending = True
    _, img_encoded = cv2.imencode('.jpg', image_data)
    files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        # Gọi API điểm danh gốc của bạn
        res = requests.post(SERVER_URL, files=files, timeout=10)
        data = res.json()
        student_name = data.get("name", "Unknown")
        student_id = data.get("student_id", "Unknown")
    except Exception as e:
        print("[Lỗi Mạng] Điểm danh thất bại:", e)
        student_name = "Loi Server"
        student_id = ""
    is_sending = False
    face_processed = True 


def upload_registration_packet_async():
    """Luồng ngầm đóng gói toàn bộ 5 ảnh kèm thông tin đẩy lên API Server của bạn"""
    global is_registering, register_message, captured_images_buffer, REG_ID, REG_NAME
    is_registering = True
    register_message = "Dang truyen anh len server..."
    
    # Chuẩn bị gói dữ liệu chữ (Form Data)
    payload = {
        'full_name': REG_NAME,
        'student_id': REG_ID
    }
    
    # Đóng gói danh sách 5 ảnh nhị phân tương thích với API gốc của bạn
    files = []
    for i, img_bytes in enumerate(captured_images_buffer):
        files.append(('images', (f"face_{i}.jpg", img_bytes, 'image/jpeg')))
        
    try:
        res = requests.post(REGISTER_URL, data=payload, files=files, timeout=25)
        
        if res.status_code == 200:
            data = res.json()
            register_message = "THANH CONG!"
            print(f">>>> Đăng ký thành công sinh viên: {REG_NAME} ({REG_ID})")
            
            # Reset thông tin để quay về chế độ điểm danh mặc định
            REG_ID = ""
            REG_NAME = ""
            captured_images_buffer = []
        else:
            try: err_detail = res.json().get('detail', 'Server tu choi')
            except: err_detail = "Loi CSDL"
            register_message = f"That bai: {err_detail}"
            captured_images_buffer = [] # Giải phóng bộ đệm nếu lỗi để làm lại
    except Exception as e:
        print("[Lỗi Mạng] Gửi ảnh đăng ký thất bại:", e)
        register_message = "Loi ket noi mang!"
        captured_images_buffer = []
        
    is_registering = False


def draw_beautiful_box(img, pt1, pt2, color, thickness, r, d):
    """Hàm vẽ khung camera góc bo tròn nghệ thuật thay vì hình vuông thô cứng"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Góc trên bên trái
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Góc trên bên phải
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Góc dưới bên trái
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Góc dưới bên phải
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# =====================================================================
# --- VÒNG LẶP XỬ LÝ CHÍNH (MAIN LOOP) ---
# =====================================================================
# Khởi tạo cửa sổ OpenCV đặt tên cố định để co giãn tỷ lệ giao diện động
window_name = 'He thong Diem danh UIT'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Cho phép người dùng kéo giãn tự do

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 💡 GIẢI PHÁP LỚN: TỰ ĐỘNG LẤY KÍCH THƯỚC CỬA SỔ HIỆN TẠI ĐỂ TÍNH TỶ LỆ GIAO DIỆN ĐỘNG
    # Nếu cửa sổ chưa được vẽ, dùng kích thước mặc định (ví dụ 1024x600)
    win_rect = cv2.getWindowImageRect(window_name)
    if win_rect[2] > 200: # Nếu cửa sổ đã được vẽ thật sự
        win_w = win_rect[2]
        win_h = win_rect[3]
    else:
        win_w = 1024
        win_h = 600

    # Phân bổ không gian: Cột Panel chiếm 30% chiều rộng màn hình, Camera chiếm 70%
    panel_w = int(win_w * 0.3)
    cam_render_w = win_w - panel_w

    # Resize khung camera gốc khớp với không gian hiển thị động (Dùng win_h để giữ tỷ lệ)
    frame_resized = cv2.resize(frame, (cam_render_w, win_h))
    img_h, img_w, _ = frame_resized.shape
    frame_area = img_w * img_h

    # Tạo Cột Side Panel nền màu xám tốii dựa trên chiều cao cửa sổ
    interface = cv2.copyMakeBorder(frame_resized, 0, 0, 0, panel_w, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    # Xử lý luồng ảnh qua Mediapipe
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    has_valid_face_in_current_frame = False
    face_to_send = None
    progress_ratio = 0.0

    if detection_result.detections:
        largest_detection = max(detection_result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bbox = largest_detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        
        # Cấu trúc Padding lấy sát vùng tóc, hạn chế vùng vai giống code cũ của bạn
        padding_top, padding_bottom, padding_side = int(h * 0.5), int(h * 0.1), int(w * 0.2)    
        new_x = max(0, x - padding_side)
        new_y = max(0, y - padding_top)
        new_w = min(w + (2 * padding_side), img_w - new_x)
        new_h = min(h + padding_top + padding_bottom, img_h - new_y)
        
        # Cắt ảnh khuôn mặt chất lượng cao từ frame gốc (Không gửi frame đã bị nén/resize)
        scale_x = frame.shape[1] / cam_render_w
        scale_y = frame.shape[0] / win_h
        face_to_send = frame[int(new_y*scale_y):int((new_y+new_h)*scale_y), int(new_x*scale_x):int((new_x+new_w)*scale_x)]

        # Phân tích điều kiện nhận diện và vẽ khung bo tròn nghệ thuật
        if (w * h / frame_area) >= MIN_FACE_AREA_RATIO:
            if is_looking_straight(largest_detection.keypoints):
                has_valid_face_in_current_frame = True
                
                # Trạng thái màu sắc giao diện tương ứng với việc điểm danh/đăng ký
                if len(captured_images_buffer) > 0 or is_registering:
                    color = (255, 0, 255)       # Đang chụp ảnh đăng ký: Hồng tím
                elif is_sending:
                    color = (0, 165, 255)       # Đang bận gửi API điểm danh: Cam
                elif face_processed:
                    color = (0, 255, 0)         # Điểm danh thành công: Xanh lá
                else:
                    color = (255, 255, 0)       # Gương mặt đạt chuẩn chờ quét: Xanh lam Cyan
                
                # Vẽ khung trên interface dynamic
                draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), color, 2, 12, 8)
            else:
                # Lỗi góc nhìn: Vẽ khung đỏ cảnh báo nhìn thẳng
                draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2, 12, 8)
                cv2.putText(interface, "Vui long nhin thang camera", (new_x, max(15, new_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        else:
            # Lỗi khoảng cách: Vẽ khung đỏ cảnh báo lại gần
            draw_beautiful_box(interface, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2, 12, 8)
            cv2.putText(interface, "Hay tien lai gan hon", (new_x, max(15, new_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    # --- ĐIỀU KHIỂN LOGIC THỜI GIAN ĐẾM NGƯỢC (PROGRESS BAR) DƯỚI ĐÁY CAM ---
    if has_valid_face_in_current_frame and len(captured_images_buffer) == 0:
        if not face_processed and not is_sending:
            if face_start_time is None:
                face_start_time = time.time()
            else:
                dt = time.time() - face_start_time
                progress_ratio = min(1.0, dt / HOLD_TIME_SECONDS)
                if dt >= HOLD_TIME_SECONDS:
                    # Kích hoạt luồng ngầm điểm danh tự động, camera hoàn toàn không bị đứng hình
                    threading.Thread(target=send_to_server_async, args=(face_to_send,)).start()
    elif not has_valid_face_in_current_frame:
        # Nếu mất mặt hoặc đổi người -> Reset toàn bộ trạng thái cũ ngay lập tức
        face_start_time = None
        face_processed = False
        student_name = ""
        student_id = ""
        if register_message and "thành công" in register_message.lower():
            register_message = ""

    # Vẽ Progress Bar thích ứng động theo cạnh dưới camera
    bar_y = win_h - int(win_h * 0.05) # Nằm sát góc dưới 5% chiều cao
    cv2.rectangle(interface, (20, bar_y), (cam_render_w - 20, bar_y + 6), (50, 50, 50), -1)
    if progress_ratio > 0:
        cv2.rectangle(interface, (20, bar_y), (20 + int((cam_render_w - 40) * progress_ratio), bar_y + 6), (0, 255, 0), -1)

    # =====================================================================
    # TÍNH TOÁN CỠ CHỮ & KHOẢNG CÁCH TEXT DÒNG ĐỘNG THEO PHẦN TRĂM CỬA SỔ
    # =====================================================================
    px = cam_render_w + int(panel_w * 0.06) # Điểm lề trái của Panel
    
    # Tính cỡ chữ phù hợp với chiều cao màn hình (Responsive Font)
    font_scale = max(0.4, win_h / 950) # Càng màn hình to, font càng phóng to theo
    spacing = int(win_h * 0.06)        # Khoảng cách dòng dãn cách đều theo chiều cao

    # Tiêu đề chính
    cv2.putText(interface, "MAY DIEM DANH UIT", (px, int(win_h * 0.08)), cv2.FONT_HERSHEY_DUPLEX, font_scale * 1.1, (255, 255, 255), 1)
    cv2.line(interface, (px, int(win_h * 0.11)), (win_w - 15, int(win_h * 0.11)), (100, 100, 100), 1)

    # Phần 1: Nhật ký điểm danh
    curr_y = int(win_h * 0.18)
    cv2.putText(interface, "STATUS LOG:", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)
    curr_y += spacing
    if is_sending:
        cv2.putText(interface, "Processing...", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 165, 255), 1)
    elif face_processed:
        cv2.putText(interface, "DA GHI NHAN!", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 255, 0), 1)
        curr_y += spacing
        cv2.putText(interface, f"SV: {student_name}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        curr_y += spacing
        cv2.putText(interface, f"MSSV: {student_id}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    else:
        cv2.putText(interface, "Scanning...", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (200, 200, 200), 1)

    # Phần 2: Đăng ký người mới
    curr_y = int(win_h * 0.52)
    cv2.line(interface, (px, curr_y - int(spacing*0.7)), (win_w - 15, curr_y - int(spacing*0.7)), (60, 60, 60), 1)
    cv2.putText(interface, "REGISTRATION:", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)
    curr_y += spacing
    if REG_ID and REG_NAME:
        cv2.putText(interface, f"Target ID: {REG_ID}", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        curr_y += spacing
        cv2.putText(interface, f"Buffer: {len(captured_images_buffer)}/5 Pics", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), 1)
        if register_message: 
            curr_y += spacing
            cv2.putText(interface, register_message, (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
    else:
        cv2.putText(interface, "IDLE (An phím R)", (px, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 1)

    # Phần 3: Menu phím bấm giả lập ghim ở đáy Panel
    cv2.line(interface, (px, int(win_h * 0.81)), (win_w - 15, int(win_h * 0.81)), (60, 60, 60), 1)
    btn_y = int(win_h * 0.85)
    btn_h = int(win_h * 0.045) # Nút bấm cao 4.5% chiều cao màn hình
    
    # Đồ họa phím R ảo thích ứng
    cv2.rectangle(interface, (px, btn_y), (px + int(panel_w * 0.12), btn_y + btn_h), (255, 0, 255), -1)
    cv2.putText(interface, "R", (px + int(panel_w * 0.04), btn_y + int(btn_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    cv2.putText(interface, "Dang ky nguoi moi", (px + int(panel_w * 0.16), btn_y + int(btn_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

    # Đồ họa phím ESC ảo thích ứng
    btn_y += int(btn_h * 1.5) # Xuống dòng
    cv2.rectangle(interface, (px, btn_y), (px + int(panel_w * 0.12), btn_y + btn_h), (50, 50, 50), -1)
    cv2.putText(interface, "ESC", (px + int(panel_w * 0.01), btn_y + int(btn_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (255, 255, 255), 1)
    cv2.putText(interface, "Thoat phan mem", (px + int(panel_w * 0.16), btn_y + int(btn_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

    # Render giao diện phối hợp tổng thể ra màn hình
    cv2.imshow(window_name, interface)
    
    # =====================================================================
    # --- ĐỌC SỰ KIỆN PHÍM BẤM VẬT LÝ ---
    # =====================================================================
    key = cv2.waitKey(20) & 0xFF
    if key == 27: # Thoát khi nhấn ESC
        break
    elif key == ord('r') or key == ord('R'):
        # CHỨC NĂNG MỚI: TÍCH HỢP Ô NHẬP POPUP NGAY TRÊN GUI
        if not REG_ID or not REG_NAME:
            # Khởi tạo một cửa sổ Tkinter ngầm để làm gốc cho Hộp thoại Popup
            root = tk.Tk()
            root.withdraw() # Ẩn cửa sổ gốc, chỉ giữ hộp thoại
            root.attributes("-topmost", True) # Đẩy popup nổi lên trên cùng màn hình camera
            
            # Hiện hộp thoại nhập MSSV và Họ tên trực quan
            input_id = simpledialog.askstring("DANG KY MOI", "Nhập Mã số sinh viên (MSSV):")
            input_name = simpledialog.askstring("DANG KY MOI", "Nhập Họ và Tên sinh viên:")
            
            root.destroy() # Giải phóng tài nguyên ngay sau khi gõ xong
            
            # Kiểm tra chống nhập rỗng (Validation) hoặc người dùng nhấn Cancel
            if not input_id or not input_name:
                print("[Cảnh báo] Thông tin không được để trống! Đã hủy lệnh đăng ký.")
                REG_ID = ""
                REG_NAME = ""
                register_message = "Loi: Thong tin trong!"
            else:
                REG_ID = input_id.strip()
                REG_NAME = input_name.strip()
                register_message = "An phím R để chụp ảnh 1/5..."
                captured_images_buffer = [] # Giải phóng bộ đệm RAM chứa ảnh cũ
        else:
            # Nếu thông tin đã gõ hợp lệ, các lần nhấn phím R tiếp theo dùng để chụp ảnh
            if has_valid_face_in_current_frame and len(captured_images_buffer) < 5 and not is_registering:
                # Chụp và nạp vào RAM
                _, img_encoded = cv2.imencode('.jpg', face_to_send)
                captured_images_buffer.append(img_encoded.tobytes())
                register_message = f"Da chụp tam {len(captured_images_buffer)}/5"
                
                # CHẠM MỐC ĐỦ ĐÚNG 5 LẦN NHẤN PHÍM R -> ĐÓNG GÓI GỬI SERVER
                if len(captured_images_buffer) == 5:
                    threading.Thread(target=upload_registration_packet_async).start()
            elif not has_valid_face_in_current_frame:
                print("Không có khuôn mặt đạt chuẩn nhìn thẳng để tiến hành chụp!")

cap.release()
cv2.destroyAllWindows()
detector.close()