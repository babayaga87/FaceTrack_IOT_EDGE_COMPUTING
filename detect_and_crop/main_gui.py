import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tcp_client import send_image_to_server

# --- CONFIG ---
MODEL_FILE = 'face_detector.tflite'
SAVE_DIR = "dataset/23520688"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- LOG SYSTEM ---
log_lines = []

def add_log(text):
    global log_lines
    log_lines.append(text)
    if len(log_lines) > 10:
        log_lines.pop(0)

def clear_log():
    global log_lines
    log_lines = []

def draw_log_panel(frame):
    h, w, _ = frame.shape
    panel_width = 300

    # Tạo panel đen
    panel = frame[:, :panel_width].copy()
    panel[:] = (0, 0, 0)

    y = 30
    for line in log_lines:
        cv2.putText(panel, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        y += 25

    combined = cv2.hconcat([panel, frame])
    return combined

# --- INIT DETECTOR ---
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# --- AUTO FIND CAMERA ---
def find_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ret, _ = cap.read()
        cap.release()
        if ret:
            return i
    return -1

cam_index = find_camera()
print("Using camera:", cam_index)

# --- OPEN CAMERA (FIX TIMEOUT) ---
cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# --- WARMUP CAMERA ---
for _ in range(10):
    cap.read()

count = 0

print("He thong da san sang!")
print("- Nhan 's' de luu + gui anh.")
print("- Nhan 'ESC' de thoat.")

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        add_log("Camera read failed!")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    face_to_save = None

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

            img_h, img_w, _ = frame.shape

            padding_top = int(h * 0.5)
            padding_bottom = int(h * 0.1)
            padding_side = int(w * 0.2)

            new_x = max(0, x - padding_side)
            new_y = max(0, y - padding_top)
            new_w = min(w + 2 * padding_side, img_w - new_x)
            new_h = min(h + padding_top + padding_bottom, img_h - new_y)

            face_to_save = frame[new_y:new_y+new_h, new_x:new_x+new_w]

            cv2.rectangle(frame, (new_x, new_y),
                          (new_x + new_w, new_y + new_h),
                          (0, 255, 0), 2)

            # if face_to_save is not None and face_to_save.size > 0:
            #     cv2.imshow('Face Crop', face_to_save)

    # --- SHOW WITH LOG PANEL ---
    display_frame = draw_log_panel(frame)
    cv2.imshow('AI Attendance System', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == ord('s'):
        if face_to_save is not None and face_to_save.size > 0:
            count += 1
            file_name = f"{SAVE_DIR}/face_{count}.jpg"
            cv2.imwrite(file_name, face_to_save)

            clear_log()
            add_log(f"Saved: {file_name}")

            # --- SEND TCP ---
            result = send_image_to_server(file_name)

            for line in result.split("\n"):
                add_log(line)

        else:
            clear_log()
            add_log("No face detected!")

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
detector.close()