import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== USB CAMERA CLASS =====
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
            # Warmup
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

# ===== CONFIG =====
MODEL_FILE = 'face_detector.tflite'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
SAVE_DIR = "dataset"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ===== MAIN =====
def main():
    # Load model
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Init camera
    cam = UsbCamera()
    if not cam.open_device(0, 640, 480):
        print("Cannot open camera:", cam.get_last_error())
        return

    print("Camera started!")
    print("-> Press 's' to save face, 'q' to quit")
    
    count = 0

    while True:
        success, frame = cam.read_frame()
        if not success or frame is None:
            print("Read frame error:", cam.get_last_error())
            break

        # Mediapipe detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        face_to_save = None

        if result.detections:
            for det in result.detections:
                bbox = det.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                img_h, img_w, _ = frame.shape
                padding_top = int(h*0.5)
                padding_bottom = int(h*0.1)
                padding_side = int(w*0.2)
                new_x = max(0, x - padding_side)
                new_y = max(0, y - padding_top)
                new_w = min(w + 2*padding_side, img_w - new_x)
                new_h = min(h + padding_top + padding_bottom, img_h - new_y)
                face = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                if face is not None and face.size > 0:
                    face_to_save = face
                cv2.rectangle(frame, (new_x,new_y), (new_x+new_w,new_y+new_h), (0,255,0), 2)

        cv2.imshow("AI Camera,'S' to save face, 'Q' to quit", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            print("Exiting...")
            break
        elif key == ord('s') and face_to_save is not None:
            count += 1
            file_name = f"{SAVE_DIR}/face_{count}.jpg"
            cv2.imwrite(file_name, face_to_save)
            print(f"Saved: {file_name}")

    cam.close_device()
    detector.close()
    cv2.destroyAllWindows()
    print("Camera stopped.")

if __name__ == "__main__":
    main()