import socket

# --- CONFIG CHO TCP CLIENT ---
SERVER_HOST = 'localhost'  
SERVER_PORT = 12345       

# --- HAM GUI ANH QUA TCP ---
def send_image_to_server(image_path):
    """
    Gửi ảnh từ đường dẫn file qua TCP đến server.

    Args:
        image_path (str): Đường dẫn đến file ảnh cần gửi.
    """
    try:
        # Đọc ảnh từ file
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Tạo socket và kết nối
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))

        # Gửi dữ liệu ảnh
        client_socket.sendall(image_data)

        # Đóng kết nối
        client_socket.close()
        print(f"Da gui anh {image_path} toi server {SERVER_HOST}:{SERVER_PORT}")
    except Exception as e:
        print(f"Loi khi gui anh: {e}")