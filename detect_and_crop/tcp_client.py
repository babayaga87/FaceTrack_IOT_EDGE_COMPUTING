import socket

SERVER_HOST = 'localhost'
SERVER_PORT = 12345

def send_image_to_server(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))

        client_socket.sendall(image_data)

        try:
            response = client_socket.recv(1024).decode()
        except:
            response = "No response"

        client_socket.close()

        return f"Sent: {image_path}\nServer: {response}"

    except Exception as e:
        return f"Error: {e}"