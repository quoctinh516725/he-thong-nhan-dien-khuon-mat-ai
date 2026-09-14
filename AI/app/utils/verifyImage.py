import requests
import cv2
import numpy as np


import base64

def verify_image(image_path: str = None, image_base64: str = None):
    if image_base64:
        # Giải mã chuỗi base64 thành mảng numpy
        img_data = base64.b64decode(image_base64)
        arr = np.frombuffer(img_data, np.uint8)
    elif image_path:
        # Tải ảnh về RAM (dữ liệu nhị phân)
        response = requests.get(image_path)
        arr = np.frombuffer(response.content, np.uint8)
    else:
        raise ValueError("Phải cung cấp image_path hoặc image_base64")

    # Giải mã mảng thành ảnh
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    print("Ảnh đã được giải mã thành công: ", img)
    return img

