import base64
from typing import Optional

import cv2
import numpy as np
import requests


class ImageInputError(ValueError):
    """Raised when an image request cannot be decoded."""


def decode_image(image_path: Optional[str] = None, image_base64: Optional[str] = None) -> np.ndarray:
    if image_base64:
        payload = _clean_base64_payload(image_base64)
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ImageInputError("image_base64 không hợp lệ") from exc
    elif image_path:
        try:
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            raise ImageInputError("Không tải được ảnh từ image_path") from exc
        image_bytes = response.content
    else:
        raise ImageInputError("Phải cung cấp image_path hoặc image_base64")

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageInputError("Không đọc được ảnh từ dữ liệu yêu cầu")
    return image


def _clean_base64_payload(image_base64: str) -> str:
    if "," in image_base64 and image_base64.startswith("data:"):
        return image_base64.split(",", 1)[1]
    return image_base64
