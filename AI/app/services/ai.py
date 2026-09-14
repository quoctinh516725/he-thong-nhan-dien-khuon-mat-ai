import cv2
import hashlib
import os
import urllib.request
from threading import Lock

import numpy as np
from fastapi import HTTPException
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from app.vision.exceptions import AmbiguousFaceError, InvalidFrameError, NoFaceDetectedError
from app.vision.detector_factory import build_face_detection_service
from app.vision.face_detection import (
    FaceDetectionConfig,
    FaceDetectionResult,
    FaceDetectionService,
    MtcnnFaceDetector,
    YuNetConfig,
    YuNetFaceDetector,
)
from app.vision.preprocessing import (
    FaceCropConfig,
    FacePreprocessingPipeline,
    classify_head_pose,
    crop_square_face,
    crop_warped_face,
    is_low_light_face,
    mask_lower_face,
    normalize_illumination_reference,
    normalize_low_light_query,
)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).lower()
    return value in {"1", "true", "yes", "on"}


def resolve_embedding_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    enable_mps = _env_enabled("AI_ENABLE_MPS")
    if enable_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# MTCNN has unsupported adaptive-pooling shapes on MPS. Keep detection on CPU
# while allowing the batch-friendly embedding model to use MPS.
embedding_device = resolve_embedding_device()
detection_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = embedding_device

# Detect mặt từ ảnh sử dụng MTCNN
detection_lock = Lock()
mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, device=detection_device)
detector_name, face_detection_service = build_face_detection_service(
    mtcnn,
    mtcnn_lock=detection_lock,
)
fallback_detector_name = "mtcnn-fallback"
fallback_face_detection_service = FaceDetectionService(
    detector=MtcnnFaceDetector(mtcnn, lock=detection_lock),
    config=FaceDetectionConfig(
        min_confidence=float(os.getenv("MTCNN_FALLBACK_FACE_MIN_CONFIDENCE", "0.80"))
    ),
)

# Lấy bộ dữ liệu VGGFace2 đã được huấn luyện sẵn cho model InceptionResnetV1
resnet = InceptionResnetV1(pretrained="vggface2").to(embedding_device).eval()

# Khởi tạo SelfieSegmentation từ MediaPipe Tasks API đã được huấn luyện sẵn bởi Google để phân tách nền chân dung
def init_selfie_segmenter():
    model_path = "./models/selfie_segmenter.tflite"
    if not os.path.exists(model_path):
        print("Downloading Selfie Segmenter TFLite model from Google CDN...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Selfie Segmenter TFLite model downloaded successfully.")
        except Exception as e:
            print(f"ERROR downloading Selfie Segmenter model: {str(e)}")
            raise

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_confidence_masks=True
    )
    return vision.ImageSegmenter.create_from_options(options)


use_segmentation = _env_enabled("AI_USE_SEGMENTATION", True)
selfie_segmenter = init_selfie_segmenter() if use_segmentation else None
segmentation_lock = Lock()
embedding_lock = Lock()
face_crop_config = FaceCropConfig(
    output_size=160,
    margin_ratio=float(os.getenv("FACE_CROP_MARGIN_RATIO", "0.12")),
)
default_yunet_model_path = "./models/face_detection_yunet_2023mar.onnx"
yunet_model_path = os.getenv(
    "YUNET_FACE_ONNX_PATH",
    default_yunet_model_path,
)


def _ensure_yunet_model(model_path: str) -> bool:
    if os.path.exists(model_path):
        return True
    if model_path != default_yunet_model_path or not _env_enabled(
        "AI_AUTO_DOWNLOAD_MODELS",
        True,
    ):
        return False

    url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    expected_sha256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    temporary_path = f"{model_path}.download"
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(url, temporary_path)
        with open(temporary_path, "rb") as model_file:
            digest = hashlib.sha256(model_file.read()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Downloaded YuNet model checksum does not match.")
        os.replace(temporary_path, model_path)
        return True
    except Exception as exc:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        print(f"WARNING: YuNet rescue model is unavailable: {exc}")
        return False


yunet_face_detection_service = None
if _ensure_yunet_model(yunet_model_path) and _env_enabled("AI_ENABLE_YUNET_RESCUE", True):
    yunet_face_detection_service = FaceDetectionService(
        detector=YuNetFaceDetector(
            YuNetConfig(
                model_path=yunet_model_path,
                confidence_threshold=float(os.getenv("YUNET_FACE_CONFIDENCE", "0.35")),
            )
        ),
        config=FaceDetectionConfig(
            min_confidence=float(os.getenv("YUNET_FACE_CONFIDENCE", "0.35"))
        ),
        preprocessing=FacePreprocessingPipeline(processors=[]),
    )


def _rotate_frame(frame: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, cv2.invertAffineTransform(matrix)


def _restore_rotated_detection(
    detection: FaceDetectionResult,
    inverse_matrix: np.ndarray,
    frame_shape: tuple[int, ...],
) -> FaceDetectionResult:
    x1, y1, x2, y2 = detection.box
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    restored = corners @ inverse_matrix.T
    height, width = frame_shape[:2]
    restored_x1 = float(np.clip(restored[:, 0].min(), 0, width - 1))
    restored_y1 = float(np.clip(restored[:, 1].min(), 0, height - 1))
    restored_x2 = float(np.clip(restored[:, 0].max(), restored_x1 + 1, width))
    restored_y2 = float(np.clip(restored[:, 1].max(), restored_y1 + 1, height))
    return FaceDetectionResult(
        box=(restored_x1, restored_y1, restored_x2, restored_y2),
        confidence=detection.confidence,
    )


def detect_faces_with_fallback(frame) -> tuple[list[FaceDetectionResult], str]:
    detections = face_detection_service.detect_faces(frame)
    if detections:
        return detections, detector_name

    if yunet_face_detection_service is not None:
        yunet_detections = yunet_face_detection_service.detect_faces(frame)
        if yunet_detections:
            return yunet_detections, "yunet-rescue"

    if detector_name == "yolo-onnx" and _env_enabled("AI_ENABLE_ROTATION_RESCUE", True):
        for angle in (-30.0, 30.0):
            rotated, inverse_matrix = _rotate_frame(frame, angle)
            rotated_detections = face_detection_service.detect_faces(rotated)
            if rotated_detections:
                restored = [
                    _restore_rotated_detection(item, inverse_matrix, frame.shape)
                    for item in rotated_detections
                ]
                return restored, f"{detector_name}-rotation-rescue"

    fallback_detections = fallback_face_detection_service.detect_faces(frame)
    return fallback_detections, fallback_detector_name


def detect_enrollment_face(frame) -> tuple[FaceDetectionResult, str]:
    """Use the same landmark detector that accepted the auto-capture frame."""
    detections = [
        detection
        for detection in fallback_face_detection_service.detect_faces(frame)
        if detection.confidence >= 0.90
    ]
    if not detections:
        raise NoFaceDetectedError("No MTCNN face detected above enrollment confidence.")
    detections.sort(key=lambda item: item.area, reverse=True)
    if len(detections) > 1 and detections[0].area < 3.0 * detections[1].area:
        print(
            "ENROLLMENT_REJECT multiple MTCNN faces:",
            [(item.box, round(item.confidence, 3)) for item in detections],
        )
        raise AmbiguousFaceError("Multiple MTCNN faces detected.")
    return detections[0], "mtcnn-enrollment"


def detect_primary_face_with_fallback(frame) -> tuple[FaceDetectionResult, str]:
    try:
        return face_detection_service.detect_primary_face(frame), detector_name
    except NoFaceDetectedError:
        return fallback_face_detection_service.detect_primary_face(frame), fallback_detector_name

def extract_face(frame):
    try:
        detect_primary_face_with_fallback(frame)
    except InvalidFrameError as exc:
        raise HTTPException(
            status_code=400,
            detail="Không đọc được ảnh từ đường dẫn yêu cầu.",
        ) from exc
    except AmbiguousFaceError as exc:
        raise HTTPException(
            status_code=400,
            detail="Phát hiện nhiều gương mặt có kích thước tương đương nhau. Vui lòng cung cấp ảnh chỉ có một gương mặt rõ ràng.",
        ) from exc
    except NoFaceDetectedError as exc:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt trong ảnh.") from exc

    # InceptionResnetV1 yêu cầu định dạng màu là RGB, trong khi OpenCV đọc ảnh ở định dạng BGR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chuyển đổi sang PIL Image vì MTCNN hoạt động tốt với PIL
    img_pil = Image.fromarray(img_rgb)

    face_tensor = mtcnn(img_pil)

    if face_tensor is not None:
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        return face_tensor.to(embedding_device)
    raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt trong ảnh.")
    
def get_embedding(face_tensor):
    return get_embeddings(face_tensor)[0]


def get_embeddings(face_tensors) -> np.ndarray:
    if isinstance(face_tensors, (list, tuple)):
        if not face_tensors:
            return np.empty((0, 512), dtype=np.float32)
        batch = torch.cat(face_tensors, dim=0)
    else:
        batch = face_tensors
    with torch.inference_mode():
        with embedding_lock:
            embeddings = resnet(batch.to(embedding_device))
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def face_crop_to_tensor(crop: np.ndarray) -> torch.Tensor:
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float()
    return ((tensor - 127.5) / 128.0).unsqueeze(0)


def extract_upper_face_tensor(frame, box) -> torch.Tensor:
    tight_crop = crop_warped_face(frame, box, output_size=160)
    return face_crop_to_tensor(mask_lower_face(tight_crop))


def extract_low_light_face_tensor(frame, box) -> torch.Tensor:
    square_crop = crop_square_face(frame, box, face_crop_config)
    return face_crop_to_tensor(normalize_low_light_query(square_crop))


def extract_illumination_reference_tensor(crop: np.ndarray) -> torch.Tensor:
    resized_crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    return face_crop_to_tensor(normalize_illumination_reference(resized_crop))


def is_low_light_face_region(frame, box) -> bool:
    return is_low_light_face(crop_square_face(frame, box, face_crop_config))


def analyze_enrollment_face(frame: np.ndarray) -> dict:
    """Estimate coarse head pose and capture quality from existing MTCNN landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with detection_lock:
        boxes, probabilities, landmarks = mtcnn.detect(
            Image.fromarray(rgb_frame),
            landmarks=True,
        )
    if boxes is None or probabilities is None or landmarks is None:
        return {"face_count": 0, "pose": "unknown", "quality_ready": False, "reason": "Không thấy khuôn mặt."}

    valid = [index for index, probability in enumerate(probabilities) if probability and probability >= 0.90]
    if len(valid) > 1:
        valid.sort(
            key=lambda index: max(0.0, boxes[index][2] - boxes[index][0])
            * max(0.0, boxes[index][3] - boxes[index][1]),
            reverse=True,
        )
        largest_area = max(0.0, boxes[valid[0]][2] - boxes[valid[0]][0]) * max(
            0.0,
            boxes[valid[0]][3] - boxes[valid[0]][1],
        )
        second_area = max(0.0, boxes[valid[1]][2] - boxes[valid[1]][0]) * max(
            0.0,
            boxes[valid[1]][3] - boxes[valid[1]][1],
        )
        if largest_area >= 3.0 * second_area:
            valid = valid[:1]
    if len(valid) != 1:
        reason = "Chỉ để một người trong khung hình." if len(valid) > 1 else "Đưa khuôn mặt vào gần camera hơn."
        return {"face_count": len(valid), "pose": "unknown", "quality_ready": False, "reason": reason}

    index = valid[0]
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = boxes[index]
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face_ratio = float((x2 - x1) * (y2 - y1) / max(width * height, 1))

    left_eye, right_eye, nose, left_mouth, right_mouth = np.asarray(landmarks[index], dtype=np.float32)
    eye_midpoint = (left_eye + right_eye) / 2.0
    mouth_midpoint = (left_mouth + right_mouth) / 2.0
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    eye_to_mouth = max(float(mouth_midpoint[1] - eye_midpoint[1]), 1.0)
    yaw = float((nose[0] - eye_midpoint[0]) / eye_distance)
    pitch = float((nose[1] - eye_midpoint[1]) / eye_to_mouth)

    pose = classify_head_pose(yaw, pitch)

    quality_ready = True
    reason = "Giữ yên."
    if face_ratio < 0.06:
        quality_ready = False
        reason = "Tiến gần camera hơn."
    elif brightness < 45.0:
        quality_ready = False
        reason = "Bổ sung ánh sáng phía trước."
    elif brightness > 235.0:
        quality_ready = False
        reason = "Giảm ánh sáng chiếu trực tiếp."
    elif sharpness < 30.0:
        quality_ready = False
        reason = "Giữ đầu ổn định để ảnh bớt nhòe."

    return {
        "face_count": 1,
        "pose": pose,
        "quality_ready": quality_ready,
        "reason": reason,
        "box": [x1, y1, x2, y2],
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "face_ratio": round(face_ratio, 4),
        "yaw": round(yaw, 3),
        "pitch": round(pitch, 3),
    }


def extract_face_crop_tensor(frame, box):
    resized_crop = crop_square_face(frame, box, face_crop_config)
    tight_crop = crop_warped_face(frame, box, output_size=160)
    rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

    mask_np = np.full((160, 160), 255, dtype=np.uint8)
    masked_crop_bgr = resized_crop
    if selfie_segmenter is not None:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        try:
            with segmentation_lock:
                segmentation_result = selfie_segmenter.segment(mp_image)
            prob_mask = segmentation_result.confidence_masks[0].numpy_view().squeeze()
            candidate_mask = (prob_mask >= 0.5).astype(np.uint8) * 255
            if candidate_mask.sum() > 0:
                mask_np = candidate_mask
                masked_crop_bgr = cv2.bitwise_and(
                    resized_crop,
                    resized_crop,
                    mask=mask_np,
                )
        except Exception as exc:
            print(f"WARNING: MediaPipe segmentation failed: {exc}. Using raw crop.")

    tensor = face_crop_to_tensor(masked_crop_bgr)
    return tensor, tight_crop, mask_np, masked_crop_bgr
    
