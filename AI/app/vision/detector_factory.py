import os

from app.vision.face_detection import (
    FaceDetectionConfig,
    FaceDetectionService,
    MtcnnFaceDetector,
    YoloOnnxConfig,
    YoloOnnxFaceDetector,
)
from app.vision.preprocessing import FacePreprocessingPipeline


def build_face_detection_service(mtcnn_model, mtcnn_lock=None) -> tuple[str, FaceDetectionService]:
    """Build the configured detector service.

    If `YOLO_FACE_ONNX_PATH` points to a valid ONNX model, detection uses YOLO.
    Otherwise the service falls back to the existing MTCNN detector so the app
    remains runnable on CPU-only servers and Apple Silicon development machines.
    """

    yolo_model_path = os.getenv("YOLO_FACE_ONNX_PATH")
    if yolo_model_path:
        yolo_input_size = int(os.getenv("YOLO_FACE_INPUT_SIZE", "640"))
        detector = YoloOnnxFaceDetector(
            YoloOnnxConfig(
                model_path=yolo_model_path,
                input_size=(yolo_input_size, yolo_input_size),
                confidence_threshold=float(os.getenv("YOLO_FACE_CONFIDENCE", "0.45")),
                nms_threshold=float(os.getenv("YOLO_FACE_NMS", "0.45")),
            )
        )
        return "yolo-onnx", FaceDetectionService(
            detector=detector,
            config=FaceDetectionConfig(
                min_confidence=float(os.getenv("FACE_MIN_CONFIDENCE", "0.45"))
            ),
            preprocessing=FacePreprocessingPipeline(processors=[]),
        )

    return "mtcnn", FaceDetectionService(
        detector=MtcnnFaceDetector(mtcnn_model, lock=mtcnn_lock)
    )
