import os
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from app.vision.exceptions import (
    AmbiguousFaceError,
    DetectionModelError,
    InvalidFrameError,
    NoFaceDetectedError,
)
from app.vision.preprocessing import FacePreprocessingPipeline, validate_bgr_frame


@dataclass(frozen=True)
class FaceDetectionConfig:
    min_confidence: float = 0.90
    primary_face_area_ratio: float = 3.0


@dataclass(frozen=True)
class YoloOnnxConfig:
    model_path: str
    input_size: tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45


@dataclass(frozen=True)
class YuNetConfig:
    model_path: str
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.30
    top_k: int = 5000


@dataclass(frozen=True)
class FaceDetectionResult:
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(
    first: FaceDetectionResult,
    second: FaceDetectionResult,
) -> float:
    left = max(first.box[0], second.box[0])
    top = max(first.box[1], second.box[1])
    right = min(first.box[2], second.box[2])
    bottom = min(first.box[3], second.box[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def deduplicate_face_detections(
    detections: Sequence[FaceDetectionResult],
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.85,
) -> list[FaceDetectionResult]:
    """Collapse duplicate boxes while preserving spatially distinct faces."""
    kept: list[FaceDetectionResult] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in kept:
            intersection = _intersection_area(detection, existing)
            union = detection.area + existing.area - intersection
            iou = intersection / max(union, 1e-12)
            containment = intersection / max(min(detection.area, existing.area), 1e-12)
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[FaceDetectionResult]:
        """Detect faces in a BGR `uint8` frame."""


class FaceDetectionService:
    """Selects one primary face from detector output."""

    def __init__(
        self,
        detector: FaceDetector,
        config: Optional[FaceDetectionConfig] = None,
        preprocessing: Optional[FacePreprocessingPipeline] = None,
    ) -> None:
        self._detector = detector
        self._config = config or FaceDetectionConfig()
        self._preprocessing = preprocessing or FacePreprocessingPipeline()

    def detect_primary_face(self, frame: Optional[np.ndarray]) -> FaceDetectionResult:
        detections = self.detect_faces(frame)

        if not detections:
            raise NoFaceDetectedError("No face detected above confidence threshold.")

        sorted_detections = sorted(detections, key=lambda item: item.area, reverse=True)
        if len(sorted_detections) > 1:
            largest = sorted_detections[0].area
            second_largest = sorted_detections[1].area
            if largest < self._config.primary_face_area_ratio * second_largest:
                raise AmbiguousFaceError(
                    "Multiple similarly sized faces detected; primary face is ambiguous."
                )

        return sorted_detections[0]

    def detect_faces(self, frame: Optional[np.ndarray]) -> list[FaceDetectionResult]:
        validate_bgr_frame(frame)
        processed_frame = self._preprocessing.run(frame)
        detections = [
            detection
            for detection in self._detector.detect(processed_frame)
            if detection.confidence >= self._config.min_confidence and detection.area > 0
        ]
        return deduplicate_face_detections(detections)


class MtcnnFaceDetector:
    """MTCNN adapter that returns detector metadata without creating embeddings."""

    def __init__(self, mtcnn_model, lock: Optional[Lock] = None) -> None:
        self._mtcnn_model = mtcnn_model
        self._lock = lock or Lock()

    def detect(self, frame: np.ndarray) -> Sequence[FaceDetectionResult]:
        validate_bgr_frame(frame)
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with self._lock:
                boxes, probabilities = self._mtcnn_model.detect(image)
        except Exception as exc:
            raise DetectionModelError(f"MTCNN detection failed: {exc}") from exc

        if boxes is None or probabilities is None:
            return []

        return [
            FaceDetectionResult(
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(probability),
            )
            for box, probability in zip(boxes, probabilities)
            if probability is not None
        ]


class YuNetFaceDetector:
    """Small OpenCV YuNet detector used only to rescue missed YOLO faces."""

    def __init__(self, config: YuNetConfig) -> None:
        if not os.path.exists(config.model_path):
            raise DetectionModelError(f"YuNet model not found: {config.model_path}")
        self._config = config
        self._lock = Lock()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                config.model_path,
                "",
                (320, 320),
                config.confidence_threshold,
                config.nms_threshold,
                config.top_k,
            )
        except Exception as exc:
            raise DetectionModelError(f"Failed to load YuNet ONNX model: {exc}") from exc

    def detect(self, frame: np.ndarray) -> Sequence[FaceDetectionResult]:
        validate_bgr_frame(frame)
        height, width = frame.shape[:2]
        try:
            with self._lock:
                self._detector.setInputSize((width, height))
                _, faces = self._detector.detect(frame)
        except Exception as exc:
            raise DetectionModelError(f"YuNet inference failed: {exc}") from exc
        if faces is None:
            return []
        return [
            FaceDetectionResult(
                box=(
                    float(face[0]),
                    float(face[1]),
                    float(face[0] + face[2]),
                    float(face[1] + face[3]),
                ),
                confidence=float(face[-1]),
            )
            for face in faces
            if face[2] > 0 and face[3] > 0
        ]


class YoloOnnxFaceDetector:
    """YOLO face detector adapter backed by OpenCV DNN and an ONNX model."""

    def __init__(self, config: YoloOnnxConfig) -> None:
        if not os.path.exists(config.model_path):
            raise DetectionModelError(f"YOLO model not found: {config.model_path}")
        self._config = config
        self._lock = Lock()
        try:
            self._network = cv2.dnn.readNetFromONNX(config.model_path)
        except Exception as exc:
            raise DetectionModelError(f"Failed to load YOLO ONNX model: {exc}") from exc

    def detect(self, frame: np.ndarray) -> Sequence[FaceDetectionResult]:
        validate_bgr_frame(frame)
        height, width = frame.shape[:2]
        input_width, input_height = self._config.input_size
        letterboxed, scale, pad_x, pad_y = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
        )

        try:
            with self._lock:
                self._network.setInput(blob)
                output = self._network.forward()
        except Exception as exc:
            raise DetectionModelError(f"YOLO inference failed: {exc}") from exc

        boxes, confidences = self._parse_yolo_output(
            output,
            width,
            height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=self._config.confidence_threshold,
            nms_threshold=self._config.nms_threshold,
        )
        if len(indices) == 0:
            return []

        flat_indices = np.array(indices).reshape(-1)
        return [
            FaceDetectionResult(
                box=(
                    float(boxes[index][0]),
                    float(boxes[index][1]),
                    float(boxes[index][0] + boxes[index][2]),
                    float(boxes[index][1] + boxes[index][3]),
                ),
                confidence=float(confidences[index]),
            )
            for index in flat_indices
        ]

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        frame_height, frame_width = frame.shape[:2]
        input_width, input_height = self._config.input_size
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_y,
            input_height - resized_height - pad_y,
            pad_x,
            input_width - resized_width - pad_x,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return letterboxed, scale, pad_x, pad_y

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        frame_width: int,
        frame_height: int,
        scale: float = 1.0,
        pad_x: int = 0,
        pad_y: int = 0,
    ) -> tuple[list[list[int]], list[float]]:
        predictions = np.squeeze(output)
        if predictions.ndim == 1 and predictions.shape[0] >= 5:
            predictions = predictions[None, :]
        if predictions.ndim != 2:
            return [], []
        if (
            predictions.shape[0] >= 5
            and predictions.shape[1] < 5
        ) or (
            predictions.shape[0] <= 20
            and predictions.shape[1] > predictions.shape[0]
        ):
            predictions = predictions.T

        boxes: list[list[int]] = []
        confidences: list[float] = []
        for prediction in predictions:
            if prediction.shape[0] < 5:
                continue
            confidence = float(prediction[4])
            if confidence < self._config.confidence_threshold:
                continue
            center_x, center_y, box_width, box_height = prediction[:4]
            x1 = int(round((center_x - box_width / 2 - pad_x) / scale))
            y1 = int(round((center_y - box_height / 2 - pad_y) / scale))
            x2 = int(round((center_x + box_width / 2 - pad_x) / scale))
            y2 = int(round((center_y + box_height / 2 - pad_y) / scale))
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(x1 + 1, min(x2, frame_width))
            y2 = max(y1 + 1, min(y2, frame_height))
            boxes.append(
                [x1, y1, x2 - x1, y2 - y1]
            )
            confidences.append(confidence)

        return boxes, confidences


__all__ = [
    "AmbiguousFaceError",
    "DetectionModelError",
    "FaceDetectionConfig",
    "FaceDetectionResult",
    "FaceDetectionService",
    "FaceDetector",
    "InvalidFrameError",
    "MtcnnFaceDetector",
    "NoFaceDetectedError",
    "YoloOnnxConfig",
    "YoloOnnxFaceDetector",
    "YuNetConfig",
    "YuNetFaceDetector",
]
