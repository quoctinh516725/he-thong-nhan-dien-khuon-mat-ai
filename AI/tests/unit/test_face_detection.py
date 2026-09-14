import numpy as np
import pytest

from app.vision.face_detection import (
    AmbiguousFaceError,
    FaceDetectionResult,
    FaceDetectionService,
    InvalidFrameError,
    NoFaceDetectedError,
    YoloOnnxConfig,
    YoloOnnxFaceDetector,
    deduplicate_face_detections,
)
from app.vision.preprocessing import FacePreprocessingPipeline
from app.vision.preprocessing import (
    classify_head_pose,
    crop_square_face,
    crop_warped_face,
    is_low_light_face,
    mask_lower_face,
    normalize_illumination_reference,
    normalize_low_light_query,
    orient_pose_for_preview,
)


def test_classify_head_pose_covers_capture_directions():
    assert classify_head_pose(0.0, 0.58) == "front"
    assert classify_head_pose(-0.20, 0.58) == "left"
    assert classify_head_pose(0.20, 0.58) == "right"
    assert classify_head_pose(0.0, 0.45) == "up"
    assert classify_head_pose(0.0, 0.72) == "down"
    assert classify_head_pose(0.145, 0.58) == "unknown"


def test_mirrored_preview_swaps_horizontal_pose_only():
    assert orient_pose_for_preview("left", mirrored=True) == "right"
    assert orient_pose_for_preview("right", mirrored=True) == "left"
    assert orient_pose_for_preview("up", mirrored=True) == "up"
    assert orient_pose_for_preview("left", mirrored=False) == "left"


class StubDetector:
    def __init__(self, results):
        self._results = results

    def detect(self, frame):
        return self._results


def make_frame(width=128, height=96):
    return np.full((height, width, 3), 120, dtype=np.uint8)


def test_detection_service_selects_largest_confident_face():
    small = FaceDetectionResult(box=(5, 5, 25, 25), confidence=0.98)
    large = FaceDetectionResult(box=(40, 10, 110, 90), confidence=0.95)
    service = FaceDetectionService(detector=StubDetector([small, large]))

    result = service.detect_primary_face(make_frame())

    assert result == large


def test_detection_service_rejects_multiple_similar_faces():
    first = FaceDetectionResult(box=(5, 5, 55, 55), confidence=0.99)
    second = FaceDetectionResult(box=(65, 5, 115, 55), confidence=0.98)
    service = FaceDetectionService(detector=StubDetector([first, second]))

    with pytest.raises(AmbiguousFaceError):
        service.detect_primary_face(make_frame())


def test_duplicate_face_boxes_are_collapsed():
    strong = FaceDetectionResult(box=(20, 10, 100, 90), confidence=0.97)
    nested = FaceDetectionResult(box=(24, 14, 96, 86), confidence=0.84)

    assert deduplicate_face_detections([nested, strong]) == [strong]


def test_spatially_distinct_faces_are_preserved():
    first = FaceDetectionResult(box=(5, 5, 45, 55), confidence=0.97)
    second = FaceDetectionResult(box=(70, 5, 115, 60), confidence=0.95)

    assert deduplicate_face_detections([first, second]) == [first, second]


def test_detection_service_ignores_low_confidence_faces():
    low = FaceDetectionResult(box=(5, 5, 55, 55), confidence=0.50)
    service = FaceDetectionService(detector=StubDetector([low]))

    with pytest.raises(NoFaceDetectedError):
        service.detect_primary_face(make_frame())


def test_detection_service_returns_all_confident_faces_for_realtime_detection():
    low = FaceDetectionResult(box=(1, 1, 5, 5), confidence=0.20)
    first = FaceDetectionResult(box=(5, 5, 55, 55), confidence=0.99)
    second = FaceDetectionResult(box=(65, 5, 115, 55), confidence=0.98)
    service = FaceDetectionService(detector=StubDetector([low, first, second]))

    results = service.detect_faces(make_frame())

    assert results == [first, second]


def test_detection_service_rejects_empty_frame():
    service = FaceDetectionService(detector=StubDetector([]))

    with pytest.raises(InvalidFrameError):
        service.detect_primary_face(None)


def test_preprocessing_pipeline_returns_copy_and_keeps_source_unchanged():
    frame = make_frame()
    original = frame.copy()
    pipeline = FacePreprocessingPipeline()

    processed = pipeline.run(frame)

    assert processed is not frame
    assert np.array_equal(frame, original)
    assert processed.shape == frame.shape
    assert processed.dtype == np.uint8


def test_low_light_classifier_uses_face_luminance_not_a_single_hotspot():
    dark_face = np.full((160, 160, 3), 18, dtype=np.uint8)
    dark_face[40:120, 95:150] = 150
    normal_face = np.full((160, 160, 3), 105, dtype=np.uint8)

    assert is_low_light_face(dark_face) is True
    assert is_low_light_face(normal_face) is False


def test_low_light_normalizers_are_deterministic_and_do_not_mutate_input():
    crop = np.full((160, 160, 3), 20, dtype=np.uint8)
    crop[:, 80:] = 145
    original = crop.copy()

    query = normalize_low_light_query(crop)
    reference = normalize_illumination_reference(crop)

    assert np.array_equal(crop, original)
    assert query.shape == crop.shape
    assert reference.shape == crop.shape
    assert query.dtype == np.uint8
    assert reference.dtype == np.uint8
    assert np.array_equal(query, normalize_low_light_query(crop))
    assert np.array_equal(reference, normalize_illumination_reference(crop))
    assert np.array_equal(query[:, :, 0], query[:, :, 1])
    assert np.array_equal(reference[:, :, 1], reference[:, :, 2])


def test_crop_square_face_preserves_shape_and_pads_frame_edges():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:50, :50] = (10, 80, 160)

    crop = crop_square_face(frame, (-5, -5, 45, 55))

    assert crop.shape == (160, 160, 3)
    assert crop.dtype == np.uint8
    assert np.any(crop != 0)


def test_yolo_parser_maps_letterboxed_box_back_to_source_frame():
    detector = object.__new__(YoloOnnxFaceDetector)
    detector._config = YoloOnnxConfig(model_path="unused.onnx")
    output = np.array(
        [
            [
                [320.0, 0.0],
                [320.0, 0.0],
                [200.0, 0.0],
                [200.0, 0.0],
                [0.9, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, confidences = detector._parse_yolo_output(
        output,
        frame_width=640,
        frame_height=480,
        scale=1.0,
        pad_x=0,
        pad_y=80,
    )

    assert boxes == [[220, 140, 200, 200]]
    assert confidences == pytest.approx([0.9])


def test_upper_face_mask_removes_lower_occlusion_without_mutating_crop():
    crop = np.full((160, 160, 3), 220, dtype=np.uint8)
    original = crop.copy()

    masked = mask_lower_face(crop, start_ratio=0.5)

    assert np.array_equal(crop, original)
    assert np.all(masked[:80] == 220)
    assert np.all(masked[80:] == 127)


def test_warped_face_crop_clamps_box_to_frame():
    frame = make_frame(width=120, height=80)

    crop = crop_warped_face(frame, (-10, -10, 60, 70))

    assert crop.shape == (160, 160, 3)
