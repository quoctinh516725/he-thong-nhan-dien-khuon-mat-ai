from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from app.vision.exceptions import InvalidFrameError, ProcessingError, VisionError


class BaseImageProcessor(ABC):
    """Base class for image processors.

    Processors receive and return BGR `uint8` images with shape `(H, W, C)`.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a BGR image and return a new image."""


@dataclass(frozen=True)
class ClaheConfig:
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class FaceCropConfig:
    output_size: int = 160
    margin_ratio: float = 0.12


class BgrClaheProcessor(BaseImageProcessor):
    """Applies CLAHE on the luminance channel of a BGR frame."""

    def __init__(self, config: Optional[ClaheConfig] = None) -> None:
        self._config = config or ClaheConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        validate_bgr_frame(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._config.clip_limit,
            tileGridSize=self._config.tile_grid_size,
        )
        enhanced = clahe.apply(lightness)
        merged = cv2.merge((enhanced, channel_a, channel_b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class FacePreprocessingPipeline:
    """Runs configurable preprocessing before face detection."""

    def __init__(self, processors: Optional[Sequence[BaseImageProcessor]] = None) -> None:
        self._processors = list(processors) if processors is not None else [BgrClaheProcessor()]

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Run preprocessing on a BGR `uint8` frame without mutating the source."""

        validate_bgr_frame(frame)
        current_frame = frame.copy()

        for index, processor in enumerate(self._processors):
            try:
                current_frame = processor.process(current_frame)
            except VisionError as exc:
                raise ProcessingError(
                    f"Preprocessing failed at step {index} "
                    f"({processor.__class__.__name__}): {exc}"
                ) from exc
            except Exception as exc:
                raise ProcessingError(
                    f"Unexpected preprocessing failure at step {index} "
                    f"({processor.__class__.__name__}): {exc}"
                ) from exc

        return current_frame


def crop_square_face(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    config: Optional[FaceCropConfig] = None,
) -> np.ndarray:
    """Crop a padded square face region without changing facial proportions."""

    validate_bgr_frame(frame)
    crop_config = config or FaceCropConfig()
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise InvalidFrameError("Face bounding box must have a positive area.")

    frame_height, frame_width = frame.shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * crop_config.margin_ratio)

    crop_x1 = int(np.floor(center_x - side / 2.0))
    crop_y1 = int(np.floor(center_y - side / 2.0))
    crop_x2 = int(np.ceil(center_x + side / 2.0))
    crop_y2 = int(np.ceil(center_y + side / 2.0))

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - frame_width)
    pad_bottom = max(0, crop_y2 - frame_height)

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_width, crop_x2)
    crop_y2 = min(frame_height, crop_y2)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise InvalidFrameError("Face bounding box is outside the frame.")

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    interpolation = (
        cv2.INTER_AREA
        if max(crop.shape[:2]) > crop_config.output_size
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        crop,
        (crop_config.output_size, crop_config.output_size),
        interpolation=interpolation,
    )


def crop_warped_face(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    output_size: int = 160,
) -> np.ndarray:
    """Reproduce the original tight crop for compatibility descriptors."""

    validate_bgr_frame(frame)
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def mask_lower_face(
    crop: np.ndarray,
    start_ratio: float = 0.50,
    fill_value: int = 127,
) -> np.ndarray:
    """Create a stable upper-face view by neutralizing mouth and chin pixels."""

    validate_bgr_frame(crop)
    if not 0.0 < start_ratio < 1.0:
        raise InvalidFrameError("Lower-face mask ratio must be between 0 and 1.")
    masked = crop.copy()
    masked[int(round(masked.shape[0] * start_ratio)) :] = fill_value
    return masked


def is_low_light_face(
    crop: np.ndarray,
    mean_threshold: float = 80.0,
    median_threshold: float = 55.0,
) -> bool:
    """Detect underexposure while ignoring a small flashlight hotspot."""

    validate_bgr_frame(crop)
    luminance = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return bool(
        float(luminance.mean()) < mean_threshold
        and float(np.median(luminance)) < median_threshold
    )


def normalize_low_light_query(crop: np.ndarray) -> np.ndarray:
    """Create a contrast-stable grayscale view of an underexposed face."""

    validate_bgr_frame(crop)
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.createCLAHE(
        clipLimit=4.0,
        tileGridSize=(4, 4),
    ).apply(grayscale)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def normalize_illumination_reference(crop: np.ndarray) -> np.ndarray:
    """Build a Tan-Triggs-style reference view for illumination matching."""

    validate_bgr_frame(crop)
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gamma_corrected = np.power(grayscale, 0.3)
    contrast = cv2.GaussianBlur(gamma_corrected, (0, 0), 0.8) - cv2.GaussianBlur(
        gamma_corrected,
        (0, 0),
        3.0,
    )
    alpha = 0.1
    tau = 10.0
    first_scale = np.mean(np.power(np.abs(contrast), alpha)) ** (1.0 / alpha)
    contrast = contrast / max(float(first_scale), 1e-6)
    second_scale = np.mean(
        np.power(np.minimum(np.abs(contrast), tau), alpha)
    ) ** (1.0 / alpha)
    contrast = contrast / max(float(second_scale), 1e-6)
    contrast = tau * np.tanh(contrast / tau)
    normalized = np.clip((contrast / (2.0 * tau) + 0.5) * 255.0, 0, 255)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def classify_head_pose(yaw: float, pitch: float) -> str:
    """Map normalized five-point landmark geometry to a coarse capture pose."""
    if yaw <= -0.16:
        return "left"
    if yaw >= 0.16:
        return "right"
    if pitch <= 0.49:
        return "up"
    if pitch >= 0.67:
        return "down"
    if abs(yaw) <= 0.135 and 0.51 <= pitch <= 0.65:
        return "front"
    return "unknown"


def orient_pose_for_preview(pose: str, mirrored: bool) -> str:
    """Translate raw-camera yaw into the direction shown in a mirrored preview."""
    if not mirrored:
        return pose
    return {"left": "right", "right": "left"}.get(pose, pose)


def validate_bgr_frame(frame: Optional[np.ndarray]) -> None:
    """Validate a BGR `uint8` frame with shape `(H, W, 3)`."""

    if frame is None:
        raise InvalidFrameError("Input frame cannot be None.")
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError("Input frame must be a numpy array.")
    if frame.size == 0:
        raise InvalidFrameError("Input frame cannot be empty.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InvalidFrameError("Input frame must have shape (H, W, 3).")
    if frame.dtype != np.uint8:
        raise InvalidFrameError("Input frame must use uint8 pixels.")
