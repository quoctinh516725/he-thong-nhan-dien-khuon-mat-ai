class VisionError(Exception):
    """Base exception for computer vision pipeline failures."""


class InvalidFrameError(VisionError):
    """Raised when an input frame is missing, empty, or has an unsupported shape."""


class ProcessingError(VisionError):
    """Raised when an image processing step fails."""


class NoFaceDetectedError(VisionError):
    """Raised when no face passes the configured detector threshold."""


class AmbiguousFaceError(VisionError):
    """Raised when multiple similarly sized faces make the primary face ambiguous."""


class DetectionModelError(VisionError):
    """Raised when the detector model cannot be initialized or executed."""
