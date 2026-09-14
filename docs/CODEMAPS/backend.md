<!-- Generated: 2026-07-15 | Files scanned: 25 | Token estimate: ~550 -->
# Backend Architecture (AI Service)

## Overview
The backend is a FastAPI web application hosting the computer vision pipelines. It wraps PyTorch/FaceNet models to perform real-time face detection, alignment, and embedding extraction, and relies on SQLAlchemy to store and query PGVector embeddings.

## Key Files
- [main.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/main.py): FastAPI app initialization and router attachment.
- [routers/simplified_faces.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/routers/simplified_faces.py): Exposes HTTP POST endpoints for face enrollment and verification.
- [services/ai.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/services/ai.py): Loads torch models, manages hardware selection (CUDA/MPS/CPU), and implements the inference pipeline.
- [common/handlers.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/common/handlers.py): Defines global exception handling and middleware registration.

## Routing Mapping
All routes are prefixed under `/api/v1/faces`.

| Method | Endpoint | Handler | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | `root()` | Welcome check returning `{"message": "Hello World"}`. |
| `POST` | `/enroll` | `enroll(...)` | Decodes base64, runs MTCNN detection, extracts embedding, evaluates similarity, and inserts/updates records. |
| `POST` | `/identify` | `identify(...)` | Detects all faces in frame, extracts embeddings, matches against database, and returns mapped names. |

### Enrollment Lifecycle Decisions
- **No Face**: Return `HTTP 400 Bad Request`.
- **Multiple Faces**: Return `HTTP 400 Bad Request` (asks for a single clear face).
- **Match >= 85%**: Return `HTTP 409 Conflict` (already enrolled under another name).
- **Match 65% - 85%**: Return `status: CONFIRMATION_REQUIRED` along with candidate list.
- **Match < 65% / Forced / Person ID specified**: Save profile and return `status: SUCCESS`.

## Middleware & Handlers
- **TraceIdMiddleware**: Adds a random string `trace_id` to request scope and returns `x-trace-id` in the response header for API request tracking.
- **CORSMiddleware**: Registered to allow cross-origin requests from all hosts (`*`) to support local Vite frontend dev servers.
- **Global Error Interceptors**:
  - `NoFaceDetectedError` -> Returns `400` with code `NO_FACE_DETECTED`
  - `AmbiguousFaceError` -> Returns `400` with code `AMBIGUOUS_FACE`
  - `SQLAlchemyError` -> Returns `500` with code `DATABASE_ERROR`
  - `RequestValidationError` -> Returns `422` with code `VALIDATION_ERROR`

## ML Pipeline Specs
The system utilizes models from `facenet-pytorch` executing on CPU, Apple Silicon (MPS), or NVIDIA CUDA:
1. **MTCNN (Face Detection & Alignment)**:
   - Config: `image_size=160`, `margin=10`, `keep_all=False`.
   - Fallback detector: Instantiated via `build_face_detection_service(...)` using fallback threshold configuration `MTCNN_FALLBACK_FACE_MIN_CONFIDENCE=0.80`.
2. **InceptionResnetV1 (Feature Extraction)**:
   - Pre-trained weights: `vggface2`.
   - Output: 512-dimensional floating-point vector representing the face.
   - Vector comparison: Normalizes embedding vectors to unit length and computes dot-products. Maps cosine similarity range to user-friendly percentages.
3. **MediaPipe Selfie Segmenter (Face Segmentation)**:
   - Architecture: Pre-trained MobileNet-based U-Net architecture optimized for real-time portrait and facial skin segmentation.
   - Threshold: `probability >= 0.5` is classified as face skin.
   - Model file: Loaded from local asset `models/selfie_segmenter.tflite` (auto-downloaded from Google CDN on startup if missing).
   - Operation: Performs bitwise masking on the BGR cropped image to set background pixels to `[0, 0, 0]`.


