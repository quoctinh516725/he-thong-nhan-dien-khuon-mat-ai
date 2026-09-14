<!-- Generated: 2026-07-15 | Files scanned: 25 | Token estimate: ~300 -->
# High-Level Architecture

## Overview
A minimalist Face Detection and Matching system utilizing a React/TypeScript Single Page Application (SPA) on the frontend and a FastAPI/PyTorch backend. The system captures face images, extracts high-dimensional embeddings using pre-trained convolutional neural networks, and queries or stores them in a PostgreSQL database using the `pgvector` extension.

## System Boundaries & Data Flow

```
                     +----------------------------+
                     |          Frontend          |
                     |  (React/Vite Webcam App)   |
                     +--------------+-------------+
                                    |
                                    | Base64 Image (Enroll / Identify)
                                    v
                     +----------------------------+
                     |          Backend           |
                     |     (FastAPI Service)      |
                     +-------+------------+-------+
                             |            |
             Extract crop    |            | Query / Store embeddings
             & embeddings    v            v
                     +-------+----+  +----+-------+
                     | PyTorch ML |  | PostgreSQL |
                     | (CPU Mod.) |  | (pgvector) |
                     +------------+  +------------+
```

### Flow 1: Face Enrollment
1. User inputs a name and captures a photo on the frontend.
2. The image is converted to Base64 and sent to `POST /api/v1/faces/enroll`.
3. The backend detects the face using **YOLOv8-Face** (or MTCNN fallback), segments the face from the background using **MediaPipe Selfie Segmenter** (which employs an optimized MobileNet-based U-Net architecture to black out the background), and extracts a 512-dimensional embedding using **InceptionResnetV1**.
4. The database is queried to ensure the person has not already been enrolled (similarity threshold >= 85%).
5. A new `Person` and associated `FaceRecord` are saved in PostgreSQL, and the segmented BGR face crop is saved locally in `AI/storage/processed_enroll/` for verification.

### Flow 2: Live CCTV Identification
1. The frontend CCTV camera loop runs every 100ms, capturing the current frame.
2. The frame is sent to `POST /api/v1/faces/identify`.
3. The backend detects all faces in the frame, crops them, runs them through **MediaPipe Selfie Segmenter** for background segmentation, and extracts embeddings from the segmented face crops.
4. Each embedding is compared to existing records in the database using dot-product cosine similarity.
5. Matches above 65% are mapped to the corresponding person; others are marked as "Unknown".
6. The frontend overlays green (identified) or red (unknown) boxes on the webcam feed.

## Core Directories & Entry Points
- **Frontend SPA**: [App.tsx](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/src/App.tsx) - Handles UI, webcam feeds, canvas drawing, and API communication.
- **FastAPI Entrypoint**: [main.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/main.py) - Boots the FastAPI app, registers routes, and configures CORS.
- **API Endpoints**: [simplified_faces.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/routers/simplified_faces.py) - Endpoint handler logic for enrollment (logs pipeline progress and saves crops) and identification.
- **AI Models & Pipeline**: [ai.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/services/ai.py) - Manages model execution (YOLOv8, MTCNN, MediaPipe Segmenter, and InceptionResnetV1) on CPU/GPU.
- **U-Net Reference Structure**: [unet.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/vision/unet.py) - Defines the PyTorch U-Net model structure (kept as a reference architecture).
- **Database Engine**: [db.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/database/db.py) - Handles SQLAlchemy engine and PostgreSQL session lifetime.
