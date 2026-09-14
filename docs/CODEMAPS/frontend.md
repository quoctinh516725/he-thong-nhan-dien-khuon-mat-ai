<!-- Generated: 2026-07-15 | Files scanned: 25 | Token estimate: ~450 -->
# Frontend Architecture (Vite React Client)

## Overview
The frontend is a single-page React application written in TypeScript and styled using Tailwind CSS and custom stylesheets. It enables users to view a live camera feed for recognition or capture portraits for registering new profiles.

## Key Files
- [main.tsx](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/src/main.tsx): SPA mount point.
- [App.tsx](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/src/App.tsx): Contains all application views, reactive state, webcam captures, and canvas drawing logic.
- [index.html](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/index.html): HTML skeleton.
- [index.css](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/src/index.css) & [App.css](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/src/App.css): Layout and typography styles.

## Page Tree & State Flow

```
                  +--------------------------------+
                  |         App Layout             |
                  |  (Tabs: cctv / enroll state)   |
                  +---------------+----------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
  +------+---------------+                          +------+---------------+
  |   CCTV Scanner Tab   |                          | Face Enrollment Tab  |
  |  (Camera Live Feed)  |                          | (Capture & Register) |
  +------+---------------+                          +------+---------------+
         |                                                 |
         | Capture frame 1s/interval                       | Snapshot & Name input
         v                                                 v
  +------+---------------+                          +------+---------------+
  | POST /faces/identify |                          |   POST /faces/enroll |
  +------+---------------+                          +------+---------------+
         |                                                 |
         | Bounding Box JSON                               | JSON Status
         v                                                 v
  +------+---------------+                          +------+---------------+
  | Draw Canvas Overlay  |                          | Success/Error/Prompt |
  +----------------------+                          +----------------------+
```

## View Tabs Details

### 1. Giám Sát Camera (CCTV Scanner)
- **Webcam & Bounding Boxes**: Renders a live stream using `<Webcam>` and places a `<canvas>` element directly over it to draw bounding boxes and names.
- **Interval Scan Loop**: A `useEffect` hook runs an async scanner every 1 second when active.
- **Render Engine**:
  - Identified: Green box (`#22c55e`), label `[Name] ([Score]%)`.
  - Unknown: Red box (`#ef4444`), label `Unknown`.

### 2. Thêm Dữ Liệu Gương Mặt (Enrollment Form)
- **Snapshot capture**: Binds to `<Webcam>` to screenshot a JPEG image string.
- **Identity Fields**: Captures the individual's full name.
- **Conflict Handling Screen**:
  - If backend reports `CONFIRMATION_REQUIRED` (suspicious matches between 65% and 85%), renders a list of potential matching profiles.
  - Clicking "Đó là tôi" updates the selected existing `person_id`.
  - Clicking "Không phải tôi, vẫn tạo profile mới" sends a request with `force: true` to bypass the similarity checks.
