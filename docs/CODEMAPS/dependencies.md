<!-- Generated: 2026-07-15 | Files scanned: 25 | Token estimate: ~330 -->
# Project Dependencies & Libraries

## AI Backend Services (Python)
Dependencies are tracked in [requirements.txt](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/requirements.txt).

### Core Frameworks
- `fastapi` (v0.128.8): ASGI web API routing and schema validation via Pydantic.
- `uvicorn` (v0.47.0): Lightning-fast ASGI production-ready server.

### Deep Learning & Machine Vision
- `torch` (v2.2.2+cpu) & `torchvision` (v0.17.2+cpu): Base PyTorch platform used for model execution. Configured for CPU-only execution to minimize container overhead.
- `facenet-pytorch` (v2.6.0): Integrates pre-trained **MTCNN** face detection and **InceptionResnetV1** feature extractors.
- `opencv-python` (v4.9.0.80): BGR/RGB colorspace converters and image crop resizing operations.
- `pillow` (v10.2.0): PIL image format handling for MTCNN inputs.
- `numpy` (v1.26.4): Matrix algebra (dot products) for cosine similarity calculation.

### Database Integration
- `SQLAlchemy` (v2.0.49): Object-Relational Mapper (ORM) used to map objects to PostgreSQL tables.
- `pgvector` (v0.4.2): PostgreSQL vector extension connector for embedding query lookups.
- `alembic` (v1.18.4): Schema version management database migrations.
- `psycopg2` (v2.9.12): PostgreSQL database adapter.

---

## Frontend Web Application (Node.js)
Dependencies are tracked in [package.json](file:///d:/HocTap/CV/Code/uth_face_detection_ai/FE/frontend/package.json).

### Production Libraries
- `react` (v19.2.7) & `react-dom` (v19.2.7): User interface rendering framework.
- `react-webcam` (v7.2.0): Accesses local media devices and captures portrait screenshots.
- `lucide-react` (v1.23.0): Clean SVG icons for UI indicators.

### Development Toolchain
- `vite` (v8.1.1) & `@vitejs/plugin-react` (v6.0.3): Dev-server compiler and bundler.
- `typescript` (v6.0.2): Type safety verification compiler.
- `oxlint` (v1.71.0): High-speed JavaScript/TypeScript AST analysis linter.
- `tailwindcss` (v4.0.0) & `@tailwindcss/postcss` (v4.0.0): Styling framework and PostCSS plugin.
- `postcss` (v8.4.49) & `autoprefixer` (v10.4.20): CSS compiling and vendor prefix processors.

