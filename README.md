# Hệ Thống Nhận Diện Khuôn Mặt Thời Gian Thực (UTH Face AI)

Hệ thống nhận diện khuôn mặt đa điều kiện (góc nghiêng, đeo kính, đeo khẩu trang, ánh sáng yếu) theo thời gian thực sử dụng **FastAPI**, **PyTorch / ONNX (YOLOv8 & InceptionResnetV1)**, **PostgreSQL (pgvector)**, **Cloudinary** và giao diện **React (TypeScript + Tailwind CSS)**.

---

## 🌟 Tính Năng Nổi Bật

- **Phát hiện khuôn mặt tốc độ cao**: Sử dụng YOLOv8-Face (ONNX) kết hợp fallback YuNet / MTCNN.
- **Tách nền thông minh**: MediaPipe Selfie Segmenter loại bỏ nhiễu nền xung quanh khuôn mặt trước khi trích xuất vector đặc trưng.
- **Nhận diện đa điều kiện**:
  - Nhận diện trực diện và đa góc quay (trái, phải, ngước lên, cúi xuống).
  - Kháng che khuất: Hỗ trợ nhận diện khi đeo khẩu trang (bóc tách vùng nửa trên khuôn mặt).
  - Tối ưu hóa trong điều kiện ánh sáng yếu (Low-light enhancement).
- **Quy trình đăng ký thông minh (Multi-Pose Enrollment Wizard)**: Hướng dẫn người dùng chụp các góc và trạng thái khuôn mặt từng bước.
- **Tìm kiếm vector tức thì (Real-time Matching)**: In-memory Cosine Similarity Indexing kết hợp lưu trữ lâu dài với PostgreSQL `pgvector`.
- **Lưu trữ đám mây**: Tích hợp Cloudinary CDN lưu trữ ảnh mẫu và ảnh crop khuôn mặt.

---

## 🏗 Kiến Trúc Hệ Thống

```
+----------------------------+
|          Frontend          |
|  (React/Vite Webcam App)   |
+--------------+-------------+
               | Base64 Frame (Enroll / Identify)
               v
+----------------------------+
|          Backend           |
|     (FastAPI Service)      |
+-------+------------+-------+
        |            |
        |            | Query / Store embeddings
        v            v
+-------+----+  +----+-------+
| PyTorch ML |  | PostgreSQL |
| (CPU/CUDA) |  | (pgvector) |
+------------+  +------------+
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Backend (FastAPI)

```bash
# Di chuyển vào thư mục AI
cd AI

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Cấu hình biến môi trường trong AI/.env
# DATABASE_URL, CLOUDINARY_*, ...

# Chạy Database Migrations
python -m alembic upgrade head

# Khởi chạy server
python -m uvicorn app.main:app --reload
```
Server chạy tại: `http://127.0.0.1:8000`

### 2. Frontend (React + Vite)

```bash
# Di chuyển vào thư mục frontend
cd FE/frontend

# Cài đặt dependencies
npm install

# Chạy môi trường phát triển
npm run dev
```
Giao diện chạy tại: `http://localhost:5173`

---

## 📄 Bản Quyền & Giấy Phép
Dự án được phát triển phục vụ mục đích học tập và nghiên cứu.
