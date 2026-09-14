<!-- Generated: 2026-07-15 | Files scanned: 25 | Token estimate: ~330 -->
# Database Schema & Data Models

## Overview
The system uses PostgreSQL with the `pgvector` extension. SQLAlchemy manages object-relational mapping, and Alembic handles schema migrations.

## Entity-Relationship Diagram

```
       +-----------------------+              +-----------------------+
       |        persons        |              |     face_records      |
       +-----------------------+              +-----------------------+
       | id (UUID, PK)         |<------------+  | id (UUID, PK)         |
       | person_code (BIGINT)  | 1       0..* | person_id (UUID, FK)  |
       | name (VARCHAR)        |              | image_path (VARCHAR)  |
       | created_at (DATETIME) |              | pose_label (VARCHAR)  |
       +-----------------------+              | embedding (VECTOR512) |
                                              | created_at (DATETIME) |
                                              +-----------------------+
```

## Database Tables

### 1. `persons`
Stores identity records of enrolled individuals.
- **Source**: [person.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/models/entities/person.py)

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, Index, Unique | Randomly generated UUID. |
| `person_code` | `BIGINT` | Identity (Auto-increment), Unique, Index | Numeric ID for human reference. |
| `name` | `VARCHAR` | Non-nullable | Full name of the user. |
| `created_at` | `DATETIME` | Default: UTC Now | Timestamp of registration. |

### 2. `face_records`
Stores multi-angle facial embeddings associated with a person.
- **Source**: [face_record.py](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/app/models/entities/face_record.py)

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, Index, Unique | Face record unique identifier. |
| `person_id` | `UUID` | ForeignKey (`persons.id`), Index | Belongs to person (Cascade Delete). |
| `image_path` | `VARCHAR` | Nullable | Path to file storage (or camera label). |
| `pose_label` | `VARCHAR` | Nullable, Index | Angle label (e.g., front, left, right). |
| `embedding` | `VECTOR(512)` | Non-nullable (pgvector) | Normalized FaceNet output vector. |
| `created_at` | `DATETIME` | Default: UTC Now | Timestamp of face addition. |

## Alembic Migrations
Migrations are stored in [alembic/versions/](file:///d:/HocTap/CV/Code/uth_face_detection_ai/AI/alembic/versions).

1. `2b502843068b_update`: Configures base schemas.
2. `7c4c8fdc9d21_add_person_identity_number`: Introduces ID mapping dependencies.
3. `c01e9842fd7a_add_person_code`: Creates identity `person_code` sequences.
4. `d7f9b2c14e6a_add_face_record_pose_label`: Adds face record pose categories to allow registering different side profile views.
5. `e8a9f3b14d2e_drop_id_mapping`: Drops the legacy `id_mapping` table and its indexes.
