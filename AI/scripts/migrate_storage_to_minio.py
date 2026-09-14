from pathlib import Path

from app.database.db import SessionLocal
from app.models.entities.face_record import FaceRecord
from app.models.entities.person import Person  # noqa: F401 - registers ORM relationship
from app.services.object_storage import ARTIFACT_PREFIXES, get_image_storage


def related_paths(reference: str) -> set[str]:
    paths = {reference}
    for source_prefix in ARTIFACT_PREFIXES:
        if source_prefix in reference:
            paths.update(
                reference.replace(source_prefix, target_prefix)
                for target_prefix in ARTIFACT_PREFIXES
            )
    return paths


def main() -> None:
    storage = get_image_storage()
    if storage.config.backend != "minio":
        raise RuntimeError("STORAGE_BACKEND must be minio for migration.")
    storage.ensure_ready()

    db = SessionLocal()
    migrated = 0
    skipped = 0
    try:
        records = db.query(FaceRecord).order_by(FaceRecord.created_at.asc()).all()
        for record in records:
            old_reference = record.image_path
            if not old_reference or old_reference.startswith("minio://"):
                skipped += 1
                continue

            artifacts = {}
            segmented_key = None
            for old_path in related_paths(old_reference):
                image = storage.read_image(old_path)
                if image is None:
                    continue
                object_key = f"enroll/migrated/{record.id}/{Path(old_path).name}"
                artifacts[object_key] = image
                if "4_segmented_crop_" in object_key:
                    segmented_key = object_key
            if not artifacts or not segmented_key:
                print(f"SKIP {record.id}: segmented source image is missing")
                skipped += 1
                continue

            references = {}
            try:
                references = storage.put_jpeg_batch(artifacts)
                record.image_path = references[segmented_key]
                db.commit()
            except Exception:
                db.rollback()
                if segmented_key in references:
                    storage.delete_related(references[segmented_key])
                raise
            storage.delete_related(old_reference)
            migrated += 1
            print(f"MIGRATED {record.id}: {len(artifacts)} objects")
    finally:
        db.close()
    print(f"Migration complete: migrated={migrated}, skipped={skipped}")


if __name__ == "__main__":
    main()
