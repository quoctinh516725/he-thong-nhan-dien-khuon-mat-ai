import base64
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session

from app.models.entities.person import Person
from app.models.entities.face_record import FaceRecord
from app.services.object_storage import get_image_storage

# In-memory dictionary mapping person_id (str) to avatar base64 string (str)
avatar_cache: dict[str, str] = {}
avatar_reference_cache: dict[str, str] = {}
face_record_reference_cache: dict[str, str] = {}

def _image_to_base64(file_path: str) -> str:
    """Converts a local image file to a base64 encoded string."""
    if not file_path:
        return ""
        
    try:
        return base64.b64encode(
            get_image_storage().read_bytes(file_path)
        ).decode("utf-8")
    except Exception as e:
        print(f"ERROR encoding stored avatar {file_path}: {str(e)}")
        return ""

def load_avatar_cache(db: Session):
    """Loads all face records from the DB and caches their avatars in RAM."""
    print("Loading Registered Face Avatar Cache into RAM...")
    
    try:
        avatar_cache.clear()
        avatar_reference_cache.clear()
        face_record_reference_cache.clear()
        # Query all records that have an image_path
        records = db.query(FaceRecord).filter(FaceRecord.image_path.isnot(None)).all()
        first_record_by_person = {}
        for record in records:
            person_id = str(record.person_id)
            face_record_reference_cache[str(record.id)] = record.image_path
            first_record_by_person.setdefault(person_id, record.image_path)

        with ThreadPoolExecutor(max_workers=8) as executor:
            encoded = executor.map(_image_to_base64, first_record_by_person.values())
        loaded_count = 0
        for p_id, base64_str in zip(first_record_by_person, encoded):
            if base64_str:
                avatar_cache[p_id] = base64_str
                avatar_reference_cache[p_id] = first_record_by_person[p_id]
                loaded_count += 1
                
        print(f"Successfully cached {loaded_count} face avatars in RAM.")
    except Exception as e:
        print(f"WARNING: Failed to load avatar cache from database: {str(e)}")

def set_avatar_in_cache(
    person_id: str,
    file_path: str,
    record_id: str | None = None,
):
    """Encodes and adds/updates a person's avatar in the memory cache."""
    global avatar_cache
    base64_str = _image_to_base64(file_path)
    if base64_str:
        avatar_cache[person_id] = base64_str
        avatar_reference_cache[person_id] = file_path
        if record_id:
            face_record_reference_cache[record_id] = file_path
        print(f"Cached avatar for person {person_id} in RAM.")
    else:
        print(f"WARNING: Could not cache avatar for person {person_id} (file not found or empty).")
