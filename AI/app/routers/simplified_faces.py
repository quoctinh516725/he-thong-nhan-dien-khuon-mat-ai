import os
import time
import numpy as np
from typing import List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database.db import get_db
from app.models.entities.person import Person
from app.models.entities.face_record import FaceRecord
from app.services.image_input import decode_image
from app.services.ai import (
    analyze_enrollment_face,
    detect_enrollment_face,
    detect_faces_with_fallback,
    extract_face_crop_tensor,
    extract_illumination_reference_tensor,
    extract_low_light_face_tensor,
    extract_upper_face_tensor,
    get_embeddings,
    is_low_light_face_region,
)
from app.vision.exceptions import AmbiguousFaceError, NoFaceDetectedError, VisionError
from app.services.avatar_cache import (
    avatar_cache,
    avatar_reference_cache,
    face_record_reference_cache,
    set_avatar_in_cache,
)
from app.services.face_index import (
    FaceEmbeddingIndex,
    IndexedIdentity,
    face_embedding_index,
    load_face_embedding_index,
    low_light_embedding_index,
    upper_face_embedding_index,
)
from app.services.object_storage import get_image_storage
from app.vision.preprocessing import orient_pose_for_preview

router = APIRouter(prefix="/api/v1/faces", tags=["simplified-faces"])

# Request/Response schemas
class EnrollRequest(BaseModel):
    name: str
    image_base64: str
    person_id: Optional[str] = None
    pose_label: Optional[str] = None

class EnrollResponse(BaseModel):
    status: str
    person_id: Optional[str] = None
    name: Optional[str] = None
    message: str

class IdentifyFaceMatch(BaseModel):
    box: List[int]  # [x_min, y_min, x_max, y_max]
    identified: bool
    person_id: Optional[str] = None
    name: str
    score: float
    avatar_base64: Optional[str] = None
    avatar_url: Optional[str] = None
    match_mode: Optional[str] = None

class IdentifyResponse(BaseModel):
    faces_detected: int
    matches: List[IdentifyFaceMatch]


class ProfileResponse(BaseModel):
    person_id: str
    name: str
    sample_count: int = 0
    sample_layers: List[str] = Field(default_factory=list)
    avatar_url: Optional[str] = None


class ProfileSampleResponse(BaseModel):
    record_id: str
    pose_label: Optional[str] = None
    image_url: Optional[str] = None


class ProfileDetailResponse(ProfileResponse):
    samples: List[ProfileSampleResponse]


class EnrollmentAnalyzeRequest(BaseModel):
    image_base64: str
    expected_pose: Literal["front", "left", "right", "up", "down"]
    mirrored: bool = False


class EnrollmentAnalyzeResponse(BaseModel):
    face_count: int
    pose: str
    ready: bool
    reason: str
    box: Optional[List[int]] = None
    brightness: Optional[float] = None
    sharpness: Optional[float] = None
    face_ratio: Optional[float] = None


@router.get("/profiles", response_model=List[ProfileResponse])
def list_profiles(db: Session = Depends(get_db)):
    people = db.query(Person).order_by(Person.name.asc()).all()
    return [
        ProfileResponse(
            person_id=str(person.id),
            name=person.name,
            sample_count=len(person.face_records),
            sample_layers=sorted({
                _pose_layer(record.pose_label)
                for record in person.face_records
            }),
            avatar_url=(
                f"/api/v1/faces/profiles/{person.id}/avatar"
                if person.face_records else None
            ),
        )
        for person in people
    ]


def _pose_layer(pose_label: Optional[str]) -> str:
    if pose_label and pose_label.startswith("glasses_mask_"):
        return "glasses_mask"
    if pose_label and pose_label.startswith("glasses_"):
        return "glasses"
    if pose_label and pose_label.startswith("mask_"):
        return "mask"
    return "plain"


def _read_stored_image(image_path: Optional[str]) -> bytes:
    if not image_path:
        raise HTTPException(status_code=404, detail="Mẫu ảnh không tồn tại.")
    try:
        return get_image_storage().read_bytes(image_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Không đọc được mẫu ảnh.") from exc


def _delete_stored_images(image_path: Optional[str]) -> None:
    try:
        get_image_storage().delete_related(image_path)
    except Exception as exc:
        print(f"WARNING: Could not delete stored enrollment artifacts: {exc}")


def _reload_runtime_face_data(db: Session) -> None:
    face_embedding_index.replace([])
    upper_face_embedding_index.replace([])
    low_light_embedding_index.replace([])
    load_face_embedding_index(db)


@router.get("/profiles/samples/{record_id}/image")
def get_profile_sample_image(record_id: str, db: Session = Depends(get_db)):
    reference = face_record_reference_cache.get(record_id)
    if reference is None:
        record = db.query(FaceRecord).filter(FaceRecord.id == record_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Không tìm thấy mẫu khuôn mặt.")
        reference = record.image_path
        face_record_reference_cache[record_id] = reference
    return Response(
        content=_read_stored_image(reference),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=86400, immutable",
            "ETag": f'"{record_id}"',
        },
    )


@router.get("/profiles/{person_id}/avatar")
def get_profile_avatar(person_id: str, db: Session = Depends(get_db)):
    reference = avatar_reference_cache.get(person_id)
    if reference is None:
        record = db.query(FaceRecord).filter(
            FaceRecord.person_id == person_id,
        ).order_by(FaceRecord.created_at.asc()).first()
        if not record:
            raise HTTPException(status_code=404, detail="Hồ sơ chưa có ảnh đại diện.")
        reference = record.image_path
        avatar_reference_cache[person_id] = reference
    return Response(
        content=_read_stored_image(reference),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/profiles/{person_id}", response_model=ProfileDetailResponse)
def get_profile_detail(person_id: str, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ.")
    samples = [
        ProfileSampleResponse(
            record_id=str(record.id),
            pose_label=record.pose_label,
            image_url=f"/api/v1/faces/profiles/samples/{record.id}/image",
        )
        for record in sorted(person.face_records, key=lambda item: str(item.id))
    ]
    return ProfileDetailResponse(
        person_id=str(person.id),
        name=person.name,
        sample_count=len(samples),
        sample_layers=sorted({_pose_layer(record.pose_label) for record in person.face_records}),
        avatar_url=(f"/api/v1/faces/profiles/{person.id}/avatar" if samples else None),
        samples=samples,
    )


@router.delete("/profiles/{person_id}/samples/{record_id}")
def delete_profile_sample(person_id: str, record_id: str, db: Session = Depends(get_db)):
    record = db.query(FaceRecord).filter(
        FaceRecord.id == record_id,
        FaceRecord.person_id == person_id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu khuôn mặt.")
    image_path = record.image_path
    db.delete(record)
    db.commit()
    _delete_stored_images(image_path)
    face_record_reference_cache.pop(record_id, None)
    avatar_cache.pop(person_id, None)
    avatar_reference_cache.pop(person_id, None)
    replacement = db.query(FaceRecord).filter(FaceRecord.person_id == person_id).first()
    if replacement:
        set_avatar_in_cache(person_id, replacement.image_path, str(replacement.id))
    _reload_runtime_face_data(db)
    return {"status": "SUCCESS", "message": "Đã xóa mẫu khuôn mặt."}


@router.delete("/profiles/{person_id}")
def delete_profile(person_id: str, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ.")
    image_paths = [record.image_path for record in person.face_records]
    record_ids = [str(record.id) for record in person.face_records]
    db.delete(person)
    db.commit()
    for image_path in image_paths:
        _delete_stored_images(image_path)
    for record_id in record_ids:
        face_record_reference_cache.pop(record_id, None)
    avatar_cache.pop(person_id, None)
    avatar_reference_cache.pop(person_id, None)
    _reload_runtime_face_data(db)
    return {"status": "SUCCESS", "message": "Đã xóa hồ sơ và toàn bộ mẫu khuôn mặt."}


@router.delete("/profiles")
def reset_profiles(confirm: str, db: Session = Depends(get_db)):
    if confirm != "RESET":
        raise HTTPException(status_code=400, detail="Thiếu xác nhận RESET.")
    db.query(FaceRecord).delete(synchronize_session=False)
    db.query(Person).delete(synchronize_session=False)
    db.commit()
    get_image_storage().delete_all()
    avatar_cache.clear()
    avatar_reference_cache.clear()
    face_record_reference_cache.clear()
    _reload_runtime_face_data(db)
    return {"status": "SUCCESS", "message": "Đã reset toàn bộ dữ liệu khuôn mặt."}


@router.post("/enrollment/analyze", response_model=EnrollmentAnalyzeResponse)
def analyze_enrollment_frame(request: EnrollmentAnalyzeRequest):
    try:
        image = decode_image(None, request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể giải mã hình ảnh Base64.")
    analysis = analyze_enrollment_face(image)
    display_pose = orient_pose_for_preview(analysis["pose"], request.mirrored)
    pose_matches = display_pose == request.expected_pose
    ready = bool(analysis["quality_ready"] and pose_matches)
    reason = analysis["reason"]
    if analysis["quality_ready"] and not pose_matches:
        reason = "Tiếp tục xoay đầu theo hướng chỉ dẫn."
    return EnrollmentAnalyzeResponse(
        face_count=analysis["face_count"],
        pose=display_pose,
        ready=ready,
        reason=reason,
        box=analysis.get("box"),
        brightness=analysis.get("brightness"),
        sharpness=analysis.get("sharpness"),
        face_ratio=analysis.get("face_ratio"),
    )

# Helper to calculate similarity percent from Cosine Similarity
def _get_similarity_percent(cosine_sim: float) -> float:
    # Scale cosine similarity (-1.0 to 1.0) into simple percentage (0.0 to 100.0)
    # Cosine of 0.75 translates directly to 75.0% on the UI
    return round(cosine_sim * 100, 2)

# Helper to find matches in DB using Cosine Similarity
def _search_db(
    embedding: np.ndarray,
    db: Session,
    threshold: float = 0.0,
) -> List[tuple[IndexedIdentity, float]]:
    records = db.query(FaceRecord).join(FaceRecord.person).all()
    fresh_index = FaceEmbeddingIndex()
    fresh_index.replace(
        (str(record.person.id), record.person.name, record.embedding)
        for record in records
    )
    return [
        (identity, _get_similarity_percent(cosine_similarity))
        for identity, cosine_similarity in fresh_index.search(
            np.asarray(embedding, dtype=np.float32)[None, :],
            threshold,
        )[0]
    ]


def _search_db_batch(
    embeddings: np.ndarray,
    db: Session,
    threshold: float = 0.0,
) -> List[List[tuple[IndexedIdentity, float]]]:
    del db
    return [
        [
            (identity, _get_similarity_percent(cosine_similarity))
            for identity, cosine_similarity in matches
        ]
        for matches in face_embedding_index.search(embeddings, threshold)
    ]


def _search_upper_face_batch(embeddings: np.ndarray):
    return [
        [
            (identity, _get_similarity_percent(cosine_similarity))
            for identity, cosine_similarity in matches
        ]
        for matches in upper_face_embedding_index.search_consensus(
            embeddings,
            threshold=0.0,
            support_count=2,
        )
    ]


def _search_low_light_batch(embeddings: np.ndarray):
    return [
        [
            (identity, _get_similarity_percent(cosine_similarity))
            for identity, cosine_similarity in matches
        ]
        for matches in low_light_embedding_index.search_consensus(
            embeddings,
            threshold=0.0,
            support_count=2,
        )
    ]

@router.post("/enroll", response_model=EnrollResponse)
def enroll(request: EnrollRequest, db: Session = Depends(get_db)):
    try:
        image = decode_image(None, request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể giải mã hình ảnh Base64.")

    # 1. Detect faces in the image
    try:
        primary_face, _ = detect_enrollment_face(image)
    except AmbiguousFaceError:
        raise HTTPException(status_code=400, detail="Phát hiện nhiều hơn 1 gương mặt trong ảnh. Vui lòng chỉ chụp duy nhất 1 người để đăng ký!")
    except NoFaceDetectedError:
        raise HTTPException(status_code=400, detail="Không phát hiện gương mặt trong ảnh. Vui lòng chụp lại ảnh rõ nét hơn!")
    except VisionError as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {str(exc)}")

    # 2. Extract embedding and save segmented image
    face_crop, yolo_crop, unet_mask, masked_img = extract_face_crop_tensor(image, primary_face.box)
    embedding, upper_embedding, low_light_embedding = get_embeddings([
        face_crop,
        extract_upper_face_tensor(image, primary_face.box),
        extract_illumination_reference_tensor(yolo_crop),
    ])
    embedding_list = embedding.astype("float32").tolist()

    # Log and save processed images for enrollment flow
    print(f"=== PIPELINE LOGS FOR ENROLLMENT ===")
    print(f"1. CV2 DECODE: BGR Image shape = {image.shape}")
    print(f"2. YOLO FACE DETECTED: Bounding Box = {primary_face.box}")
    print(f"3. U-NET FACE SEGMENTATION: Masked Face size = {masked_img.shape}")
    
    # Calculate face skin pixels ratio
    non_zero_px = np.count_nonzero(masked_img)
    total_px = masked_img.size
    print(f"   - Segmented face pixel ratio: {non_zero_px / total_px * 100:.2f}% face skin, {(total_px - non_zero_px) / total_px * 100:.2f}% background removed")
    
    # Prepare deterministic object keys; the upload happens after identity checks.
    timestamp = time.time_ns()
    
    # Strip accents for ASCII-safe filenames on Windows/Linux
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', request.name)
    ascii_name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    safe_name = "".join([c if c.isalnum() else "_" for c in ascii_name])
    
    object_prefix = f"enroll/{timestamp}"
    artifact_images = {
        f"{object_prefix}/1_decoded_input_{safe_name}_{timestamp}.jpg": image,
        f"{object_prefix}/2_yolo_crop_{safe_name}_{timestamp}.jpg": yolo_crop,
        f"{object_prefix}/3_unet_mask_{safe_name}_{timestamp}.jpg": unet_mask,
        f"{object_prefix}/4_segmented_crop_{safe_name}_{timestamp}.jpg": masked_img,
    }
    segmented_key = next(
        key for key in artifact_images if "4_segmented_crop_" in key
    )

    print(f"4. FACENET EMBEDDING: Vector dimension = {len(embedding_list)}")
    print(f"   - Embedding sample (first 10 values): {embedding_list[:10]}")
    print(f"=====================================")

    # 3. Check identity consistency before creating or extending a profile.
    enroll_threshold = float(os.getenv("FACE_ENROLL_MIN_CONFIDENCE", "0.76"))
    matches = _search_db(embedding, db, threshold=enroll_threshold)
    load_face_embedding_index(db)
    upper_matches = _search_upper_face_batch(upper_embedding[None, :])[0]
    partial_threshold = float(os.getenv("FACE_PARTIAL_MIN_CONFIDENCE", "0.76")) * 100
    partial_margin = float(os.getenv("FACE_PARTIAL_MIN_MARGIN", "0.06")) * 100
    upper_duplicate = None
    if upper_matches:
        upper_best_score = upper_matches[0][1]
        upper_second_score = upper_matches[1][1] if len(upper_matches) > 1 else 0.0
        if (
            upper_best_score >= partial_threshold
            and upper_best_score - upper_second_score >= partial_margin
        ):
            upper_duplicate = upper_matches[0]

    if request.person_id:
        conflicting_match = None
        if matches and matches[0][1] >= (enroll_threshold * 100):
            conflicting_match = matches[0]
        elif upper_duplicate:
            conflicting_match = upper_duplicate
        if conflicting_match and conflicting_match[0].person_id != request.person_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Ảnh này nhận diện rõ là "
                    f"{conflicting_match[0].name}, không khớp hồ sơ đã chọn."
                ),
            )

    if not request.person_id and (matches or upper_duplicate):
        match_str = ""
        if matches:
            match_str += f"Khớp toàn mặt: {matches[0][0].name} ({matches[0][1]}%)"
        if upper_duplicate:
            if match_str:
                match_str += " | "
            match_str += f"Khớp nửa trên: {upper_duplicate[0].name} ({upper_duplicate[1]}%)"

        print(f"=== ENROLL DUPLICATE DETECTED ===")
        print(f"Matches (Full face): {[(m[0].name, m[1]) for m in matches] if matches else []}")
        print(f"Upper matches: {[(m[0].name, m[1]) for m in upper_matches] if upper_matches else []}")
        print(f"Upper duplicate: {(upper_duplicate[0].name, upper_duplicate[1]) if upper_duplicate else None}")
        print(f"=================================")

        raise HTTPException(
            status_code=409,
            detail=f"Khuôn mặt khá giống dữ liệu đã có. Hãy chọn hồ sơ hiện có để bổ sung mẫu. ({match_str})",
        )

    existing_person = None
    if request.person_id:
        existing_person = db.query(Person).filter(Person.id == request.person_id).first()
        if not existing_person:
            raise HTTPException(
                status_code=404,
                detail="Hồ sơ người dùng cần cập nhật không tồn tại.",
            )

    try:
        stored_artifacts = get_image_storage().put_jpeg_batch(artifact_images)
        img_filename_4 = stored_artifacts[segmented_key]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể lưu ảnh lên object storage: {str(exc)}",
        ) from exc

    # 4. Create new person or link to existing person_id
    try:
        if request.person_id:
            new_record = FaceRecord(
                person_id=existing_person.id,
                image_path=img_filename_4,
                embedding=embedding_list,
                pose_label=request.pose_label,
            )
            db.add(new_record)
            db.commit()
            face_embedding_index.add(
                str(existing_person.id),
                existing_person.name,
                embedding,
            )
            upper_face_embedding_index.add(
                str(existing_person.id),
                existing_person.name,
                upper_embedding,
            )
            low_light_embedding_index.add(
                str(existing_person.id),
                existing_person.name,
                low_light_embedding,
            )
            
            # Sync cache in memory immediately
            set_avatar_in_cache(
                str(existing_person.id),
                img_filename_4,
                str(new_record.id),
            )
            
            db.refresh(existing_person)
            return EnrollResponse(
                status="SUCCESS",
                person_id=str(existing_person.id),
                name=existing_person.name,
                message="Cập nhật thêm dữ liệu góc mặt cho hồ sơ thành công."
            )
        else:
            # Create new person
            new_person = Person(name=request.name)
            db.add(new_person)
            db.flush()

            new_record = FaceRecord(
                person_id=new_person.id,
                image_path=img_filename_4,
                embedding=embedding_list,
                pose_label=request.pose_label,
            )
            db.add(new_record)
            db.commit()
            face_embedding_index.add(str(new_person.id), new_person.name, embedding)
            upper_face_embedding_index.add(
                str(new_person.id),
                new_person.name,
                upper_embedding,
            )
            low_light_embedding_index.add(
                str(new_person.id),
                new_person.name,
                low_light_embedding,
            )
            
            # Sync cache in memory immediately
            set_avatar_in_cache(
                str(new_person.id),
                img_filename_4,
                str(new_record.id),
            )
            
            db.refresh(new_person)
            return EnrollResponse(
                status="SUCCESS",
                person_id=str(new_person.id),
                name=new_person.name,
                message="Đăng ký dữ liệu gương mặt thành công."
            )
    except HTTPException:
        get_image_storage().delete_related(img_filename_4)
        raise
    except (IntegrityError, Exception) as exc:
        db.rollback()
        get_image_storage().delete_related(img_filename_4)
        raise HTTPException(status_code=500, detail=f"Lỗi lưu trữ dữ liệu: {str(exc)}")

@router.post("/identify", response_model=IdentifyResponse)
def identify(request: EnrollRequest, db: Session = Depends(get_db)):
    # Note: We reuse EnrollRequest because it only needs image_base64 (name is ignored here)
    try:
        image = decode_image(None, request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể giải mã hình ảnh Base64.")

    # 1. Detect all faces in the frame
    try:
        detections, _ = detect_faces_with_fallback(image)
    except VisionError as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {str(exc)}")

    face_tensors = [extract_face_crop_tensor(image, item.box)[0] for item in detections]
    low_light_indices = [
        index
        for index, detection in enumerate(detections)
        if is_low_light_face_region(image, detection.box)
    ]
    proactive_upper_tensors = [
        extract_upper_face_tensor(image, detections[index].box)
        for index in low_light_indices
    ]
    low_light_tensors = [
        extract_low_light_face_tensor(image, detections[index].box)
        for index in low_light_indices
    ]
    batched_embeddings = get_embeddings(
        face_tensors + proactive_upper_tensors + low_light_tensors
    )
    face_count = len(face_tensors)
    low_light_count = len(low_light_indices)
    embeddings = batched_embeddings[:face_count]
    proactive_upper_embeddings = batched_embeddings[
        face_count:face_count + low_light_count
    ]
    low_light_embeddings = batched_embeddings[face_count + low_light_count:]
    matches_by_face = _search_db_batch(embeddings, db, threshold=0.0)
    full_threshold = float(os.getenv("FACE_MATCH_MIN_CONFIDENCE", "0.76")) * 100
    full_margin = float(os.getenv("FACE_MATCH_MIN_MARGIN", "0.08")) * 100
    rescue_indices = [
        index
        for index, matches in enumerate(matches_by_face)
        if (
            not matches
            or matches[0][1] < full_threshold
            or matches[0][1] - (matches[1][1] if len(matches) > 1 else 0.0) < full_margin
        )
    ]
    upper_matches_by_face = [[] for _ in detections]
    low_light_matches_by_face = [[] for _ in detections]
    rescue_index_set = set(rescue_indices)
    low_light_index_set = set(low_light_indices)
    for index, upper_matches, low_light_matches in zip(
        low_light_indices,
        _search_upper_face_batch(proactive_upper_embeddings),
        _search_low_light_batch(low_light_embeddings),
    ):
        if index in rescue_index_set:
            upper_matches_by_face[index] = upper_matches
            low_light_matches_by_face[index] = low_light_matches

    remaining_upper_indices = [
        index for index in rescue_indices if index not in low_light_index_set
    ]
    if remaining_upper_indices:
        upper_face_tensors = [
            extract_upper_face_tensor(image, detections[index].box)
            for index in remaining_upper_indices
        ]
        upper_embeddings = get_embeddings(upper_face_tensors)
        for index, upper_matches in zip(
            remaining_upper_indices,
            _search_upper_face_batch(upper_embeddings),
        ):
            upper_matches_by_face[index] = upper_matches

    matches_list = []
    partial_threshold = float(os.getenv("FACE_PARTIAL_MIN_CONFIDENCE", "0.77")) * 100
    partial_margin = float(os.getenv("FACE_PARTIAL_MIN_MARGIN", "0.12")) * 100
    partial_full_support = float(os.getenv("FACE_PARTIAL_MIN_FULL_SUPPORT", "0.45")) * 100
    low_light_full_support = float(
        os.getenv("FACE_LOW_LIGHT_MIN_FULL_SUPPORT", "0.42")
    ) * 100
    for detection, db_matches, upper_matches, low_light_matches in zip(
        detections,
        matches_by_face,
        upper_matches_by_face,
        low_light_matches_by_face,
    ):
        box = [int(val) for val in detection.box]
        full_candidates = db_matches
        full_top_score = full_candidates[0][1] if full_candidates else 0.0
        full_second_score = full_candidates[1][1] if len(full_candidates) > 1 else 0.0
        db_matches = (
            [full_candidates[0]]
            if full_top_score >= full_threshold and full_top_score - full_second_score >= full_margin
            else []
        )
        match_mode = "full"
        if not db_matches and low_light_matches:
            top_score = low_light_matches[0][1]
            second_score = low_light_matches[1][1] if len(low_light_matches) > 1 else 0.0
            full_supports_low_light = bool(
                full_candidates
                and full_candidates[0][0].person_id == low_light_matches[0][0].person_id
                and full_candidates[0][1] >= low_light_full_support
            )
            if (
                full_supports_low_light
                and top_score >= full_threshold
                and top_score - second_score >= full_margin
            ):
                db_matches = [low_light_matches[0]]
                match_mode = "low-light"
        if not db_matches and upper_matches:
            top_score = upper_matches[0][1]
            second_score = upper_matches[1][1] if len(upper_matches) > 1 else 0.0
            full_supports_upper = bool(
                full_candidates
                and full_candidates[0][0].person_id == upper_matches[0][0].person_id
                and full_candidates[0][1] >= partial_full_support
            )
            if (
                full_supports_upper
                and top_score >= partial_threshold
                and top_score - second_score >= partial_margin
            ):
                db_matches = [upper_matches[0]]
                match_mode = "upper-face"
        if db_matches:
            best_person, best_sim = db_matches[0]
            matches_list.append(IdentifyFaceMatch(
                box=box,
                identified=True,
                person_id=best_person.person_id,
                name=best_person.name,
                score=best_sim,
                avatar_base64=None,
                avatar_url=f"/api/v1/faces/profiles/{best_person.person_id}/avatar",
                match_mode=match_mode,
            ))
        else:
            matches_list.append(IdentifyFaceMatch(
                box=box,
                identified=False,
                person_id=None,
                name="Unknown",
                score=0.0,
                avatar_base64=None,
                avatar_url=None,
                match_mode=None,
            ))

    return IdentifyResponse(
        faces_detected=len(detections),
        matches=matches_list
    )
