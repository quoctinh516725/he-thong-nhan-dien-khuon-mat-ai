from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IndexedIdentity:
    person_id: str
    name: str


class FaceEmbeddingIndex:
    """Thread-safe in-memory cosine index backed by immutable snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._identities: tuple[IndexedIdentity, ...] = ()
        self._matrix = np.empty((0, 512), dtype=np.float32)
        self._version = 0
        self._source_signature = None

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def source_signature(self):
        with self._lock:
            return self._source_signature

    def replace(
        self,
        entries: Iterable[tuple[str, str, object]],
        expected_version: int | None = None,
        source_signature=None,
    ) -> bool:
        identities = []
        vectors = []
        for person_id, name, embedding in entries:
            vector = self._normalize(embedding)
            if vector.shape[0] != 512:
                continue
            identities.append(IndexedIdentity(person_id=person_id, name=name))
            vectors.append(vector)

        matrix = (
            np.stack(vectors).astype(np.float32, copy=False)
            if vectors
            else np.empty((0, 512), dtype=np.float32)
        )
        with self._lock:
            if expected_version is not None and self._version != expected_version:
                return False
            self._identities = tuple(identities)
            self._matrix = matrix
            self._version += 1
            self._source_signature = source_signature
        return True

    def add(self, person_id: str, name: str, embedding: object) -> None:
        vector = self._normalize(embedding)
        if vector.shape[0] != 512:
            raise ValueError("Face embedding must contain 512 values.")
        with self._lock:
            self._identities = self._identities + (
                IndexedIdentity(person_id=person_id, name=name),
            )
            self._matrix = np.concatenate((self._matrix, vector[None, :]), axis=0)
            self._version += 1

    def search(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> list[list[tuple[IndexedIdentity, float]]]:
        query_matrix = np.asarray(embeddings, dtype=np.float32)
        if query_matrix.ndim != 2 or query_matrix.shape[1] != 512:
            raise ValueError("Embeddings must have shape (N, 512).")
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix = query_matrix / np.maximum(norms, 1e-12)

        with self._lock:
            identities = self._identities
            matrix = self._matrix
        if matrix.shape[0] == 0:
            return [[] for _ in range(len(query_matrix))]

        similarity_matrix = query_matrix @ matrix.T
        results = []
        for similarities in similarity_matrix:
            best_by_person = {}
            for identity, cosine_similarity in zip(identities, similarities):
                similarity = float(cosine_similarity)
                if similarity < threshold:
                    continue
                current = best_by_person.get(identity.person_id)
                if current is None or similarity > current[1]:
                    best_by_person[identity.person_id] = (identity, similarity)
            results.append(
                sorted(
                    best_by_person.values(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        return results

    def search_consensus(
        self,
        embeddings: np.ndarray,
        threshold: float,
        support_count: int = 2,
    ) -> list[list[tuple[IndexedIdentity, float]]]:
        """Rank identities by the mean of their strongest supporting samples."""

        if support_count < 1:
            raise ValueError("Consensus support count must be positive.")
        query_matrix = np.asarray(embeddings, dtype=np.float32)
        if query_matrix.ndim != 2 or query_matrix.shape[1] != 512:
            raise ValueError("Embeddings must have shape (N, 512).")
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix = query_matrix / np.maximum(norms, 1e-12)

        with self._lock:
            identities = self._identities
            matrix = self._matrix
        if matrix.shape[0] == 0:
            return [[] for _ in range(len(query_matrix))]

        similarity_matrix = query_matrix @ matrix.T
        results = []
        for similarities in similarity_matrix:
            scores_by_person = {}
            for identity, cosine_similarity in zip(identities, similarities):
                current = scores_by_person.setdefault(
                    identity.person_id,
                    (identity, []),
                )
                current[1].append(float(cosine_similarity))

            consensus_matches = []
            for identity, scores in scores_by_person.values():
                if len(scores) < support_count:
                    continue
                strongest = sorted(scores, reverse=True)[:support_count]
                consensus_similarity = float(np.mean(strongest))
                if consensus_similarity >= threshold:
                    consensus_matches.append((identity, consensus_similarity))
            results.append(
                sorted(
                    consensus_matches,
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        return results

    @staticmethod
    def _normalize(embedding: object) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        return vector / max(float(norm), 1e-12)


face_embedding_index = FaceEmbeddingIndex()
upper_face_embedding_index = FaceEmbeddingIndex()
low_light_embedding_index = FaceEmbeddingIndex()


def load_face_embedding_index(db) -> int:
    from app.models.entities import Person, FaceRecord
    from sqlalchemy import func

    index_version = face_embedding_index.version
    upper_index_version = upper_face_embedding_index.version
    low_light_index_version = low_light_embedding_index.version
    source_signature = tuple(
        db.query(
            func.count(FaceRecord.id),
            func.max(FaceRecord.created_at),
        ).one()
    )
    if (
        source_signature == face_embedding_index.source_signature
        and source_signature == upper_face_embedding_index.source_signature
        and source_signature == low_light_embedding_index.source_signature
    ):
        return 0

    records = db.query(FaceRecord).join(FaceRecord.person).all()
    entries = (
        (
            str(record.person.id),
            record.person.name,
            record.embedding,
        )
        for record in records
    )
    face_embedding_index.replace(
        entries,
        expected_version=index_version,
        source_signature=source_signature,
    )

    import cv2

    from app.services.ai import (
        extract_illumination_reference_tensor,
        face_crop_to_tensor,
        get_embeddings,
    )
    from app.services.object_storage import get_image_storage
    from app.vision.preprocessing import mask_lower_face

    upper_tensors = []
    low_light_tensors = []
    upper_identities = []
    crop_references = []
    for record in records:
        crop_path = record.image_path
        if crop_path and "4_segmented_crop_" in crop_path:
            crop_path = crop_path.replace("4_segmented_crop_", "2_yolo_crop_")
        crop_references.append(crop_path)

    storage = get_image_storage()
    with ThreadPoolExecutor(max_workers=8) as executor:
        crops = list(executor.map(
            lambda reference: storage.read_image(reference) if reference else None,
            crop_references,
        ))
    for record, crop in zip(records, crops):
        if crop is None:
            continue
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
        upper_tensors.append(face_crop_to_tensor(mask_lower_face(crop)))
        low_light_tensors.append(extract_illumination_reference_tensor(crop))
        upper_identities.append((str(record.person.id), record.person.name))
    auxiliary_embeddings = get_embeddings(upper_tensors + low_light_tensors)
    split_index = len(upper_tensors)
    upper_embeddings = auxiliary_embeddings[:split_index]
    low_light_embeddings = auxiliary_embeddings[split_index:]
    upper_face_embedding_index.replace(
        (
            (person_id, name, embedding)
            for (person_id, name), embedding in zip(
                upper_identities,
                upper_embeddings,
            )
        ),
        expected_version=upper_index_version,
        source_signature=source_signature,
    )
    low_light_embedding_index.replace(
        (
            (person_id, name, embedding)
            for (person_id, name), embedding in zip(
                upper_identities,
                low_light_embeddings,
            )
        ),
        expected_version=low_light_index_version,
        source_signature=source_signature,
    )
    return len(records)
