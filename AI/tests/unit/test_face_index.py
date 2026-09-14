import numpy as np

from app.services.face_index import FaceEmbeddingIndex


def test_face_index_batches_queries_and_keeps_best_record_per_person():
    index = FaceEmbeddingIndex()
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    similar = first.copy()
    similar[1] = 0.1
    second = np.zeros(512, dtype=np.float32)
    second[2] = 1.0
    index.replace(
        [
            ("person-1", "First", first),
            ("person-1", "First", similar),
            ("person-2", "Second", second),
        ]
    )

    matches = index.search(np.stack((first, second)), threshold=0.65)

    assert [item[0].person_id for item in matches[0]] == ["person-1"]
    assert [item[0].person_id for item in matches[1]] == ["person-2"]
    assert matches[0][0][1] == 1.0


def test_face_index_adds_new_embedding_without_reloading_database():
    index = FaceEmbeddingIndex()
    embedding = np.ones(512, dtype=np.float32)

    index.add("person-1", "First", embedding)

    matches = index.search(embedding[None, :], threshold=0.99)
    assert matches[0][0][0].name == "First"


def test_stale_refresh_does_not_overwrite_new_enrollment():
    index = FaceEmbeddingIndex()
    embedding = np.ones(512, dtype=np.float32)
    refresh_version = index.version
    index.add("person-1", "First", embedding)

    replaced = index.replace([], expected_version=refresh_version)

    assert replaced is False
    assert index.search(embedding[None, :], threshold=0.99)[0][0][0].name == "First"


def test_consensus_search_requires_two_supporting_samples_per_person():
    index = FaceEmbeddingIndex()
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0

    def vector_with_similarity(similarity):
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = similarity
        vector[1] = np.sqrt(1.0 - similarity ** 2)
        return vector

    index.replace([
        ("person-1", "Supported", vector_with_similarity(1.0)),
        ("person-1", "Supported", vector_with_similarity(0.8)),
        ("person-2", "Single outlier", vector_with_similarity(0.99)),
    ])

    matches = index.search_consensus(
        query[None, :],
        threshold=0.75,
        support_count=2,
    )

    assert [item[0].person_id for item in matches[0]] == ["person-1"]
    assert np.isclose(matches[0][0][1], 0.9)
