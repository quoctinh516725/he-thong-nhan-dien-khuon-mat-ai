import cv2
import numpy as np

from app.services.object_storage import ImageStorage, StorageConfig


class FakeResponse:
    def __init__(self, data: bytes):
        self.data = data

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinio:
    def __init__(self):
        self.buckets = set()
        self.objects = {}

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)

    def put_object(self, bucket, key, stream, length, content_type):
        assert content_type == "image/jpeg"
        self.objects[(bucket, key)] = stream.read(length)

    def get_object(self, bucket, key):
        return FakeResponse(self.objects[(bucket, key)])

    def remove_object(self, bucket, key):
        self.objects.pop((bucket, key), None)


def make_image(value: int) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_minio_storage_uploads_jpegs_in_batch_and_reads_them_back():
    client = FakeMinio()
    storage = ImageStorage(
        StorageConfig(backend="minio", bucket="faces", upload_workers=4),
        minio_client=client,
    )

    references = storage.put_jpeg_batch(
        {
            "session/1_decoded_input_user.jpg": make_image(40),
            "session/4_segmented_crop_user.jpg": make_image(180),
        }
    )

    assert client.buckets == {"faces"}
    assert references["session/4_segmented_crop_user.jpg"] == (
        "minio://faces/session/4_segmented_crop_user.jpg"
    )
    decoded = cv2.imdecode(
        np.frombuffer(
            storage.read_bytes(references["session/4_segmented_crop_user.jpg"]),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (32, 32, 3)
    assert decoded.mean() > 170


def test_delete_related_removes_all_enrollment_artifacts():
    client = FakeMinio()
    storage = ImageStorage(
        StorageConfig(backend="minio", bucket="faces", upload_workers=2),
        minio_client=client,
    )
    references = storage.put_jpeg_batch(
        {
            f"session/{prefix}_user.jpg": make_image(100)
            for prefix in (
                "1_decoded_input",
                "2_yolo_crop",
                "3_unet_mask",
                "4_segmented_crop",
            )
        }
    )

    storage.delete_related(references["session/4_segmented_crop_user.jpg"])

    assert client.objects == {}
