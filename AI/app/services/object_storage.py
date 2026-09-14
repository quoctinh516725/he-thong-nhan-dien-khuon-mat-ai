from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from threading import Lock
from typing import Mapping, Optional
from urllib.parse import urlparse

import cv2
from dotenv import load_dotenv
import numpy as np
import requests
import urllib3


AI_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(AI_ROOT / ".env")
ARTIFACT_PREFIXES = (
    "1_decoded_input_",
    "2_yolo_crop_",
    "3_unet_mask_",
    "4_segmented_crop_",
)


def _env_enabled(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "local"
    bucket: str = "face-detection"
    upload_workers: int = 4
    # MinIO
    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    secure: bool = True
    ca_cert: Optional[str] = None
    # Cloudinary
    cloudinary_cloud_name: Optional[str] = None
    cloudinary_api_key: Optional[str] = None
    cloudinary_api_secret: Optional[str] = None
    cloudinary_folder: str = "face-detection"
    # Local
    local_root: Path = AI_ROOT / "storage" / "processed_enroll"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        ca_cert = os.getenv("MINIO_CA_CERT")
        if ca_cert and not Path(ca_cert).is_absolute():
            ca_cert = str((AI_ROOT / ca_cert).resolve())
        return cls(
            backend=os.getenv("STORAGE_BACKEND", "local").lower(),
            bucket=os.getenv("MINIO_BUCKET", "face-detection"),
            upload_workers=max(1, int(os.getenv("STORAGE_UPLOAD_WORKERS", "4"))),
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=_env_enabled("MINIO_SECURE", True),
            ca_cert=ca_cert,
            cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            cloudinary_api_key=os.getenv("CLOUDINARY_API_KEY"),
            cloudinary_api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            cloudinary_folder=os.getenv("CLOUDINARY_FOLDER", "face-detection"),
        )


class ImageStorage:
    """Stores enrollment artifacts with parallel I/O and object storage support."""

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        minio_client=None,
    ) -> None:
        self.config = config or StorageConfig.from_env()
        self._ready_lock = Lock()
        self._ready = False
        self._client = minio_client
        if self.config.backend == "minio" and self._client is None:
            self._client = self._build_minio_client()
        elif self.config.backend == "cloudinary":
            self._init_cloudinary()

    def _init_cloudinary(self) -> None:
        try:
            import cloudinary
        except ImportError as exc:
            raise RuntimeError("Cloudinary package is not installed. Run `pip install cloudinary`") from exc

        required = {
            "CLOUDINARY_CLOUD_NAME": self.config.cloudinary_cloud_name,
            "CLOUDINARY_API_KEY": self.config.cloudinary_api_key,
            "CLOUDINARY_API_SECRET": self.config.cloudinary_api_secret,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing Cloudinary configuration: {', '.join(missing)}")

        cloudinary.config(
            cloud_name=self.config.cloudinary_cloud_name,
            api_key=self.config.cloudinary_api_key,
            api_secret=self.config.cloudinary_api_secret,
            secure=True,
        )

    def _build_minio_client(self):
        from minio import Minio
        required = {
            "MINIO_ENDPOINT": self.config.endpoint,
            "MINIO_ACCESS_KEY": self.config.access_key,
            "MINIO_SECRET_KEY": self.config.secret_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing MinIO configuration: {', '.join(missing)}")

        pool_kwargs = {
            "timeout": urllib3.Timeout(connect=3.0, read=15.0),
            "maxsize": self.config.upload_workers + 4,
            "retries": urllib3.Retry(total=2, backoff_factor=0.2),
        }
        if self.config.secure:
            pool_kwargs["cert_reqs"] = "CERT_REQUIRED"
            if self.config.ca_cert:
                pool_kwargs["ca_certs"] = self.config.ca_cert
        http_client = urllib3.PoolManager(**pool_kwargs)
        return Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
            http_client=http_client,
        )

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            if self.config.backend == "minio":
                if not self._client.bucket_exists(self.config.bucket):
                    self._client.make_bucket(self.config.bucket)
            elif self.config.backend == "cloudinary":
                self._init_cloudinary()
                try:
                    import cloudinary.api
                    cloudinary.api.ping()
                except Exception as exc:
                    raise RuntimeError(f"Failed to connect to Cloudinary: {exc}") from exc
            else:
                self.config.local_root.mkdir(parents=True, exist_ok=True)
            self._ready = True

    def put_jpeg_batch(self, images: Mapping[str, np.ndarray]) -> dict[str, str]:
        self.ensure_ready()
        encoded = {}
        for key, image in images.items():
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                raise RuntimeError(f"Could not encode enrollment artifact: {key}")
            encoded[key] = buffer.tobytes()

        references = {}
        uploaded_keys = []
        try:
            with ThreadPoolExecutor(max_workers=self.config.upload_workers) as executor:
                futures = {
                    executor.submit(self._put_bytes, key, data): key
                    for key, data in encoded.items()
                }
                for future in as_completed(futures):
                    key = futures[future]
                    references[key] = future.result()
                    uploaded_keys.append(key)
        except Exception:
            with ThreadPoolExecutor(max_workers=self.config.upload_workers) as executor:
                list(executor.map(self._delete_key, uploaded_keys))
            raise
        return references

    def _put_bytes(self, key: str, data: bytes) -> str:
        if self.config.backend == "cloudinary":
            import cloudinary.uploader
            clean_key = key.lstrip("/")
            if clean_key.endswith(".jpg"):
                clean_key = clean_key[:-4]
            public_id = f"{self.config.cloudinary_folder}/{clean_key}"
            result = cloudinary.uploader.upload(
                BytesIO(data),
                public_id=public_id,
                resource_type="image",
                format="jpg",
                overwrite=True,
                invalidate=True,
            )
            return result.get("secure_url") or result.get("url")

        if self.config.backend == "minio":
            self._client.put_object(
                self.config.bucket,
                key,
                BytesIO(data),
                length=len(data),
                content_type="image/jpeg",
            )
            return f"minio://{self.config.bucket}/{key}"

        path = (self.config.local_root / key).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def read_bytes(self, reference: str) -> bytes:
        if reference.startswith("http://") or reference.startswith("https://"):
            response = requests.get(reference, timeout=15)
            response.raise_for_status()
            return response.content

        if reference.startswith("minio://"):
            bucket, key = self._parse_minio_reference(reference)
            response = self._client.get_object(bucket, key)
            try:
                return response.data
            finally:
                response.close()
                response.release_conn()

        path = Path(reference)
        if not path.is_absolute():
            path = (AI_ROOT / path).resolve()
        return path.read_bytes()

    def read_image(self, reference: str) -> Optional[np.ndarray]:
        try:
            data = self.read_bytes(reference)
        except Exception:
            return None
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def delete_related(self, reference: Optional[str]) -> None:
        if not reference:
            return
        related = {reference}
        for source_prefix in ARTIFACT_PREFIXES:
            if source_prefix not in reference:
                continue
            related.update(
                reference.replace(source_prefix, target_prefix)
                for target_prefix in ARTIFACT_PREFIXES
            )
        with ThreadPoolExecutor(max_workers=self.config.upload_workers) as executor:
            list(executor.map(self.delete, related))

    def delete_all(self) -> None:
        self.ensure_ready()
        if self.config.backend == "cloudinary":
            import cloudinary.api
            try:
                cloudinary.api.delete_resources_by_prefix(f"{self.config.cloudinary_folder}/")
            except Exception as exc:
                print(f"WARNING: Cloudinary delete_resources_by_prefix failed: {exc}")
            return

        if self.config.backend == "minio":
            keys = [item.object_name for item in self._client.list_objects(
                self.config.bucket,
                recursive=True,
            )]
            with ThreadPoolExecutor(max_workers=self.config.upload_workers) as executor:
                list(executor.map(self._delete_key, keys))
            return

        for path in self.config.local_root.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    def delete(self, reference: str) -> None:
        if reference.startswith("http://") or reference.startswith("https://"):
            import cloudinary.uploader
            public_id = self._parse_cloudinary_public_id(reference)
            try:
                cloudinary.uploader.destroy(public_id, invalidate=True)
            except Exception as exc:
                print(f"WARNING: Could not delete Cloudinary asset {public_id}: {exc}")
            return

        if reference.startswith("minio://"):
            bucket, key = self._parse_minio_reference(reference)
            self._client.remove_object(bucket, key)
            return

        path = Path(reference)
        if not path.is_absolute():
            path = (AI_ROOT / path).resolve()
        try:
            path.relative_to(self.config.local_root.resolve())
        except ValueError:
            return
        path.unlink(missing_ok=True)

    def _delete_key(self, key: str) -> None:
        if self.config.backend == "cloudinary":
            clean_key = key.lstrip("/")
            if clean_key.endswith(".jpg"):
                clean_key = clean_key[:-4]
            self.delete(f"{self.config.cloudinary_folder}/{clean_key}")
        elif self.config.backend == "minio":
            self._client.remove_object(self.config.bucket, key)
        else:
            self.delete(str(self.config.local_root / key))

    @staticmethod
    def _parse_minio_reference(reference: str) -> tuple[str, str]:
        parsed = urlparse(reference)
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def _parse_cloudinary_public_id(reference: str) -> str:
        if reference.startswith("http://") or reference.startswith("https://"):
            parsed_path = urlparse(reference).path
            parts = parsed_path.split("/image/upload/")
            if len(parts) == 2:
                subpath = parts[1]
                # Strip transformation/version prefix if present, e.g. v1726276262/
                path_segments = subpath.split("/")
                filtered_segments = []
                for idx, seg in enumerate(path_segments):
                    if idx == 0 and seg.startswith("v") and seg[1:].isdigit():
                        continue
                    filtered_segments.append(seg)
                subpath = "/".join(filtered_segments)
                # Strip extension (.jpg, .png, etc.)
                public_id = str(Path(subpath).with_suffix("")).replace("\\", "/")
                return public_id
        return reference


_storage = None
_storage_lock = Lock()


def get_image_storage() -> ImageStorage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = ImageStorage()
    return _storage
