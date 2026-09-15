import os
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadminpassword")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
BUCKET_NAME = os.environ.get("MINIO_BUCKET", "openlist")

_client = None

def get_minio_client():
    global _client
    if _client is None:
        _client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
    return _client

def ensure_bucket(bucket_name=BUCKET_NAME):
    client = get_minio_client()
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    return True

def upload_file(local_path, object_name, content_type="video/mp4", bucket_name=BUCKET_NAME):
    """
    Upload a local file to MinIO
    """
    client = get_minio_client()
    ensure_bucket(bucket_name)
    
    client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=local_path,
        content_type=content_type
    )
    return object_name
