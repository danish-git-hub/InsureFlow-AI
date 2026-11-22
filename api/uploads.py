"""Safe reading of uploaded files (size limit enforced while reading)."""

from fastapi import UploadFile

from app.config import settings
from services.claim_service import ServiceError

CHUNK = 1024 * 1024


async def read_upload(file: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await file.read(CHUNK):
        data.extend(chunk)
        if len(data) > settings.max_upload_bytes:
            raise ServiceError(f"'{file.filename}' exceeds the {settings.max_upload_mb} MB upload limit.", 413)
    return bytes(data)
