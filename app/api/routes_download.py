import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/download-local/{job_id}")
def download_local(job_id: str):
    path = storage_service.output_path(job_id)
    if not path:
        raise HTTPException(404, "Image not found or expired")
    return FileResponse(path, media_type="image/png", filename="optic-pixel-enhanced.png")
