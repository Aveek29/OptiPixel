import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import EnhanceRequest
from app.services import enhancement_service, storage_service
from app.services.image_analyzer import analyze_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/enhance", response_model=EnhanceRequest)
async def enhance_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:12]
    start_time = time.time()
    logger.info("Job %s: upload '%s' (type=%s)", job_id, file.filename, file.content_type)

    _validate_upload(file)

    raw_bytes = await file.read()
    if len(raw_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit")

    ext = _get_extension(file.filename or "upload.png")

    with tempfile.TemporaryDirectory(prefix="optic_pixel_") as workdir:
        input_path = Path(workdir) / f"input{ext}"
        output_path = Path(workdir) / "output.png"
        input_path.write_bytes(raw_bytes)

        try:
            analysis = analyze_image(str(input_path))
        except Exception as e:
            logger.error("Job %s: analysis failed: %s", job_id, e)
            raise HTTPException(422, "Image could not be decoded or analyzed")

        if analysis.pixel_count > settings.max_image_pixels:
            raise HTTPException(413, f"Image too large ({analysis.pixel_count:,} pixels). Max: {settings.max_image_pixels:,}.")

        try:
            future = _executor.submit(enhancement_service.run_enhancement, str(input_path), str(output_path), analysis)
            out_w, out_h, scale_used = future.result(timeout=settings.max_processing_seconds)
        except FuturesTimeout:
            raise HTTPException(504, "Processing took too long. Try a smaller image.")
        except Exception as e:
            logger.error("Job %s: enhancement failed: %s", job_id, e)
            raise HTTPException(500, "Enhancement failed.")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(500, "Enhancement produced no output")

        output_size = output_path.stat().st_size
        storage_service.save_output(job_id, output_path.read_bytes())
        download_url = f"/api/download-local/{job_id}"

        processing_ms = (time.time() - start_time) * 1000
        logger.info("Job %s: done in %.0fms, %dx%d -> %dx%d", job_id, processing_ms, analysis.width, analysis.height, out_w, out_h)

    return EnhanceRequest(
        job_id=job_id,
        status="completed",
        message="Enhancement complete",
        analysis=analysis,
        download_url=download_url,
        processing_ms=round(processing_ms, 1),
        output_width=out_w,
        output_height=out_h,
        output_size_bytes=output_size,
        scale=scale_used,
    )


def _validate_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(415, f"Unsupported type: {file.content_type}. Use JPG, PNG, or WebP.")
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported extension: {ext}")


def _get_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".jpeg":
        return ".jpg"
    if ext in ALLOWED_EXTENSIONS:
        return ext
    return ".png"
