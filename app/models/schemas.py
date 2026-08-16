from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EnhancementProfile(str, Enum):
    LIGHT = "LIGHT"
    SUPER_RES = "SUPER_RES"
    RESTORE = "RESTORE"


class ImageAnalysis(BaseModel):
    width: int
    height: int
    channels: int
    mode: str
    fmt: str
    file_size_bytes: int
    pixel_count: int
    sharpness_score: float
    mean_brightness: float
    contrast_score: float
    profile: EnhancementProfile
    profile_description: str


class EnhanceRequest(BaseModel):
    job_id: str
    status: str
    message: str
    analysis: Optional[ImageAnalysis] = None
    download_url: Optional[str] = None
    processing_ms: Optional[float] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    output_size_bytes: Optional[int] = None
    scale: Optional[int] = None
    engine: str = "Real-ESRGAN (CPU)"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
