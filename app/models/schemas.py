from enum import Enum

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
