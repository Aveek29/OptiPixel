import logging
from pathlib import Path

from app.models.schemas import EnhancementProfile, ImageAnalysis

logger = logging.getLogger(__name__)

PROFILE_DESCRIPTIONS = {
    EnhancementProfile.LIGHT: "Mild contrast and sharpness correction",
    EnhancementProfile.SUPER_RES: "AI super-resolution upscaling",
    EnhancementProfile.RESTORE: "Noise reduction + AI super-resolution",
}


def analyze_image(path: str) -> ImageAnalysis:
    import cv2
    import numpy as np
    from PIL import Image

    pil_image = Image.open(path)
    pil_image.load()

    width, height = pil_image.size
    channels = len(pil_image.getbands())
    pixel_count = width * height
    fmt = pil_image.format or "UNKNOWN"
    mode = pil_image.mode

    cv_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if cv_image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    file_size = Path(path).stat().st_size
    sharpness = float(cv2.Laplacian(cv_image, cv2.CV_64F).var())
    mean_brightness = float(np.mean(cv_image))
    contrast_score = float(np.std(cv_image))
    profile = _choose_profile(pixel_count, sharpness, mean_brightness)

    analysis = ImageAnalysis(
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        fmt=fmt,
        file_size_bytes=file_size,
        pixel_count=pixel_count,
        sharpness_score=round(sharpness, 2),
        mean_brightness=round(mean_brightness, 2),
        contrast_score=round(contrast_score, 2),
        profile=profile,
        profile_description=PROFILE_DESCRIPTIONS[profile],
    )

    logger.info(
        "Analyzed: %dx%d, pixels=%d, sharpness=%.2f, brightness=%.2f, profile=%s",
        width, height, pixel_count, sharpness, mean_brightness, profile.value,
    )
    return analysis


def _choose_profile(pixel_count: int, sharpness: float, brightness: float) -> EnhancementProfile:
    if pixel_count < 500_000:
        return EnhancementProfile.SUPER_RES
    if sharpness < 80.0:
        return EnhancementProfile.RESTORE
    if brightness < 60.0 or brightness > 200.0:
        return EnhancementProfile.RESTORE
    return EnhancementProfile.LIGHT
