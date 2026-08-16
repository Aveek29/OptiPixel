import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.models.schemas import EnhancementProfile, ImageAnalysis
from app.services import model_service

logger = logging.getLogger(__name__)


def run_enhancement(input_path: str, output_path: str, analysis: ImageAnalysis) -> tuple[int, int, int]:
    profile = analysis.profile
    scale = _select_scale(profile, analysis)
    logger.info("Running profile=%s, scale=%d", profile.value, scale)

    if profile == EnhancementProfile.LIGHT:
        w, h = _light_enhance(input_path, output_path)
        return w, h, 1

    if profile == EnhancementProfile.RESTORE:
        w, h = _restore_enhance(input_path, output_path, scale)
        return w, h, scale

    w, h = _super_res_enhance(input_path, output_path, scale)
    return w, h, scale


def _select_scale(profile: EnhancementProfile, analysis: ImageAnalysis) -> int:
    if profile == EnhancementProfile.LIGHT:
        return 1
    if analysis.pixel_count > 5_000_000:
        return 2
    return min(settings.default_scale, settings.max_scale)


def _light_enhance(input_path: str, output_path: str) -> tuple[int, int]:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)

    cv2.imwrite(output_path, sharpened)
    h, w = sharpened.shape[:2]
    return w, h


def _restore_enhance(input_path: str, output_path: str, scale: int) -> tuple[int, int]:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    temp_path = input_path + ".denoised.png"
    cv2.imwrite(temp_path, denoised)

    try:
        w, h = model_service.enhance_image(temp_path, output_path, scale, settings.model_weights_dir)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return w, h


def _super_res_enhance(input_path: str, output_path: str, scale: int) -> tuple[int, int]:
    return model_service.enhance_image(input_path, output_path, scale, settings.model_weights_dir)
