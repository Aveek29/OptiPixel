import logging
from pathlib import Path

import cv2
import torch

from app.config import settings
from app.services.esrgan.realesrganer import RealESRGANer
from app.services.esrgan.rrdbnet_arch import RRDBNet
from app.services.model_downloader import ensure_model_weights

logger = logging.getLogger(__name__)

_MODEL_INSTANCE = None


def get_model(weights_dir: str) -> RealESRGANer:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    model_path = Path(weights_dir) / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        ensure_model_weights()
    if not model_path.exists():
        raise RuntimeError(f"Model weights not found at {model_path} and download failed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Real-ESRGAN on %s", device)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    _MODEL_INSTANCE = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=settings.model_tile_size,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )

    logger.info("Real-ESRGAN loaded")
    return _MODEL_INSTANCE


def enhance_image(input_path: str, output_path: str, scale: int = 2, weights_dir: str = "models") -> tuple[int, int]:
    model = get_model(weights_dir)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    output, _ = model.enhance(img, outscale=scale)
    cv2.imwrite(output_path, output)

    h, w = output.shape[:2]
    logger.info("Enhanced: %dx%d -> %dx%d (scale=%d)", img.shape[1], img.shape[0], w, h, scale)
    return w, h
