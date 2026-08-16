"""Download Real-ESRGAN pretrained model weights if missing.

Runs at app startup so the application works without bundling weights.
"""

import logging
import urllib.request
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

MODELS_DIR = Path(settings.model_weights_dir)

WEIGHTS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


def ensure_model_weights() -> None:
    """Download any missing weights into the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in WEIGHTS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 1024:
            logger.info("Model %s already present (%.1f MB)", name, dest.stat().st_size / 1024 / 1024)
            continue

        logger.info("Downloading model %s ...", name)
        try:
            urllib.request.urlretrieve(url, str(dest))
            logger.info("Downloaded %s (%.1f MB)", name, dest.stat().st_size / 1024 / 1024)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
