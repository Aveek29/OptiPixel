#!/usr/bin/env python3
"""Download Real-ESRGAN pretrained model weights.

Run this script before starting the application:
    python3 scripts/download_model.py

The model files will be placed in the models/ directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.model_downloader import MODELS_DIR, WEIGHTS, ensure_model_weights


def main() -> None:
    """Download all required model weights."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Optic Pixel — Model Downloader")
    print(f"Target directory: {MODELS_DIR}\n")

    ensure_model_weights()

    print("\nModel download complete.")


if __name__ == "__main__":
    main()
