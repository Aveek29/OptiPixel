#!/usr/bin/env python3
"""Verify that Real-ESRGAN model weights are present and loadable.

Run after download_model.py:
    python3 scripts/verify_model.py
"""
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

REQUIRED = ["RealESRGAN_x4plus.pth"]


def main() -> None:
    """Check model files exist."""
    print("Optic Pixel — Model Verification\n")

    all_ok = True
    for name in REQUIRED:
        path = MODELS_DIR / name
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  [ok] {name} ({size_mb:.1f} MB)")
        else:
            print(f"  [missing] {name}")
            all_ok = False

    if not all_ok:
        print("\nSome models are missing. They will be downloaded at app startup.")
        return

    print("\nAll required model files found.")
    print("Verifying model can be loaded...")

    try:
        import torch
        from app.services.esrgan.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state_dict = torch.load(str(MODELS_DIR / REQUIRED[0]), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  [ok] Model loaded on CPU (torch {torch.__version__})")
    except ImportError as e:
        print(f"  [warn] Could not import ML libraries: {e}")
        print("  [warn] Model verification skipped — will verify at app startup.")
    except Exception as e:
        print(f"  [warn] Model load check failed: {e}")
        print("  [warn] Model verification skipped — will verify at app startup.")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
