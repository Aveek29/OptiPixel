# Model Weights

This directory stores Real-ESRGAN pretrained model weights.

## Setup

Run the download script before starting the application:

```bash
python scripts/download_model.py
```

This will download the required model files (~65MB each).

## Files

- `RealESRGAN_x4plus.pth` — General-purpose 4x super-resolution
- `RealESRGAN_x2plus.pth` — General-purpose 2x super-resolution  
- `RealESRGAN_x4plus_anime_6B.pth` — Anime/illustration 4x (optional)

## License

Model weights are subject to the Real-ESRGAN license:
https://github.com/xinntao/Real-ESRGAN
