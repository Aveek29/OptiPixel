# Optic Pixel — Dynamic AI Image Enhancement

Upload an image; the app analyzes it (sharpness, brightness, contrast) and runs the best enhancement profile — CLAHE/sharpening for light fixes, Real-ESRGAN super-resolution for upscaling.

Runs as a **Streamlit** app on CPU. No cloud storage, no AWS, no GPU needed. Real-ESRGAN's `basicsr`/`realesrgan` packages are vendored into `app/services/esrgan/` so no C++ compilation or heavy ML dependencies are needed at install time.

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python scripts/download_model.py  # fetches ~145MB of weights into models/
streamlit run streamlit_app.py
```

Open the URL Streamlit prints. Model weights download automatically at first enhance if missing.

## Deploy to Streamlit Community Cloud (free)

1. **Push this repo to a public GitHub repo** (default branch `main`).
   - `models/` is gitignored — weights download at first use on the cloud, no need to commit 145MB.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**.
3. Pick the repo → branch `main` → main file path **`streamlit_app.py`**.
4. Click **Advanced settings** and set **Python version: 3.11**.
   - `requirements.txt` pins CPU-only PyTorch wheels built for **cp311** (manylinux x86_64). Streamlit Cloud defaults to 3.12, which will fail to find those wheels — 3.11 is required.
5. Click **Deploy**. First build ~5-8 min (pip installs torch CPU). Then open the `*.streamlit.app` URL.

Notes:
- **Free tier:** 2 cores / up to 2.7GB RAM — enough for Real-ESRGAN with 256px tiles (~1.5GB).
- **Sleep:** free apps sleep after ~12h idle; the first visitor gets a "waking up" page. Model weights re-download on cold start, so the first enhancement after wake is slow (~1-2 min), then instant.
- **Disk:** no persistent storage; uploads and outputs are temporary by design.
- **Memory:** if you hit resource limits, upload smaller images (max 10MB) — the app selects profiles accordingly.

## Deploy to other platforms

- **Hugging Face Spaces (Docker):** requires a paid PRO plan to create a Docker Space ($9/mo). Not needed — use Streamlit Cloud above.
- **Render / Fly.io / Railway / any Docker VPS:** the repo no longer ships a Dockerfile; use the Streamlit-native path (`streamlit run streamlit_app.py`) or port `streamlit_app.py` into a container manually.
- **Vercel:** not suitable — no long-running Python processes.

## How it works

1. **Analyze** — OpenCV reads width/height, sharpness, brightness, contrast.
2. **Choose profile** — LIGHT (CLAHE + sharpening), SUPER_RES (Real-ESRGAN 2x), or RESTORE (denoise + super-res).
3. **Enhance** — tiled Real-ESRGAN inference on CPU (256px tiles, ~2GB memory).
4. **Download** — before/after shown side by side with a download button.

## Project layout

```
streamlit_app.py            # Streamlit UI (deploy entrypoint)
app/
  config.py                 # env-based settings
  services/
    esrgan/                 # vendored RRDBNet + RealESRGANer (Apache-2.0)
    model_service.py        # model loading / inference
    model_downloader.py     # weights download (skip if present)
    enhancement_service.py  # profile selection + enhancement
    image_analyzer.py       # OpenCV analysis
    storage_service.py      # local disk output
scripts/                    # download_model.py, verify_model.py
models/                     # weights (downloaded, gitignored)
```

## License

Real-ESRGAN model weights and vendored code are under [Apache-2.0](https://github.com/xinntao/Real-ESRGAN).
