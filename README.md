# Optic Pixel — Dynamic AI Image Enhancement

**Optic Pixel** is a lightweight, fully CPU-based AI image enhancement platform. Upload a photo, and the app analyzes it (sharpness, brightness, contrast, resolution), then runs the most suitable enhancement profile — CLAHE contrast correction + sharpening for light fixes, or Real-ESRGAN super-resolution for genuine upscaling with detail reconstruction.

It runs as a **Streamlit** app. No GPU, no cloud storage, no AWS, no accounts.

Real-ESRGAN's `basicsr` / `realesrgan` packages are **vendored** into `app/services/esrgan/`, so there is no C++ compilation or heavy ML dependency at install time — just PyTorch (CPU), OpenCV, and NumPy.

---

## Features

- **Automatic image analysis** — OpenCV reads resolution, sharpness, brightness, and contrast before enhancement.
- **Three enhancement profiles**:

  | Profile     | What it does                                                        | Speed (CPU)      |
  |-------------|---------------------------------------------------------------------|------------------|
  | `LIGHT`     | CLAHE contrast correction + sharpening. Same resolution.            | ~1–2s            |
  | `SUPER_RES` | Real-ESRGAN 2x upscale with detail reconstruction.                  | 30s – 2min       |
  | `RESTORE`   | Noise reduction (fastNlMeansDenoising) + Real-ESRGAN 2x upscale.    | 45s – 2min       |

- **Tiled Real-ESRGAN inference** — 256px tiles with padding, keeping peak memory low (~1.5GB) so it runs on the free tier.
- **Before/after preview** and one-click **download** of the enhanced PNG.
- **Privacy-first** — files are temporary, processed in memory, never uploaded to any third party.
- **Fun progress messages** while the AI works.

---

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_model.py  # fetches model weights into models/
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (default http://localhost:8501).

- Model weights download automatically on first enhance if missing.
- Python **3.11 or 3.13** is recommended (the pinned torch wheel is cp311).

---

## Deploy to Streamlit Community Cloud (free)

1. **Push this repo to a public GitHub repo** (default branch `main`).
   - `models/` is gitignored — weights download at first use on the cloud; no need to commit ~145MB.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**.
3. Pick the repo → branch `main` → main file path **`streamlit_app.py`**.
4. Click **Advanced settings** and set **Python version: 3.11**.
   - `requirements.txt` pins CPU-only PyTorch wheels built for **cp311**. Streamlit Cloud defaults to 3.12, which cannot find those wheels — 3.11 is required.
5. Click **Deploy**. First build takes ~5–8 min (pip installs torch CPU). Then open the `*.streamlit.app` URL.

### Free tier limits you must know

| Resource      | Free tier      | Optic Pixel usage                          |
|---------------|----------------|--------------------------------------------|
| CPU           | 2 vCPU         | Real-ESRGAN inference (CPU threads)        |
| RAM           | up to 2.7 GB   | ~1.5 GB during inference, ~0.6 GB idle     |
| Disk          | ephemeral      | Models re-download on cold start           |
| Sleep         | ~12h idle      | First visit after sleep is slow (cold boot)|

- **Sleep:** free apps sleep after ~12h idle. The first visitor gets a "waking up" page, and model weights re-download on cold start, so the first enhancement after wake can take 1–2 min. After that it is instant.
- **Memory:** if you hit resource limits, upload smaller images (max 10MB / 4MP). The app caps output scale and uses tiled inference to stay under budget.

---

## Model & CPU limitations (read this)

Real-ESRGAN was designed for GPUs. On a CPU-only server every model choice is a trade-off between **output quality**, **memory**, and **speed**. Here is exactly how Optic Pixel handles it.

### Which models ship with the app

| Weights file                | Blocks | Size   | Use                                            |
|-----------------------------|--------|--------|------------------------------------------------|
| `RealESRGAN_x4plus.pth`     | 23     | ~64 MB | Default — general photos, best quality         |
| `RealESRGAN_x4plus_anime_6B.pth` | 6  | ~18 MB | Anime / illustrations, much faster, less RAM   |
| `RealESRGAN_x2plus.pth`     | 23     | ~33 MB | 2x-focused model, faster than the 4x model     |

All three are listed in `app/services/model_downloader.py` and fetched automatically. The app currently loads **`RealESRGAN_x4plus.pth`** (see `model_service.get_model()`).

### Why "very big models" are limited on CPU

Bigger models = more RRDB blocks and more parameters. On CPU:

1. **Memory:** a large model (e.g. 23-block 4x) plus activations for a 256px tile already needs ~1.5GB. Free-tier pods cap at 2.7GB. Very large models (more blocks / wider channels) or running the whole image in one pass will OOM-kill the pod.
2. **Speed:** CPU inference is ~20–50× slower than a mid-range GPU. A large model can push a single enhance past the 3-minute processing budget.
3. **Precision:** `half=True` (FP16) is unavailable on most CPU builds, so we run `half=False` (FP32), doubling memory vs. a GPU build.

### How to use a bigger / better model safely

The scaling knobs are already wired in:

- **Tile size** — inference runs tile-by-tile, so peak memory is bounded by *one tile*, not the whole image.
  - Lower `model_tile_size` (e.g. 256 → 128) → less RAM, slightly more border seams.
  - Raise it only if you have RAM to spare and want fewer seams.
  - `config.py`: `model_tile_size`; `model_service.get_model()` passes `tile=...`.
- **Output scale** — capped at 2x on CPU (`default_scale`/`max_scale` in `config.py`). 4x upscaling would produce output 4× the area, ballooning memory and time.
- **Input size** — the app suggests profiles by resolution; uploading smaller images keeps every stage fast.

**Concrete recommendations:**

| Environment                          | Recommended setup                                             |
|--------------------------------------|---------------------------------------------------------------|
| Streamlit free tier (2.7GB RAM)      | Keep `RealESRGAN_x4plus` with `tile=256`, `outscale=2`, ≤4MP input. This is the shipped default. |
| Local PC, ≥8GB RAM                   | You can use `tile=256–512` and `outscale=2–4` with the 23-block model for higher-quality results. |
| Local PC, ≥16GB RAM + patience       | Swap to a larger custom model (more blocks) by editing `model_service.get_model()` — just make sure `num_block` matches the weights file. |
| GPU / Colab / heavy batch            | Use the official `realesrgan` + Real-ESRGAN `4x` models at full resolution; tiling barely matters. |

> **Switching models:** to use a different weights file, change the filename in `model_service.get_model()` (and adjust `scale` + `num_block` to match the checkpoint — see `RRDBNet` defaults). The downloader already fetches all three weights, so the file will be present.

**If a tile fails with an out-of-memory error**, lower the tile size (256 → 128) and re-run. The tiling wrapper reports the failing tile number to help you size it.

---

## How it works

1. **Analyze** — OpenCV reads width/height, sharpness, brightness, contrast, pixel count.
2. **Choose profile** — the app preselects `LIGHT`, `SUPER_RES`, or `RESTORE` based on image quality; you can override it.
3. **Enhance** — tiled Real-ESRGAN inference on CPU (256px tiles, ~1.5GB peak).
4. **Download** — before/after shown side by side with a download button.

### Enhancement pipeline

```
upload ─► image_analyzer.analyze_image()  ─► profile + ImageAnalysis
          enhancement_service.run_enhancement()
            ├─ LIGHT  → CLAHE (LAB) + sharpen kernel
            ├─ RESTORE→ denoise + Real-ESRGAN
            └─ SUPER_RES → Real-ESRGAN 2x (tiled, 256px)
          model_service.enhance_image() → RealESRGANer.enhance()
          output PNG ─► preview + download
```

### Key settings (`app/config.py`)

| Setting                 | Default | Meaning                                  |
|-------------------------|---------|------------------------------------------|
| `max_upload_mb`         | 10      | Upload size cap                          |
| `max_image_pixels`      | 4M      | Largest input accepted                   |
| `default_scale`         | 2       | Upscale factor                           |
| `max_scale`             | 2       | Hard cap for CPU                         |
| `model_tile_size`       | 256     | Inference tile size (memory bound)       |
| `max_processing_seconds`| 180     | Processing budget                        |

---

## Project layout

```
streamlit_app.py            # Streamlit UI (deploy entrypoint)
app/
  config.py                 # settings (limits, tile size, scales)
  models/
    schemas.py              # EnhancementProfile, ImageAnalysis (pydantic)
  services/
    esrgan/                 # vendored RRDBNet + RealESRGANer (Apache-2.0)
    model_service.py        # model loading + inference
    model_downloader.py     # weights download (skip if present)
    enhancement_service.py  # profile selection + enhancement logic
    image_analyzer.py       # OpenCV analysis
scripts/
  download_model.py         # pre-fetch weights locally
  verify_model.py           # check weights exist and load
models/                     # weights (downloaded, gitignored)
tests/                      # unit tests (analyzer + enhancement)
```

---

## Running tests

```bash
python -m pytest tests -q
```

Covers the image analyzer and the `LIGHT` enhancement path (no weights required).

---

## Deploy to other platforms

- **Hugging Face Spaces (Docker):** requires a paid PRO plan to create a Docker Space ($9/mo). Not needed — use Streamlit Cloud above.
- **Render / Fly.io / Railway / any Docker VPS:** the repo no longer ships a Dockerfile; run the Streamlit-native path (`streamlit run streamlit_app.py`), or wrap it in a container manually.
- **Vercel:** not suitable — no long-running Python processes.

---

## Roadmap / possible improvements

- Make the active model selectable in the UI (e.g. `x4plus` vs `anime_6B`).
- Add a RAM-usage estimator before running the model.
- Optional GPU path (`half=True`, larger tiles) when a CUDA device is detected — the code already checks `torch.cuda.is_available()`.

---

## License

Real-ESRGAN model weights and vendored code are under [Apache-2.0](https://github.com/xinntao/Real-ESRGAN).
