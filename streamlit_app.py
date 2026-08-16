"""Optic Pixel - AI image enhancement, Streamlit edition.

Reuses the vendored Real-ESRGAN pipeline from app/services.
Deploy target: Streamlit Community Cloud.
"""

import os
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

from app.config import settings
from app.models.schemas import EnhancementProfile
from app.services import enhancement_service, model_service
from app.services.image_analyzer import PROFILE_DESCRIPTIONS, analyze_image

st.set_page_config(
    page_title="Optic Pixel",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROFILE_LABELS = {
    EnhancementProfile.SUPER_RES: "✨ AI Super Resolution (upscale)",
    EnhancementProfile.RESTORE: "🩹 Restore (denoise + upscale)",
    EnhancementProfile.LIGHT: "💡 Light (contrast + sharpen)",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface-2: #1a1a26;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-dim: #888898;
    --accent: #4a9eff;
    --accent-dim: #2a6ecc;
    --success: #34d399;
    --ai: #a78bfa;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 4rem; }

.hero { text-align: center; padding: 30px 0 22px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.hero-badges { display: flex; justify-content: center; gap: 10px; margin-bottom: 16px; }
.badge {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 3px 12px;
    border-radius: 20px; border: 1px solid var(--border); color: var(--text-dim); background: var(--surface);
}
.badge.green { border-color: rgba(52,211,153,0.3); color: var(--success); }
.hero h1 {
    font-family: 'JetBrains Mono', monospace; font-size: 34px; font-weight: 700;
    letter-spacing: 10px; color: var(--accent); margin: 0;
}
.hero .tagline { font-size: 15px; font-weight: 500; margin-top: 10px; color: var(--text); }
.hero .sub { font-size: 12px; color: var(--text-dim); margin-top: 5px; font-family: 'JetBrains Mono', monospace; }

.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 22px 24px; margin-bottom: 16px;
}
.card h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px; color: var(--text); }
.card p, .card li { color: var(--text-dim); font-size: 13px; line-height: 1.7; }
.card ul { margin: 0; padding-left: 18px; }
.card b { color: var(--text); }

.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.grid .item {
    padding: 14px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
}
.grid .item h4 { font-size: 13px; margin-bottom: 4px; color: var(--text); }
.grid .item p, .grid .item li { font-size: 12px; line-height: 1.6; }
.grid .item ul { list-style: none; padding: 0; }
.grid .item li::before { content: "• "; color: var(--accent); }

.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.grid3 .item {
    padding: 14px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
}
.grid3 .item h4 { font-size: 13px; margin-bottom: 4px; color: var(--text); }
.grid3 .item p { font-size: 12px; line-height: 1.5; }

.num {
    width: 26px; height: 26px; background: var(--accent); color: #fff; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600;
    margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
}

.profile-item { display: flex; align-items: center; gap: 12px; padding: 12px 14px; background: var(--surface-2); border-radius: 8px; border: 1px solid var(--border); margin-bottom: 8px; }
.profile-item p { font-size: 12px; margin: 0; }
.profile-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 4px; white-space: nowrap; flex-shrink: 0; }
.profile-badge.light { background: rgba(52,211,153,0.15); color: var(--success); }
.profile-badge.super { background: rgba(74,158,255,0.15); color: var(--accent); }
.profile-badge.restore { background: rgba(167,139,250,0.15); color: var(--ai); }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px; color: var(--text-dim);
}
.chip.ai { color: var(--ai); border-color: rgba(167,139,250,0.3); }
.chip-group { margin-bottom: 12px; }
.chip-group:last-child { margin-bottom: 0; }
.chip-group h4 {
    font-size: 11px; color: var(--text-dim); margin: 0 0 6px; text-transform: uppercase;
    letter-spacing: 1px; font-family: 'JetBrains Mono', monospace;
}

.arch {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); background: var(--surface-2);
    padding: 14px; border-radius: 8px; border: 1px solid var(--border); overflow-x: auto; line-height: 1.6;
    white-space: pre; display: table; margin: 0 auto; text-align: left;
}

.resource-link {
    display: block; padding: 12px 14px; background: var(--surface-2); border-radius: 8px;
    border: 1px solid var(--border); text-decoration: none; margin-bottom: 10px;
}
.resource-link:hover { border-color: var(--accent); }
.resource-link h4 { font-size: 13px; color: var(--accent); margin: 0 0 2px; }
.resource-link p { font-size: 11px; margin: 0; }

.regards { text-align: center; padding: 6px 0; }
.regards h2 { font-size: 16px; font-weight: 600; margin: 0 0 10px; }
.regards p { color: var(--text-dim); font-size: 13px; margin: 0; }
.regards .signoff { font-size: 20px; font-weight: 600; color: var(--text); margin-top: 8px; }
.regards .role { font-size: 13px; color: var(--accent); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.regards .certs { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 12px; }
.cert-badge { font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 3px 10px; background: rgba(255,153,0,0.1); border: 1px solid rgba(255,153,0,0.3); color: #ff9900; border-radius: 4px; }
.regards .links { margin-top: 14px; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
.regards .links a {
    display: inline-flex; align-items: center; gap: 6px; color: var(--accent); text-decoration: none;
    font-size: 13px; padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border);
}
.regards .links a:hover { border-color: var(--accent); }

[data-testid="stFileUploader"] section {
    border: 2px dashed var(--border) !important; border-radius: 10px !important;
    background: var(--surface) !important; padding: 30px 20px !important; text-align: center;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] button {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
[data-testid="stFileUploader"] small { color: var(--text-dim) !important; }

.stButton > button, .stDownloadButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important; padding: 11px 24px !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--accent-dim) !important; }

[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace; }

[data-testid="stStatusWidget"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 8px; }
.stSuccess, .stInfo { background: var(--surface) !important; border-color: var(--border) !important; }

h1, h2, h3 { font-family: 'Inter', sans-serif; color: var(--text); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text-dim); }
[data-testid="stMarkdownContainer"] strong { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] p { color: var(--text-dim) !important; }

[data-testid="stRadio"] label { color: var(--text) !important; }
[data-testid="stRadio"] div[role="radiogroup"] > label {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 14px; margin: 4px;
}
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    border-color: var(--accent); background: rgba(74,158,255,0.08);
}

@media (max-width: 720px) {
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .hero h1 { font-size: 26px; letter-spacing: 7px; }
    .hero .tagline { font-size: 13px; }
    .hero .sub { font-size: 11px; }
    .hero-badges { flex-wrap: wrap; }
    .grid, .grid3 { grid-template-columns: 1fr; }
    .profile-item { flex-wrap: wrap; }
    .profile-badge { flex-shrink: 0; }
    .card { padding: 16px 14px; }
    .resource-link, .grid .item, .grid3 .item { padding: 12px; }
    .arch { font-size: 10px; padding: 10px; }
    [data-testid="stMetric"] { padding: 8px 12px; }
    .stButton > button, .stDownloadButton > button { padding: 10px 16px !important; }
}
</style>
"""


FUN_MESSAGES = [
    "AI is squinting at your pixels...",
    "Teaching neural networks to see better...",
    "Asking the computer to imagine higher resolution...",
    "Reconstructing details that never existed...",
    "Applying magic — the mathematical kind...",
    "Telling apart noise from actual content...",
    "Your image is getting a PhD in clarity...",
    "Running Real-ESRGAN — sounds cooler than it feels on CPU...",
    "Converting blurry dreams into sharp reality...",
    "The model is thinking very hard right now...",
    "Enhancing. This is what AI was made for.",
    "Every pixel matters. Processing them all.",
    "Super-resolution isn't super fast on CPU. Hold on.",
    "Your image is in the gym. Getting gains.",
    "Almost like Photoshop, but with math.",
    "Real-ESRGAN says hello. It's working.",
    "Convolutions happen. Details emerge.",
    "This would be instant on a GPU. Blame the CPU.",
    "Denoising. Sharpening. Upscaling. Repeat.",
    "The AI promises it's worth the wait.",
]


@st.cache_resource(show_spinner=False)
def _get_model():
    return model_service.get_model(settings.model_weights_dir)


def _profile_label(profile: EnhancementProfile) -> str:
    return PROFILE_LABELS[profile]


def _save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def _render_analysis(analysis):
    cols = st.columns(4)
    cols[0].metric("Resolution", f"{analysis.width} × {analysis.height}")
    cols[1].metric("Format", f"{analysis.fmt} · {analysis.mode}")
    cols[2].metric("Sharpness", f"{analysis.sharpness_score:.1f}")
    cols[3].metric("Brightness", f"{analysis.mean_brightness:.1f}")
    st.caption(
        f"Suggested profile: **{analysis.profile_description}** "
        f"({_profile_label(analysis.profile)})"
    )


def _run_enhancement(input_path: str, output_path: str, profile: EnhancementProfile):
    analysis = analyze_image(input_path)

    with st.status("Preparing AI model…", expanded=True) as status:
        _get_model()
        status.update(label="Model ready", state="complete")

    with st.status(f"Enhancing ({_profile_label(profile)})…", expanded=True) as status:
        placeholder = st.empty()
        result = {}

        def _work():
            result["w"], result["h"], result["scale"] = (
                enhancement_service.run_enhancement(input_path, output_path, analysis)
            )

        t0 = time.time()
        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        idx = 0
        while worker.is_alive():
            placeholder.markdown(f"### {FUN_MESSAGES[idx]}")
            time.sleep(2.5)
            idx = (idx + 1) % len(FUN_MESSAGES)
        worker.join()

        elapsed = time.time() - t0
        placeholder.empty()
        status.update(
            label=f"Done in {elapsed:.1f}s · {result['w']}×{result['h']} (scale {result['scale']})",
            state="complete",
        )

    return analysis, result["w"], result["h"], result["scale"], elapsed


def _render_hero():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badges">
                <span class="badge">v1.0.0</span>
                <span class="badge green">CPU Mode</span>
            </div>
            <h1>OPTIC PIXEL</h1>
            <p class="tagline">Dynamic AI Image Enhancement Platform</p>
            <p class="sub">Upload. Analyze. Enhance. Download. Files are temporary and removed on restart.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card(html: str):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def _render_about():
    _render_card(
        """
        <h2>About Optic Pixel</h2>
        <p>Optic Pixel is a lightweight AI image enhancement platform. It analyzes every upload with
        OpenCV and enhances it with Real-ESRGAN super-resolution — all on CPU, no GPU required.
        Images stay temporary: they are processed in memory, never stored or shared.</p>
        <div class="grid">
            <div class="item">
                <h4>Purpose</h4>
                <p>Sharpen, denoise, and upscale photos up to 2x using state-of-the-art AI reconstruction.</p>
            </div>
            <div class="item">
                <h4>Engine</h4>
                <p>Real-ESRGAN (PyTorch CPU) with tiled inference, wrapped in a friendly Streamlit interface.</p>
            </div>
            <div class="item">
                <h4>Who it's for</h4>
                <p>Photographers, designers, and anyone restoring old or low-resolution images.</p>
            </div>
            <div class="item">
                <h4>Status</h4>
                <p>Version 1.0.0 — hosted on Streamlit Community Cloud, open source, free to use.</p>
            </div>
        </div>
        """
    )


def _render_how_it_works():
    _render_card(
        """
        <h2>How It Works</h2>
        <div class="grid">
            <div class="item">
                <div class="num">1</div>
                <h4>Upload</h4>
                <p>Drag and drop or select a JPG, PNG, WebP, BMP, or TIFF image. Max 10MB / 4 megapixels.</p>
            </div>
            <div class="item">
                <div class="num">2</div>
                <h4>Analyze</h4>
                <p>OpenCV reads sharpness, brightness, contrast, and resolution. The system picks the best enhancement profile automatically.</p>
            </div>
            <div class="item">
                <div class="num">3</div>
                <h4>Enhance</h4>
                <p>Real-ESRGAN AI model runs super-resolution. Tiled inference on CPU. 2x upscale with detail reconstruction.</p>
            </div>
            <div class="item">
                <div class="num">4</div>
                <h4>Download</h4>
                <p>Get your enhanced image instantly. Files are temporary and cleared on server restart.</p>
            </div>
        </div>
        """
    )


def _render_profiles():
    _render_card(
        """
        <h2>Enhancement Profiles</h2>
        <div class="profile-item">
            <span class="profile-badge light">LIGHT</span>
            <p>CLAHE contrast correction + sharpening. Same resolution. Fast (~1-2s).</p>
        </div>
        <div class="profile-item">
            <span class="profile-badge super">SUPER_RES</span>
            <p>Real-ESRGAN 2x upscaling with detail reconstruction. 30s-2min on CPU.</p>
        </div>
        <div class="profile-item">
            <span class="profile-badge restore">RESTORE</span>
            <p>Noise reduction + Real-ESRGAN super-resolution. For noisy/low-quality input.</p>
        </div>
        """
    )


def _render_performance():
    _render_card(
        """
        <h2>Performance &amp; Requirements</h2>
        <div class="grid">
            <div class="item">
                <h4>CPU Inference</h4>
                <p>Real-ESRGAN runs on CPU (no GPU required). Tiled processing keeps memory under 2GB.</p>
            </div>
            <div class="item">
                <h4>Processing Time</h4>
                <ul>
                    <li>LIGHT profile: ~1-2 seconds</li>
                    <li>SUPER_RES 2x: 30s - 2min (CPU)</li>
                    <li>RESTORE: 45s - 2min (CPU)</li>
                </ul>
            </div>
            <div class="item">
                <h4>Limits</h4>
                <ul>
                    <li>Max upload: 10MB</li>
                    <li>Max resolution: 4 megapixels</li>
                    <li>Timeout: 3 minutes</li>
                    <li>Scale: 2x (locked for CPU)</li>
                </ul>
            </div>
            <div class="item">
                <h4>Memory</h4>
                <ul>
                    <li>Model: ~65MB weights</li>
                    <li>Inference: ~1.5GB RAM</li>
                    <li>Tiled: 256px tiles</li>
                </ul>
            </div>
        </div>
        """
    )


def _render_privacy():
    _render_card(
        """
        <h2>Privacy &amp; Data</h2>
        <div class="grid3">
            <div class="item">
                <h4>Temporary Storage</h4>
                <p>Uploaded and enhanced images are stored temporarily on the server's local disk. Files are not persisted — removed on server restart or redeploy.</p>
            </div>
            <div class="item">
                <h4>No Permanent Retention</h4>
                <p>No cloud storage is used. Images are processed in memory and served back to you directly. Nothing is shared or made public.</p>
            </div>
            <div class="item">
                <h4>No Signup Required</h4>
                <p>No accounts, no tracking, no cookies. Just upload, enhance, and download.</p>
            </div>
        </div>
        """
    )


def _render_tech_stack():
    _render_card(
        """
        <h2>Technology Stack</h2>
        <div class="chip-group">
            <h4>Backend</h4>
            <div class="chips">
                <span class="chip">Python 3.11</span>
                <span class="chip">Streamlit</span>
                <span class="chip">Pydantic</span>
            </div>
        </div>
        <div class="chip-group">
            <h4>AI / Image</h4>
            <div class="chips">
                <span class="chip ai">PyTorch</span>
                <span class="chip ai">Real-ESRGAN</span>
                <span class="chip ai">OpenCV</span>
                <span class="chip ai">Pillow</span>
                <span class="chip ai">NumPy</span>
            </div>
        </div>
        <div class="chip-group">
            <h4>Hosting</h4>
            <div class="chips">
                <span class="chip">Streamlit Community Cloud</span>
            </div>
        </div>
        """
    )


def _render_architecture():
    _render_card(
        """
        <h2>Architecture</h2>
        <div class="arch">Browser ──► Streamlit ──► Image Analysis (OpenCV)
                    │
                    ▼
          Real-ESRGAN Inference (PyTorch CPU)
                    │
                    ▼
            Local Disk (temporary)
                    │
                    ▼
                 Download</div>
        """
    )


def _render_resources():
    _render_card(
        """
        <h2>Resources</h2>
        <div class="grid">
            <a class="resource-link" href="https://github.com/xinntao/Real-ESRGAN" target="_blank">
                <h4>Real-ESRGAN</h4>
                <p>Open-source super-resolution model</p>
            </a>
            <a class="resource-link" href="https://streamlit.io/" target="_blank">
                <h4>Streamlit</h4>
                <p>Python app framework docs</p>
            </a>
        </div>
        """
    )


def _render_regards():
    st.markdown(
        """
        <div class="card regards">
            <h2>Regards</h2>
            <p class="signoff">Aveek Patel</p>
            <p class="role">AWS Certified Solutions Architect &amp; Developer | Cloud &amp; Systems Engineer</p>
            <div class="certs">
                <span class="cert-badge">AWS CCP</span>
                <span class="cert-badge">AWS SAA</span>
                <span class="cert-badge">AWS DVA</span>
                <span class="cert-badge">AWS AI Practitioner</span>
            </div>
            <div class="links">
                <a href="https://github.com/Aveek29" target="_blank">GitHub</a>
                <a href="https://aveekcloud-tech.vercel.app/" target="_blank">Portfolio</a>
                <a href="https://linkedin.com/in/aveek-patel-473996327" target="_blank">LinkedIn</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer():
    st.markdown(
        '<div style="text-align:center; color:#888898; font-size:12px; font-family:JetBrains Mono,monospace; padding:18px 0 6px;">'
        "Optic Pixel v1.0.0 &mdash; Real-ESRGAN inference &bull; PyTorch CPU engine &bull; "
        "Tiled super-resolution &bull; Temporary local storage</div>",
        unsafe_allow_html=True,
    )


st.markdown(CSS, unsafe_allow_html=True)
_render_hero()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
    help=f"Max {settings.max_upload_mb} MB · JPG, PNG, WebP, BMP, TIFF. Profile auto-suggested from image quality.",
)

if uploaded is None:
    st.info("⬆️ Upload an image above to get started.")
else:
    input_path = _save_upload(uploaded)
    analysis = analyze_image(input_path)
    st.subheader(f"📄 {uploaded.name}")
    _render_analysis(analysis)
    st.divider()

    profile_choice = st.radio(
        "Enhancement profile",
        options=list(EnhancementProfile),
        index=list(EnhancementProfile).index(analysis.profile),
        format_func=_profile_label,
        horizontal=True,
    )

    output_path = os.path.join(tempfile.gettempdir(), f"optic_pixel_{int(time.time()*1000)}.png")

    if st.button("🚀 Enhance", type="primary", use_container_width=True):
        result_analysis, w, h, scale, elapsed = _run_enhancement(
            input_path, output_path, profile_choice
        )
        st.success(
            f"Enhanced in **{elapsed:.1f}s** — {result_analysis.width}×{result_analysis.height} "
            f"→ **{w}×{h}** (scale {scale})"
        )
        left, right = st.columns(2)
        left.subheader("Before")
        left.image(input_path, use_container_width=True)
        right.subheader("After")
        right.image(output_path, use_container_width=True)
        with open(output_path, "rb") as fh:
            st.download_button(
                "⬇️ Download enhanced image",
                data=fh.read(),
                file_name=f"enhanced_{Path(uploaded.name).stem}.png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )

st.divider()
_render_about()
_render_how_it_works()
_render_profiles()
_render_performance()
_render_privacy()
_render_tech_stack()
_render_architecture()
_render_resources()
_render_regards()
_render_footer()
