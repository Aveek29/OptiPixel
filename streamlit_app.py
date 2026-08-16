"""Optic Pixel - AI image enhancement, Streamlit edition.

Reuses the vendored Real-ESRGAN pipeline from app/services.
Deploy target: Streamlit Community Cloud.
"""

import os
import tempfile
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
    initial_sidebar_state="expanded",
)

PROFILE_LABELS = {
    EnhancementProfile.SUPER_RES: "✨ AI Super Resolution (upscale)",
    EnhancementProfile.RESTORE: "🩹 Restore (denoise + upscale)",
    EnhancementProfile.LIGHT: "💡 Light (contrast + sharpen)",
}


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
        model = _get_model()
        status.update(label="Model ready", state="complete")

    with st.status(f"Enhancing ({_profile_label(profile)})…", expanded=True) as status:
        t0 = time.time()
        w, h, scale = enhancement_service.run_enhancement(
            input_path, output_path, analysis
        )
        elapsed = time.time() - t0
        status.update(
            label=f"Done in {elapsed:.1f}s · {w}×{h} (scale {scale})",
            state="complete",
        )

    return analysis, w, h, scale, elapsed


def _render_header():
    top = st.columns([1, 3, 1])
    top[2].markdown(
        "`v1.0.0` `CPU Mode`",
        unsafe_allow_html=True,
    )
    st.markdown("## ✨ OPTIC PIXEL")
    st.markdown("**Dynamic AI Image Enhancement Platform**")
    st.caption("Upload. Analyze. Enhance. Download. Files are temporary and removed on restart.")


def _render_how_it_works():
    st.subheader("How It Works")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1 · Upload**\n\nDrag & drop a JPG, PNG, WebP, BMP, TIFF, or GIF. Max 10MB / 4 megapixels.")
    c2.markdown("**2 · Analyze**\n\nOpenCV reads sharpness, brightness, contrast, and resolution. The system picks the best profile automatically.")
    c3.markdown("**3 · Enhance**\n\nReal-ESRGAN AI runs tiled super-resolution on CPU. 2x upscale with detail reconstruction.")
    c4.markdown("**4 · Download**\n\nGet your enhanced image instantly. Files are temporary and cleared on restart.")


def _render_profiles():
    st.subheader("Enhancement Profiles")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**`LIGHT`**\n\nCLAHE contrast correction + sharpening. Same resolution. Fast (~1-2s).")
    c2.markdown("**`SUPER_RES`**\n\nReal-ESRGAN 2x upscaling with detail reconstruction. 30s-2min on CPU.")
    c3.markdown("**`RESTORE`**\n\nNoise reduction + Real-ESRGAN super-resolution. For noisy/low-quality input.")


def _render_performance():
    st.subheader("Performance & Requirements")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**CPU Inference**\n\nReal-ESRGAN runs on CPU (no GPU required). Tiled processing keeps memory under 2GB.")
    c2.markdown("**Processing Time**\n\n- LIGHT: ~1-2s\n- SUPER_RES 2x: 30s-2min\n- RESTORE: 45s-2min")
    c3.markdown("**Limits**\n\n- Max upload: 10MB\n- Max resolution: 4MP\n- Timeout: 3 min\n- Scale: 2x (CPU)")
    c4.markdown("**Memory**\n\n- Model: ~65MB weights\n- Inference: ~1.5GB RAM\n- Tiled: 256px tiles")


def _render_privacy():
    st.subheader("Privacy & Data")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Temporary Storage**\n\nUploaded and enhanced images are stored temporarily on the server's local disk. Files are not persisted — removed on restart or redeploy.")
    c2.markdown("**No Permanent Retention**\n\nNo cloud storage is used. Images are processed in memory and served back directly. Nothing is shared or made public.")
    c3.markdown("**No Signup Required**\n\nNo accounts, no tracking, no cookies. Just upload, enhance, and download.")


def _render_tech_stack():
    st.subheader("Technology Stack")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Backend**\n\n`Python 3.11` `Streamlit` `Pydantic`")
    c2.markdown("**AI / Image**\n\n`PyTorch` `Real-ESRGAN` `OpenCV` `Pillow` `NumPy`")
    c3.markdown("**Hosting**\n\n`Streamlit Community Cloud` (free, 2 CPU / 2.7GB RAM)")


def _render_architecture():
    st.subheader("Architecture")
    st.code(
        "Browser ──► Streamlit ──► Image Analysis (OpenCV)\n"
        "                    │\n"
        "                    ▼\n"
        "          Real-ESRGAN Inference (PyTorch CPU)\n"
        "                    │\n"
        "                    ▼\n"
        "            Local Disk (temporary)\n"
        "                    │\n"
        "                    ▼\n"
        "                 Download",
        language="text",
    )


def _render_resources():
    st.subheader("Resources")
    c1, c2 = st.columns(2)
    c1.markdown(
        "[**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — open-source super-resolution model"
    )
    c2.markdown(
        "[**Streamlit**](https://streamlit.io/) — Python app framework docs"
    )


def _render_creator():
    st.subheader("Creator")
    st.markdown(
        "**Aveek Patel** — AWS Certified Solutions Architect & Developer | Cloud & Systems Engineer"
    )
    st.markdown("`AWS CCP`  `AWS SAA`  `AWS DVA`  `AWS AI Practitioner`")
    c1, c2, c3 = st.columns(3)
    c1.markdown("[GitHub](https://github.com/Aveek29)")
    c2.markdown("[Portfolio](https://aveekcloud-tech.vercel.app/)")
    c3.markdown("[LinkedIn](https://linkedin.com/in/aveek-patel-473996327)")


def _render_footer():
    st.divider()
    st.caption(
        "Optic Pixel v1.0.0 — Real-ESRGAN inference · PyTorch CPU engine · "
        "Tiled super-resolution · Temporary local storage"
    )


_render_header()

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"Max upload: **{settings.max_upload_mb} MB**\n\n"
        "Profile auto-suggested from image quality."
    )

if uploaded is None:
    st.info("⬆️ Upload an image from the sidebar to get started.")
else:
    input_path = _save_upload(uploaded)
    analysis = analyze_image(input_path)
    st.subheader(f"📄 {uploaded.name}")
    _render_analysis(analysis)
    st.divider()

    col_left, col_mid, col_right = st.columns([1, 2, 1])
    profile_choice = col_mid.radio(
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
_render_how_it_works()
st.divider()
_render_profiles()
st.divider()
_render_performance()
st.divider()
_render_privacy()
st.divider()
_render_tech_stack()
st.divider()
_render_architecture()
st.divider()
_render_resources()
st.divider()
_render_creator()
_render_footer()
