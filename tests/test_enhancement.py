import tempfile
from pathlib import Path
from PIL import Image
from app.models.schemas import EnhancementProfile, ImageAnalysis
from app.services.enhancement_service import _light_enhance


def _make_test_image(width=200, height=150):
    img = Image.new("RGB", (width, height), (100, 150, 200))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    return tmp.name


def _make_analysis(width=200, height=150, profile=EnhancementProfile.LIGHT):
    return ImageAnalysis(
        width=width, height=height, channels=3, mode="RGB", fmt="PNG",
        file_size_bytes=0, pixel_count=width * height, sharpness_score=100.0,
        mean_brightness=128.0, contrast_score=50.0, profile=profile,
        profile_description="Test",
    )


def test_light_enhance():
    input_path = _make_test_image()
    output_path = input_path + ".enhanced.png"
    try:
        w, h = _light_enhance(input_path, output_path)
        assert w == 200
        assert h == 150
        assert Path(output_path).exists()
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
