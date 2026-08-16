import tempfile
from pathlib import Path
from app.services.image_analyzer import analyze_image


def _make_test_image(width=200, height=150, color=(100, 150, 200)):
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    return tmp.name


def test_analyze_basic():
    path = _make_test_image()
    try:
        result = analyze_image(path)
        assert result.width == 200
        assert result.height == 150
        assert result.pixel_count == 30_000
        assert result.profile is not None
        assert result.profile_description
    finally:
        Path(path).unlink()


def test_analyze_small_image():
    path = _make_test_image(100, 100)
    try:
        result = analyze_image(path)
        assert result.pixel_count == 10_000
    finally:
        Path(path).unlink()
