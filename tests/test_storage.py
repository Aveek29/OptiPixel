import os
import tempfile
from pathlib import Path
from app.services.cleanup_service import cleanup_temp_files, cleanup_temp_dir


def test_cleanup_temp_files():
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    f.write(b"test")
    f.close()
    assert os.path.exists(f.name)
    cleanup_temp_files(f.name)
    assert not os.path.exists(f.name)


def test_cleanup_missing_file():
    cleanup_temp_files("/nonexistent/path/file.tmp")


def test_cleanup_temp_dir():
    tmpdir = tempfile.mkdtemp()
    (Path(tmpdir) / "test.txt").write_text("hello")
    cleanup_temp_dir(tmpdir)
    assert not os.path.exists(tmpdir)
