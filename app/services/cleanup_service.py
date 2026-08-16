import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_temp_files(*paths: str | Path) -> None:
    """Remove temporary files, logging but not raising on failure."""
    for p in paths:
        try:
            path = Path(p)
            if path.exists():
                path.unlink()
                logger.debug("Removed temp file: %s", path)
        except OSError as e:
            logger.warning("Failed to remove temp file %s: %s", p, e)


def cleanup_temp_dir(dir_path: str | Path) -> None:
    """Remove a temporary directory and its contents."""
    import shutil

    try:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path)
            logger.debug("Removed temp directory: %s", path)
    except OSError as e:
        logger.warning("Failed to remove temp dir %s: %s", dir_path, e)
