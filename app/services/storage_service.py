import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_local_outputs_dir = Path("local_outputs")


def _outputs_dir() -> Path:
    _local_outputs_dir.mkdir(parents=True, exist_ok=True)
    return _local_outputs_dir


def save_output(job_id: str, data: bytes, suffix: str = "png") -> Path:
    """Persist an output file locally and return its path."""
    dest = _outputs_dir() / f"{job_id}.{suffix}"
    dest.write_bytes(data)
    logger.info("Saved local output %s (%d bytes)", dest, len(data))
    return dest


def output_path(job_id: str, suffix: str = "png") -> Path | None:
    """Resolve the local path for a job output if it still exists."""
    path = _outputs_dir() / f"{job_id}.{suffix}"
    return path if path.exists() else None
