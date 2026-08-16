from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"
    debug: bool = False

    max_upload_mb: int = 10
    max_image_pixels: int = 4_000_000
    max_output_pixels: int = 16_000_000

    default_scale: int = 2
    max_scale: int = 2
    model_tile_size: int = 256
    max_processing_seconds: int = 180

    model_weights_dir: str = str(Path(__file__).resolve().parent.parent / "models")


settings = Settings()
