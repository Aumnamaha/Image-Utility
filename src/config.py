"""Global configuration via Pydantic Settings."""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class ImageConfig(BaseSettings):
    """Configuration for image processing parameters.

    All values can be overridden via environment variables using the prefix ``IMG_``.
    For example, set ``IMG_DEFAULT_DEVICE=cuda`` to force GPU processing.
    """

    # --- resolution bounds ---------------------------------------------------
    min_resolution: int = 64
    max_resolution: int = 8192

    # --- dataset paths -------------------------------------------------------
    dataset_root: Path = Path(os.getenv("DATASET_ROOT", "./data"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./output"))

    # --- processing device ---------------------------------------------------
    default_device: Literal["cpu", "cuda"] = "cpu"

    # --- defaults ------------------------------------------------------------
    default_dtype: str = "float32"
    default_channels: int = 3

    class Config:
        env_prefix = "IMG_"
        extra = "ignore"


# Module-level singleton for convenience.
config = ImageConfig()


def get_config() -> ImageConfig:
    """Return the current (possibly environment-overridden) config."""
    return ImageConfig()
