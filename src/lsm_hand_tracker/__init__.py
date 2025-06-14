"""
lsm_hand_tracker
===============

Package for preprocessing, feature extraction, training
and gesture inference of Mexican Sign Language.
"""

__version__ = "0.1.0"

# Directory constants and utils
from .utils.path_config import (
    BASE_PATH,
    EXTERNAL_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    get_dir,
    set_base_path,
)


__all__ = [
    # version
    "__version__",
    # path constants
    "BASE_PATH",
    "EXTERNAL_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "RAW_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    # path utilities
    "get_dir",
    "set_base_path",
]
