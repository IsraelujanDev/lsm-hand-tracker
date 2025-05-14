import os
from pathlib import Path

# 1) Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# 2) Determine base path
def _get_base_path() -> Path:
    env = os.getenv("LSM_BASE")
    if env:
        return Path(env)
    cwd = Path.cwd()
    print(f"⚠️ Environment variable 'LSM_BASE' not set. Using current directory: {cwd}")
    return cwd

BASE_PATH = _get_base_path()

# 3) Define all directories
RAW_DIR           = BASE_PATH / "data" / "raw"
PROCESSED_DIR     = BASE_PATH / "data" / "processed"
IMAGES_DIR        = PROCESSED_DIR / "images"
VIDEOS_DIR        = PROCESSED_DIR / "videos"
NOT_DETECTED_DIR  = PROCESSED_DIR / "not_detected"
METADATA_DIR      = PROCESSED_DIR / "metadata"
MODELS_DIR        = BASE_PATH / "models"
MODEL_PATH        = MODELS_DIR / "hand_landmarker.task"

# 4) Ensure they exist
for d in (RAW_DIR, IMAGES_DIR, VIDEOS_DIR, NOT_DETECTED_DIR, METADATA_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 5) Registry for lookup
_DIRS = {
    "base":         BASE_PATH,
    "raw":          RAW_DIR,
    "processed":    PROCESSED_DIR,
    "images":       IMAGES_DIR,
    "videos":       VIDEOS_DIR,
    "not_detected": NOT_DETECTED_DIR,
    "metadata":     METADATA_DIR,
    "models":       MODELS_DIR,
}

def get_dir(name: str) -> Path:
    """
    Return the Path for one of the keys:
    base, raw, processed, images, videos, not_detected, metadata, models.
    """
    try:
        return _DIRS[name]
    except KeyError:
        valid = ", ".join(_DIRS.keys())
        raise KeyError(f"Unknown directory '{name}'. Valid keys: {valid}")

def set_base_path(path):
    """
    Override BASE_PATH and recompute all directories.
    """
    global BASE_PATH, RAW_DIR, PROCESSED_DIR, IMAGES_DIR
    global VIDEOS_DIR, NOT_DETECTED_DIR, METADATA_DIR, MODELS_DIR, MODEL_PATH, _DIRS

    BASE_PATH     = Path(path)
    RAW_DIR       = BASE_PATH / "data" / "raw"
    PROCESSED_DIR = BASE_PATH / "data" / "processed"
    IMAGES_DIR    = PROCESSED_DIR / "images"
    VIDEOS_DIR    = PROCESSED_DIR / "videos"
    NOT_DETECTED_DIR = PROCESSED_DIR / "not_detected"
    METADATA_DIR  = PROCESSED_DIR / "metadata"
    MODELS_DIR    = BASE_PATH / "models"
    MODEL_PATH    = MODELS_DIR / "hand_landmarker.task"

    # Recreate dirs
    for d in (RAW_DIR, IMAGES_DIR, VIDEOS_DIR, NOT_DETECTED_DIR, METADATA_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Update registry
    _DIRS = {
        "base":         BASE_PATH,
        "raw":          RAW_DIR,
        "processed":    PROCESSED_DIR,
        "images":       IMAGES_DIR,
        "videos":       VIDEOS_DIR,
        "not_detected": NOT_DETECTED_DIR,
        "metadata":     METADATA_DIR,
        "models":       MODELS_DIR,
    }
