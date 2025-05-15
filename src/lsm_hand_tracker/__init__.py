# src/lsm_hand_tracker/__init__.py

"""
lsm_hand_tracker
===============

Package for preprocessing, feature extraction, training
and gesture inference of Mexican Sign Language.
"""

__version__ = "0.1.0"

# Core pipeline
from .metadata import generate_metadata

# Configuration utilities and directory constants
from . import path_config

__all__ = [
    "generate_metadata",
    "path_config",
]
