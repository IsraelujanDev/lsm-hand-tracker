# src/lsm_hand_tracker/__init__.py

"""
lsm_hand_tracker
===============

Package for preprocessing, feature extraction, training
and gesture inference of Mexican Sign Language.
"""

__version__ = "0.1.0"

# Configuration utilities and directory constants
from . import path_config

# Core pipeline
from .metadata import generate_metadata
from .json_to_csv import flatten_metadata_to_csv


__all__ = [
    "path_config",
    "generate_metadata",
    "flatten_metadata_to_csv",
]
