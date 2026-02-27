"""
Data layer modules for MRE preprocessing and workspace preparation.
"""

from .pipeline_prep import build_workspace_from_pairs
from .manifest import read_json, write_json_atomic

__all__ = ["build_workspace_from_pairs", "read_json", "write_json_atomic"]