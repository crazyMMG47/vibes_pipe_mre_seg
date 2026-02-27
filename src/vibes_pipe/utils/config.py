""""
Tiny YAML loader to load our configs. 
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {p}")
    return data