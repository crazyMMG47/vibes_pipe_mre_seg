from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

JsonDict = Dict[str, Any]


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=dst.parent) as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


def sha256_file(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    if not src_p.exists() or not src_p.is_file():
        raise FileNotFoundError(f"Source file not found: {src_p}")

    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if dst_p.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists (overwrite=False): {dst_p}")

    with src_p.open("rb") as sf, dst_p.open("wb") as df:
        while True:
            chunk = sf.read(1024 * 1024)
            if not chunk:
                break
            df.write(chunk)
    return dst_p


def iso8601_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")