from __future__ import annotations
import os
from pathlib import Path


class Config:
    output_dir: Path
    manifest_path: Path | None
    workspace_root: Path | None
    port: int

    def __init__(self) -> None:
        raw_output = os.environ.get("OUTPUT_DIR", "")
        if not raw_output:
            raise RuntimeError(
                "OUTPUT_DIR env var is required. "
                "Set it to the InferenceEngine save_dir (e.g. /path/to/output)."
            )
        self.output_dir = Path(raw_output).expanduser().resolve()

        mp = os.environ.get("MANIFEST_PATH", "")
        self.manifest_path = Path(mp).expanduser().resolve() if mp else None

        wr = os.environ.get("WORKSPACE_ROOT", "")
        self.workspace_root = Path(wr).expanduser().resolve() if wr else None

        self.port = int(os.environ.get("PORT", "8000"))

    def log_startup(self) -> None:
        ok = "\033[32m✓\033[0m"
        warn = "\033[33m⚠\033[0m"
        print(f"\n  vibes_pipe GUI — backend config")
        print(f"  {ok}  OUTPUT_DIR      : {self.output_dir} {'(exists)' if self.output_dir.exists() else '(NOT FOUND)'}")
        if self.manifest_path:
            exists = self.manifest_path.exists()
            print(f"  {ok if exists else warn}  MANIFEST_PATH   : {self.manifest_path} {'(exists)' if exists else '(NOT FOUND — manifest endpoints degraded)'}")
        else:
            print(f"  {warn}  MANIFEST_PATH   : not set — manifest endpoints disabled")
        if self.workspace_root:
            exists = self.workspace_root.exists()
            print(f"  {ok if exists else warn}  WORKSPACE_ROOT  : {self.workspace_root} {'(exists)' if exists else '(NOT FOUND)'}")
        else:
            print(f"  {warn}  WORKSPACE_ROOT  : not set — pseudo-GT export disabled")
        print(f"  {ok}  PORT            : {self.port}\n")


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
