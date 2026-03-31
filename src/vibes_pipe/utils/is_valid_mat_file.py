from typing import Iterable
from pathlib import Path

def is_valid_mat_file(path: Path, required_keywords: Iterable[str] = None) -> bool:
    """Check if file is a valid, non-corrupt .mat file."""
    try:
        # Basic checks
        if path.suffix.lower() != ".mat":
            return False
        if path.name.startswith("._"):  # Skip macOS hidden files
            return False
        if not path.is_file() or path.stat().st_size < 256:
            return False
        
        # Keyword check if specified
        if required_keywords:
            name_lower = path.name.lower()
            if not any(keyword.lower() in name_lower for keyword in required_keywords):
                return False
        
        return True
    except Exception:
        return False