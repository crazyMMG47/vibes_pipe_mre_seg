import json
from pathlib import Path

from vibes_pipe.data.processor import ImageProcessor

def register_subject(image_path, label_path, database_dir):
    """
    1. Processes a raw pair.
    2. Saves standardized versions to database_dir/processed.
    3. Updates a JSON manifest (the 'Database').
    """
    # TODO: Call the image processor 
    
    
    entry = {
        "subject_id": Path(image_path).stem,
        "original_path": str(image_path),
        "processed_path": "path/to/db/processed/sub_01.npy",
        "timestamp": "2026-02-03"
    }
    # Update master_manifest.json