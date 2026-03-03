""""
This module is used for extract the noise profile. 
"""

import json
import scipy.io as sio
from pathlib import Path
import numpy as np
from ge_noise import *
from siemens_noise import * 


# --- USER CONFIGURATION ---
SCANNER_MODE = "GE"  # Options: "GE" or "SIEMENS"
MANIFEST_PATH = "./workspace_root/manifest.json"
# --------------------------

def run_batch():
    """
    Assumption: Subjects from a particular cohort uses the same scanner. 
    
    This function is used to speed up the noise profile extraction. 
    """
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    workspace_root = Path(manifest["workspace_root"])
    # Create a new noise directory under workspace, store separately.
    noise_dir = workspace_root / "noise_profiles"
    noise_dir.mkdir(exist_ok=True)

    print(f"Starting batch processing for: {SCANNER_MODE}")

    for pair in manifest["pairs"]:
        pid = pair["id"]
        # Pull the source path from the manifest
        src_path = Path(pair["files"]["t2stack"]["src"])
        
        if not src_path.exists():
            print(f"Skip {pid}: File not found at {src_path}")
            continue

        # Load the data
        mat_data = sio.loadmat(src_path)
        # Note: adjust 'data' to whatever key your .mat files use
        raw_arr = mat_data.get('data') or mat_data.get('magimg') or mat_data.get('img')

        # Apply the chosen logic
        if SCANNER_MODE.upper() == "GE":
            _, noise, _, _ = compute_ge_noise(raw_arr)
        else:
            _, noise, _, _, _ = compute_siemens_noise(raw_arr)

        # Save specifically for the NoiseAugmenter to find
        save_path = noise_dir / f"{pid}_noise.mat"
        sio.savemat(save_path, {"noise": noise, "scanner_type": SCANNER_MODE})
        print(f"Saved: {save_path.name}")

if __name__ == "__main__":
    run_batch()