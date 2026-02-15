# inspect_data.py 
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

def save_slice_previews(
    dataset_pkl: str,
    output_dir: str,
    slices: List[int],
    max_subjects: Optional[int] = None,
) -> None:
    dataset_pkl = str(Path(dataset_pkl).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_pkl, "rb") as f:
        data = pickle.load(f)

    samples = data["samples"]
    if max_subjects is not None:
        samples = samples[:max_subjects]

    for sample in samples:
        subject_id = sample.get("subject_id", "unknown")
        image = sample["image"]
        label = sample["label"]

        # ensure [H,W,D]
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]

        for z in slices:
            if z >= image.shape[2]:
                continue

            img = image[:, :, z]
            vmin, vmax = float(img.min()), float(img.max())
            img_norm = (img - vmin) / (vmax - vmin + 1e-8)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor("black")

            axes[0].imshow(img_norm, cmap="gray")
            axes[0].set_title(f"Magnitude (z={z})", color="white")
            axes[0].axis("off")

            axes[1].imshow(img_norm, cmap="gray")
            axes[1].imshow(label[:, :, z], cmap="magma", alpha=0.5 * (label[:, :, z] > 0))
            axes[1].set_title(f"Overlay (z={z})", color="white")
            axes[1].axis("off")

            fig.suptitle(f"Subject: {subject_id}", color="white", fontsize=14)

            out = Path(output_dir) / f"{subject_id}_slice_{z}.png"
            plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
