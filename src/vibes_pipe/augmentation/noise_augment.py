# noise_augment.py
# NoiseAugmenter: apply relative noise to images using either:
# 1) a noise_path provided by the manifest, or
# 2) a subject_id matched against preloaded profiles in noise_dir

import numpy as np
import scipy.io as sio
import random
from scipy.ndimage import zoom
from pathlib import Path
from typing import Dict, Optional, Union, Tuple


class NoiseAugmenter:
    """
    Manages loading and applying scanner-specific noise profiles.

    Supports two modes:
    1. Manifest-driven: load a noise field directly from `noise_path`
    2. Directory-driven: preload profiles from `noise_dir` and match by `subject_id`
    """

    def __init__(
        self,
        noise_dir: Optional[Union[str, Path]] = None,
        noise_strength_range: Tuple[float, float] = (0.05, 0.15),
    ):
        """
        Args:
            noise_dir: Optional directory containing *_noise.mat profiles.
                       Can be None if noise paths come directly from the manifest.
            noise_strength_range: Noise strength as a fraction of image std.
        """
        self.noise_dir = Path(noise_dir) if noise_dir is not None else None
        self.noise_strength_range = tuple(noise_strength_range)

        self.noise_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        self.available_subjects = []

        if self.noise_dir is not None:
            self._load_all_profiles()

            if self.available_subjects:
                print(
                    f"✓ NoiseAugmenter initialized. Found "
                    f"{len(self.available_subjects)} noise profiles."
                )
            else:
                print(
                    f"WARNING: NoiseAugmenter initialized, but no noise profiles "
                    f"were found in {self.noise_dir}"
                )
        else:
            print("✓ NoiseAugmenter initialized in manifest-path mode (no noise_dir).")

    def _extract_noise_array(self, noise_data: dict) -> Optional[np.ndarray]:
        """Extract the noise array from a loaded .mat dict."""
        if "noise" in noise_data:
            noise_array = noise_data["noise"]
        elif "noise_scaled" in noise_data:
            noise_array = noise_data["noise_scaled"]
        else:
            return None

        return np.asarray(noise_array).squeeze()

    def _load_mat_noise(self, mat_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load a single .mat noise file."""
        mat_path = Path(mat_path)

        if not mat_path.exists():
            print(f"Warning: noise file does not exist: {mat_path}")
            return None

        try:
            noise_data = sio.loadmat(str(mat_path))
            return self._extract_noise_array(noise_data)
        except Exception as e:
            print(f"Failed to load noise file {mat_path}: {e}")
            return None

    def _load_all_profiles(self):
        """Load all noise profiles from noise_dir."""
        if self.noise_dir is None:
            return

        if not self.noise_dir.exists():
            raise ValueError(f"Noise directory does not exist: {self.noise_dir}")

        noise_files = sorted(self.noise_dir.glob("*_noise.mat"))

        if not noise_files:
            print(f"Warning: No *_noise.mat files found in {self.noise_dir}")
            return

        for noise_file in noise_files:
            try:
                subject_id = noise_file.stem.replace("_noise", "")
                noise_array = self._load_mat_noise(noise_file)

                if noise_array is None:
                    continue

                self.noise_profiles[subject_id] = {
                    "noise": noise_array,
                    "shape": noise_array.shape,
                }
                self.available_subjects.append(subject_id)

            except Exception as e:
                print(f"Failed to load {noise_file.name}: {e}")

    def _get_profile(self, subject_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Find the best matching preloaded profile for a subject_id."""
        if subject_id in self.noise_profiles:
            return self.noise_profiles[subject_id]

        for profile_id in self.available_subjects:
            if subject_id.startswith(profile_id):
                return self.noise_profiles[profile_id]

        for profile_id in self.available_subjects:
            if profile_id in subject_id:
                return self.noise_profiles[profile_id]

        return None

    def has_profile(self, subject_id: str) -> bool:
        """Check if a matching preloaded profile exists for a subject."""
        return self._get_profile(subject_id) is not None

    def load_field(
        self,
        subject_id: Optional[str] = None,
        noise_path: Optional[Union[str, Path]] = None,
        is_2d: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Load a raw noise field.

        Priority:
        1. If `noise_path` is provided, load directly from that file.
        2. Else if `subject_id` is provided and profiles were preloaded from `noise_dir`,
           retrieve the matched subject profile.

        Args:
            subject_id: Subject identifier for directory-driven lookup.
            noise_path: Exact manifest-provided path to a noise .mat file.
            is_2d: If True and the noise is 3D, extract the middle slice.

        Returns:
            Noise field array or None.
        """
        noise_profile = None

        if noise_path is not None:
            noise_profile = self._load_mat_noise(noise_path)

        elif subject_id is not None:
            profile = self._get_profile(subject_id)
            if profile is not None:
                noise_profile = profile["noise"]

        if noise_profile is None:
            return None

        if is_2d and noise_profile.ndim == 3:
            mid_slice = noise_profile.shape[2] // 2
            return noise_profile[:, :, mid_slice]

        return noise_profile

    def add(self, image: np.ndarray, noise_field: np.ndarray) -> np.ndarray:
        """
        Add a pre-loaded, optionally pre-warped noise field to an image.
        """
        if noise_field.shape != image.shape:
            zoom_factors = tuple(
                img_dim / noise_dim
                for img_dim, noise_dim in zip(image.shape, noise_field.shape)
            )
            noise_field = zoom(noise_field, zoom_factors, order=1)

        valid_noise = noise_field[np.isfinite(noise_field)]
        if valid_noise.size == 0:
            return image

        noise_std = np.std(valid_noise)
        if noise_std == 0:
            return image

        normalized_noise = (noise_field - np.mean(valid_noise)) / noise_std

        valid_image = image[image > 0]
        if valid_image.size == 0:
            return image

        image_std = np.std(valid_image)
        noise_strength = random.uniform(*self.noise_strength_range)

        scaled_noise = normalized_noise * image_std * noise_strength
        noisy_image = image + scaled_noise # key to add noise per subject

        if image.min() >= 0:
            noisy_image = np.maximum(noisy_image, 0)

        return noisy_image


def build_noise_augmenter(cfg: dict) -> Optional[NoiseAugmenter]:
    """
    Builder for NoiseAugmenter.

    Supports either:
    - augmentation.noise.noise_dir
    - or no noise_dir at all when the manifest provides noise_path
    """
    aug_cfg = cfg.get("augmentation", {})
    noise_cfg = aug_cfg.get("noise", {})

    if not aug_cfg.get("enabled", False):
        return None
    if not noise_cfg.get("enabled", False):
        return None

    noise_dir = noise_cfg.get("noise_dir", None)
    strength = noise_cfg.get("noise_strength_range", [0.05, 0.15])

    return NoiseAugmenter(
        noise_dir=noise_dir,
        noise_strength_range=tuple(strength),
    )