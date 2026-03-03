# noise_augmenter.py
# NoiseAugmenter: apply relative noise to images using matched profile
import numpy as np
import scipy.io as sio
import random
from scipy.ndimage import zoom
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple


class NoiseAugmenter:
    """
    Manages loading, matching, and applying scanner-specific noise profiles.
    
    This class combines the logic of the manager and the augmenter.
    It loads all profiles at initialization and provides methods to:
    1. `load_field`: Get a specific subject's noise array.
    2. `add`: Apply a given noise array to an image with relative strength.
    """
    
    def __init__(self, 
                 noise_dir: Union[str, Path],
                 noise_strength_range: Tuple[float, float] = (0.05, 0.15)):
        """
        Initialize the noise augmenter.
        
        Args:
            noise_dir: Directory containing all noise profiles (*_noise.mat)
            noise_strength_range: Noise strength as a fraction of image std.
        """
        self.noise_dir = Path(noise_dir)
        self.noise_strength_range = noise_strength_range
        self.noise_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        self.available_subjects: List[str] = []
        
        # Load all profiles on initialization
        self._load_all_profiles()
        
        if self.available_subjects:
            print(f"✓ NoiseAugmenter initialized. Found {len(self.available_subjects)} noise profiles.")
        else:
            print(f"WARNING: NoiseAugmenter initialized, but no noise profiles were found in {self.noise_dir}")

    
    def _load_all_profiles(self):
        """Load all noise profiles from the specified directory."""
        if not self.noise_dir.exists():
            raise ValueError(f"Noise directory does not exist: {self.noise_dir}")
        
        noise_files = sorted(self.noise_dir.glob("*_noise.mat"))
        
        if not noise_files:
            print(f"Warning: No *_noise.mat files found in {self.noise_dir}")
            return
        
        for noise_file in noise_files:
            try:
                # Extract subject ID (e.g., S001_noise.mat -> S001)
                subject_id = noise_file.stem.replace('_noise', '')
                
                # Load noise profile
                noise_data = sio.loadmat(str(noise_file))
                if 'noise' in noise_data:
                    noise_array = noise_data.get('noise')
                elif 'noise_scaled' in noise_data:
                    noise_array = noise_data.get('noise_scaled')
                else:
                    noise_array = None
                
                if noise_array is None:
                    continue
                
                noise_array = np.asarray(noise_array).squeeze()
                
                # Store profile
                self.noise_profiles[subject_id] = {
                    'noise': noise_array,
                    'shape': noise_array.shape
                }
                self.available_subjects.append(subject_id)
                
            except Exception as e:
                print(f"Failed to load {noise_file.name}: {e}")
    
    
    def _get_profile(self, subject_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Find the best-matching noise profile for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S001', 'G028')
        
        Returns:
            Dictionary containing noise array or None if not found
        """
        # 1. Direct match
        if subject_id in self.noise_profiles:
            return self.noise_profiles[subject_id]
        
        # 2. Fuzzy match: subject_id starts with profile_id
        for profile_id in self.available_subjects:
            if subject_id.startswith(profile_id):
                return self.noise_profiles[profile_id]
        
        # 3. Partial match: profile_id is in subject_id
        for profile_id in self.available_subjects:
            if profile_id in subject_id:
                return self.noise_profiles[profile_id]
        
        # No match found
        return None
    
    
    def has_profile(self, subject_id: str) -> bool:
        """Check if a matching noise profile exists for a subject."""
        return self._get_profile(subject_id) is not None
    
    
    def load_field(self, subject_id: str, is_2d: bool = False) -> Optional[np.ndarray]:
        """
        Load the raw noise field for a subject.
        
        This is the method your `MREAugmentation` pipeline should call first.
        
        Args:
            subject_id: Subject identifier.
            is_2d: If True, extracts the middle slice from a 3D noise profile.
        
        Returns:
            Noise field array or None if profile not found.
        """
        profile = self._get_profile(subject_id)
        if profile is None:
            return None
        
        noise_profile = profile['noise']
        
        # Extract 2D slice if needed
        if is_2d and noise_profile.ndim == 3:
            mid_slice = noise_profile.shape[2] // 2
            return noise_profile[:, :, mid_slice]
        
        return noise_profile

    
    def add(self, image: np.ndarray, noise_field: np.ndarray) -> np.ndarray:
        """
        Adds a pre-loaded (and possibly pre-warped) noise field to an image.
        
        This is the method your `MREAugmentation` pipeline should call *after*
        it has applied spatial transforms to the noise field.
        
        Args:
            image: Input image.
            noise_field: The noise field to apply (already spatially warped).
        
        Returns:
            The image with noise added.
        """
        
        # --- 1. Match dimensions (resize noise to image) ---
        if noise_field.shape != image.shape:
            zoom_factors = tuple(img_dim / noise_dim 
                                for img_dim, noise_dim in zip(image.shape, noise_field.shape))
            noise_field = zoom(noise_field, zoom_factors, order=1)
        
        
        # --- 2. Apply relative noise ---
        
        # Normalize noise to unit variance
        valid_noise = noise_field[np.isfinite(noise_field)]
        if valid_noise.size == 0:
            return image
        
        noise_std = np.std(valid_noise)
        if noise_std == 0:
            return image
        
        normalized_noise = (noise_field - np.mean(valid_noise)) / noise_std
        
        # Calculate image std (only from non-zero voxels)
        valid_image = image[image > 0]
        if valid_image.size == 0:
            return image
        
        image_std = np.std(valid_image)
        
        # Pick a random noise strength
        noise_strength = random.uniform(*self.noise_strength_range)
        
        # Scale and add noise
        scaled_noise = normalized_noise * image_std * noise_strength
        noisy_image = image + scaled_noise
        
        # Preserve non-negative values if original image was non-negative
        if image.min() >= 0:
            noisy_image = np.maximum(noisy_image, 0)
        
        return noisy_image