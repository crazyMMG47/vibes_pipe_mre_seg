# Augmentation pipeline that combines the basic augmentation with mre subject's noise profile 
# to perform data augmentation on medical images

# Order matters when we augment our data!
# Augmentation order:
# 1. spatially augment (image, label) -> record transformation 
# 2. Apply the same spatial transformation to the noise profile of a subject
# 3. Add the transformed noise of the subject to the spatially augmented image.
import numpy as np
import random
from typing import Optional, Dict, Any, Tuple
from .noise_augmenter import NoiseAugmenter
from .basic_augment import SpatialAugmenter

class MREAugmentation:
    """
    Final augmentation pipeline to chain Spatial, Intensity, and Noise augmenters.
    
    This pipeline correctly does the following:
    1. Applies the *same* spatial transform to the image, label, and input noise field.
    2. Applies intensity augmentation to the spatially-transformed image.
    3. Adds *new* noise to the augmented image.
    """

    def __init__(self,
                 spatial_augmenter: Optional[SpatialAugmenter] = None,
                 noise_augmenter: Optional[NoiseAugmenter] = None,
                 apply_prob: float = 0.80):
        
        self.spatial_augmenter = spatial_augmenter
        self.noise_augmenter = noise_augmenter
        self.apply_prob = apply_prob
        
        print(f"✓ MREAugmentation pipeline initialized. Apply prob: {self.apply_prob*100}%")
        if self.spatial_augmenter:
            print("  → Spatial augmentations enabled.")
        if self.noise_augmenter:
            print("  → Additive noise augmentation enabled.")


    def __call__(self,
                 image: np.ndarray,
                 label: np.ndarray,
                 subject_id: str,
                 noise_field: Optional[np.ndarray] = None,
                 is_2d: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply the full augmentation chain.
        
        Args:
            image: Input image [D, H, W]
            label: Input label [D, H, W]
            subject_id: Subject ID for loading new noise
            noise_field: The noise profile from the dataset [D, H, W]
            is_2d: Flag for 2D/3D operations
        
        Returns:
            Augmented (image, label, noise_field) tuple
        """
        
        # --- 1. Randomly skip all augmentations ---
        if random.random() > self.apply_prob:
            return image, label, noise_field

        
        # --- 2. Spatial Augmentation ---
        spatial_params = None
        if self.spatial_augmenter is not None:
            # Apply aug to image & label, and get the params
            image, label, spatial_params = self.spatial_augmenter(
                image, label, return_params=True
            )
            
            # Apply the *same* transform to the dataset's noise field
            if noise_field is not None and spatial_params is not None:
                noise_field = self.spatial_augmenter.apply_to(
                    noise_field, 
                    spatial_params, 
                    is_label=False  # Noise is continuous, not discrete
                )
        
        
        
        # --- 4. Additive Noise Augmentation ---
        # Adds *new* noise to the already-augmented image
        if self.noise_augmenter is not None:
            # Load a new noise field for this subject
            new_noise_to_add = self.noise_augmenter.load_field(subject_id, is_2d)
            
            if new_noise_to_add is not None:
                # The .add() method handles resizing this new noise
                # to the (potentially scaled) image shape.
                image = self.noise_augmenter.add(image, new_noise_to_add)

        
        # --- 5. Return all augmented components ---
        return image, label, noise_field