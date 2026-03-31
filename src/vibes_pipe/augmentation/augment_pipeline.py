# Augmentation pipeline that combines the basic augmentation with mre subject's noise profile 
# to perform data augmentation on medical images

# Order matters when we augment our data!
# Augmentation order:
# 1. spatially augment (image, label) -> record transformation 
# 2. Apply the same spatial transformation to the noise profile of a subject
# 3. Add the transformed noise of the subject to the spatially augmented image.
import numpy as np
import random
from typing import Optional, Tuple
from src.vibes_pipe.augmentation.noise_augment import NoiseAugmenter
from src.vibes_pipe.augmentation.basic_augment import SpatialAugmenter

class MREAugmentation:
    """
    Augmentation pipeline:
    1. Spatially augment image + label, while recording params
    2. Apply same spatial transform to noise fields
    3. Add transformed noise to transformed image
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

        if random.random() > self.apply_prob:
            return image, label, noise_field

        spatial_params = None

        # 1. Spatial augmentation on image + label
        if self.spatial_augmenter is not None:
            image, label, spatial_params = self.spatial_augmenter(
                image, label, return_params=True
            )

            # 2. Apply same spatial transform to existing dataset noise field
            if noise_field is not None and spatial_params is not None:
                noise_field = self.spatial_augmenter.apply_to(
                    noise_field,
                    spatial_params,
                    is_label=False
                )

        # 3. Load new noise, spatially transform it too, then add it
        if self.noise_augmenter is not None:
            new_noise_to_add = self.noise_augmenter.load_field(subject_id, is_2d)

            if new_noise_to_add is not None:
                if self.spatial_augmenter is not None and spatial_params is not None:
                    new_noise_to_add = self.spatial_augmenter.apply_to(
                        new_noise_to_add,
                        spatial_params,
                        is_label=False
                    )

                image = self.noise_augmenter.add(image, new_noise_to_add)

        return image, label, noise_field