# Contains builder functions that is able to retrieve config info from yaml 
from src.vibes_pipe.augmentation.basic_augment import SpatialAugmenter
from src.vibes_pipe.augmentation.noise_augment import NoiseAugmenter
from src.vibes_pipe.augmentation.augment_pipeline import MREAugmentation


def build_spatial_augmenter(cfg):
    aug_cfg = cfg.get("augmentation", {})
    s = aug_cfg.get("spatial", {})
    if not s:
        return None
    return SpatialAugmenter(
        rotation_range=tuple(s.get("rotation_range", [-10, 10])),
        scale_range=tuple(s.get("scale_range", [0.95, 1.05])),
        is_2d=s.get("is_2d", False),
    )
    
    
def build_noise_augmenter(cfg):
    aug_cfg = cfg.get("augmentation", {})
    n = aug_cfg.get("noise", {})

    if not n.get("enabled", False):
        return None

    return NoiseAugmenter(
        noise_strength_range=tuple(n.get("strength_range", [0.05, 0.15]))
    )
    
    
def build_train_augmenter(cfg):
    spatial_augmenter = build_spatial_augmenter(cfg)
    noise_augmenter = build_noise_augmenter(cfg)

    if spatial_augmenter is None and noise_augmenter is None:
        return None

    return MREAugmentation(
        spatial_augmenter=spatial_augmenter,
        noise_augmenter=noise_augmenter,
        apply_prob=cfg.get("noise_augment", {}).get("apply_prob", 0.8),
    )
    
    