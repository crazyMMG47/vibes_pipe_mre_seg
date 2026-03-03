import numpy as np 
import Tuple, Dict


def compute_siemens_noise(
    img: np.ndarray,
    spatial_dims: int = 3,
    ref_dim: int = None,
    ref_idx: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Compute noise profiles from Siemens MRE raw data.
    Properly handles COMPLEX data!
    """
    if img.ndim < spatial_dims:
        raise ValueError(f"Expected ≥{spatial_dims} dimensions, got {img.shape}")
    
    print(f"Processing Siemens data with shape: {img.shape}")
    print(f"Data type: {img.dtype} (complex: {np.iscomplexobj(img)})")
    
    # CRITICAL: Take magnitude FIRST if data is complex
    if np.iscomplexobj(img):
        print("⚠️  Data is complex - taking magnitude first")
        img = np.abs(img)
    
    # Calculate mean over all non-spatial dimensions
    non_spatial_axes = tuple(range(spatial_dims, img.ndim))
    t2stack = img.mean(axis=non_spatial_axes) if non_spatial_axes else img
    t2stack = t2stack.astype(np.float32)
    t2_mean = np.mean(t2stack)
    
    print(f"T2 Stack: mean={t2_mean:.1f}, std={np.std(t2stack):.1f}")
    
    # Auto-detect best reference if not specified
    if ref_dim is None or ref_idx is None:
        print("\nAuto-detecting best reference frame...")
        
        best_score = float('inf')
        best_dim = None
        best_idx = None
        best_frame = None
        
        for dim in range(spatial_dims, img.ndim):
            for idx in range(img.shape[dim]):
                frame = np.take(img, idx, axis=dim)
                extra = tuple(range(spatial_dims, frame.ndim))
                if extra:
                    frame = frame.mean(axis=extra)
                
                frame = frame.astype(np.float32)  # Already magnitude from line 23
                noise = t2stack - frame
                noise_mean = np.mean(noise)
                
                print(f"  Dim {dim}, Idx {idx}: noise_mean={noise_mean:.2f}")
                
                # Score: lower absolute value is better
                score = abs(noise_mean)
                
                if score < best_score:
                    best_score = score
                    best_dim = dim
                    best_idx = idx
                    best_frame = frame
                    best_noise_mean = noise_mean
        
        ref_dim = best_dim
        ref_idx = best_idx
        ref_frame = best_frame
        
        print(f"\n✓ Selected: Dimension {ref_dim}, Index {ref_idx}")
        print(f"  Noise mean: {best_noise_mean:.2f} (closest to zero)")
    else:
        print(f"Using specified reference: Dimension {ref_dim}, Index {ref_idx}")
        ref_frame = np.take(img, ref_idx, axis=ref_dim)
        extra_axes = tuple(range(spatial_dims, ref_frame.ndim))
        if extra_axes:
            ref_frame = ref_frame.mean(axis=extra_axes)
        ref_frame = ref_frame.astype(np.float32)
    
    print(f"Reference frame: mean={np.mean(ref_frame):.1f}, std={np.std(ref_frame):.1f}")
    
    # Compute noise maps
    noise = t2stack - ref_frame
    noise_scaled = noise * 1000.0
    t2noise = t2stack + noise
    
    # Calculate noise statistics
    valid_noise = noise[np.isfinite(noise)]
    noise_mean = np.mean(valid_noise) if valid_noise.size > 0 else np.nan
    noise_std = np.std(valid_noise) if valid_noise.size > 0 else np.nan
    
    print(f"\nFinal noise statistics: μ={noise_mean:.2f}, σ={noise_std:.2f}")
    
    if abs(noise_mean) > 10:
        print(f"⚠️  WARNING: Noise mean is {noise_mean:.2f}, should be close to 0!")
        print(f"   Consider trying a different reference frame.")
    else:
        print(f"✓ Noise looks good (mean ≈ 0)")
    
    metadata = {
        "original_shape": img.shape,
        "ref_dim": ref_dim,
        "ref_idx": ref_idx,
        "t2stack_shape": t2stack.shape,
        "ref_frame_shape": ref_frame.shape,
        "noise_mean": float(noise_mean),
        "noise_std": float(noise_std),
    }
    
    return t2stack, noise, noise_scaled, t2noise, metadata