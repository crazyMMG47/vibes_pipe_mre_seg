class Preprocessor:
    """
    Simple preprocessor for medical images.
    """
    
    def __init__(self, target_spacing=(1.5, 1.5, 1.5), target_size=(128, 128, 64), is_2d=False):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.is_2d = is_2d
    
    def load_nifti(self, filepath):
        """Load NIfTI file"""
        nii = nib.load(filepath)
        data = nii.get_fdata().astype(np.float32)
        spacing = nii.header.get_zooms()[:3]
        return data, spacing
    
    def resample(self, volume, original_spacing, target_spacing, order=1):
        """Resample to target spacing"""
        zoom_factors = [orig/target for orig, target in zip(original_spacing, target_spacing)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
    
    def resize(self, volume, target_size, order=1):
        """Resize to target size"""
        zoom_factors = [target/current for target, current in zip(target_size, volume.shape)]
        return ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
    
    def apply_clahe(self, volume):
        """Applies 3D CLAHE slice by slice."""
        # CLAHE is 2D, so apply it to each slice (e.g., in the Z-dimension)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Slices are (H, W, D). We iterate over D (axis 2).
        # We must scale to 0-255 (uint8) for OpenCV and then scale back.
        
        # 1. Clip and scale to 0-255
        p_low, p_high = np.percentile(volume, (1, 99))
        volume_clipped = np.clip(volume, p_low, p_high)
        volume_norm_8bit = (255 * (volume_clipped - p_low) / (p_high - p_low + 1e-8)).astype(np.uint8)

        # 2. Apply CLAHE slice-by-slice
        slices_clahe = []
        for i in range(volume_norm_8bit.shape[2]):
            slice_clahe = clahe_obj.apply(volume_norm_8bit[:, :, i])
            slices_clahe.append(slice_clahe)
        
        volume_clahe_8bit = np.stack(slices_clahe, axis=2)
        
        # 3. Scale back to original float range (approximately)
        volume_clahe_float = (volume_clahe_8bit.astype(np.float32) / 255.0) * (p_high - p_low) + p_low
        return volume_clahe_float
    
    def normalize(self, volume):
        """Z-score normalization"""
        valid_voxels = volume[volume > 0]
        if valid_voxels.size == 0:
            return volume
        mean = np.mean(valid_voxels)
        std = np.std(valid_voxels)
        return (volume - mean) / (std + 1e-8)
    
    def process_pair(self, image_path, label_path):
        """Process image-label pair"""
        # Load
        image, img_spacing = self.load_nifti(image_path)
        label, lbl_spacing = self.load_nifti(label_path)
        
        original_shape = image.shape
        
        image = self.apply_clahe(image)
        
        # Resample to target spacing
        image = self.resample(image, img_spacing, self.target_spacing, order=1)
        label = self.resample(label, lbl_spacing, self.target_spacing, order=0)
        
        # Match label size to image
        if label.shape != image.shape:
            label = self.resize(label, image.shape, order=0)
        
        # Resize to target size
        if self.target_size:
            image = self.resize(image, self.target_size, order=1)
            label = self.resize(label, self.target_size, order=0)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize label
        label = (label > 0.5).astype(np.float32)
        
        return image, label, original_shape

