import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, Optional


class DiceScore:
    """Compute Dice Score (F1 score for segmentation)."""
    
    def __init__(self, smooth: float = 1e-6, per_class: bool = True):
        self.smooth = smooth
        self.per_class = per_class
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, ...] predictions (binary or probabilities)
            target: [B, C, ...] ground truth
        Returns:
            dice: scalar or [C] if per_class=True
        """
        # Ensure binary
        if pred.max() > 1.0:
            pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        # Flatten spatial dimensions
        pred_flat = pred.flatten(2)      # [B, C, N]
        target_flat = target.flatten(2)  # [B, C, N]
        
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        pred_sum = pred_flat.sum(dim=2)                      # [B, C]
        target_sum = target_flat.sum(dim=2)                  # [B, C]
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        if self.per_class:
            return dice.mean(dim=0)  # [C]
        else:
            return dice.mean()  # scalar


class IoU:
    """Intersection over Union (Jaccard Index)."""
    
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.max() > 1.0:
            pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean()


class HausdorffDistance:
    """Compute Hausdorff Distance (95th percentile) between segmentations."""
    
    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: Optional[tuple] = None
    ) -> float:
        """
        Args:
            pred: [H, W] or [D, H, W] binary prediction
            target: [H, W] or [D, H, W] binary ground truth
            spacing: voxel spacing (e.g., (1.5, 1.5, 1.5))
        Returns:
            hausdorff distance (95th percentile)
        """
        pred_np = pred.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        if spacing is None:
            spacing = np.ones(pred_np.ndim)
        
        # Compute distance transforms
        pred_dt = distance_transform_edt(~pred_np, sampling=spacing)
        target_dt = distance_transform_edt(~target_np, sampling=spacing)
        
        # Distances from pred surface to target
        pred_surface = pred_np & ~distance_transform_edt(pred_np, sampling=spacing).astype(bool)
        target_surface = target_np & ~distance_transform_edt(target_np, sampling=spacing).astype(bool)
        
        if not pred_surface.any() or not target_surface.any():
            return float('inf')
        
        # Directed Hausdorff distances
        dist_pred_to_target = pred_dt[target_surface]
        dist_target_to_pred = target_dt[pred_surface]
        
        # 95th percentile
        hd95 = max(
            np.percentile(dist_pred_to_target, self.percentile),
            np.percentile(dist_target_to_pred, self.percentile)
        )
        
        return float(hd95)


class SurfaceDice:
    """Normalized Surface Dice (boundary-focused metric)."""
    
    def __init__(self, tolerance: float = 1.0):
        """
        Args:
            tolerance: surface distance tolerance in voxels
        """
        self.tolerance = tolerance
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: Optional[tuple] = None
    ) -> float:
        """Compute normalized surface Dice."""
        pred_np = pred.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        if spacing is None:
            spacing = np.ones(pred_np.ndim)
        
        # Compute surfaces
        pred_dt = distance_transform_edt(pred_np, sampling=spacing)
        target_dt = distance_transform_edt(target_np, sampling=spacing)
        
        pred_surface = pred_np & (pred_dt <= 1)
        target_surface = target_np & (target_dt <= 1)
        
        if not pred_surface.any() or not target_surface.any():
            return 0.0
        
        # Distance from each surface to the other
        pred_dt_to_target = distance_transform_edt(~target_np, sampling=spacing)
        target_dt_to_pred = distance_transform_edt(~pred_np, sampling=spacing)
        
        # Count surface points within tolerance
        pred_close = (pred_dt_to_target[pred_surface] <= self.tolerance).sum()
        target_close = (target_dt_to_pred[target_surface] <= self.tolerance).sum()
        
        nsd = (pred_close + target_close) / (pred_surface.sum() + target_surface.sum())
        return float(nsd)


class SegmentationMetrics:
    """Comprehensive evaluation metrics for segmentation."""
    
    def __init__(
        self,
        metrics: list = ['dice', 'iou', 'hausdorff', 'surface_dice'],
        spacing: Optional[tuple] = None
    ):
        self.spacing = spacing
        self.metrics_to_compute = metrics
        
        self.dice = DiceScore(per_class=False)
        self.iou = IoU()
        self.hausdorff = HausdorffDistance()
        self.surface_dice = SurfaceDice()
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.
        
        Args:
            pred: [B, C, D, H, W] or [B, C, H, W] predictions
            target: [B, C, D, H, W] or [B, C, H, W] ground truth
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        if 'dice' in self.metrics_to_compute:
            results['dice'] = float(self.dice(pred, target).item())
        
        if 'iou' in self.metrics_to_compute:
            results['iou'] = float(self.iou(pred, target).item())
        
        # For HD and Surface Dice, process first sample only (expensive)
        if 'hausdorff' in self.metrics_to_compute or 'surface_dice' in self.metrics_to_compute:
            pred_binary = (torch.sigmoid(pred[0, 0]) > 0.5).cpu()
            target_binary = target[0, 0].cpu()
            
            if 'hausdorff' in self.metrics_to_compute:
                results['hausdorff95'] = self.hausdorff(pred_binary, target_binary, self.spacing)
            
            if 'surface_dice' in self.metrics_to_compute:
                results['surface_dice'] = self.surface_dice(pred_binary, target_binary, self.spacing)
        
        return results