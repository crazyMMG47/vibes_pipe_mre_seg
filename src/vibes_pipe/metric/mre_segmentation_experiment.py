"""
MRE Segmentation Regional Metrics Evaluation
Notebook for evaluating Prob-UNet variants across anatomical regions
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# SEGMENTATION METRICS (from your provided code)
# ============================================================================

class DiceScore:
    """Compute Dice Score (F1 score for segmentation)."""
    
    def __init__(self, smooth: float = 1e-6, per_class: bool = True):
        self.smooth = smooth
        self.per_class = per_class
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, ...] predictions (binary or probabilities)
            target: [B, C, ...] ground truth
        Returns:
            dice: scalar or [C] if per_class=True
        """
        if pred.max() > 1.0:
            pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        pred_sum = pred_flat.sum(dim=2)
        target_sum = target_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        if self.per_class:
            return dice.mean(dim=0)
        else:
            return dice.mean()


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
            spacing: voxel spacing
        Returns:
            hausdorff distance (95th percentile)
        """
        pred_np = pred.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        if spacing is None:
            spacing = np.ones(pred_np.ndim)
        
        pred_dt = distance_transform_edt(~pred_np, sampling=spacing)
        target_dt = distance_transform_edt(~target_np, sampling=spacing)
        
        pred_surface = pred_np & ~distance_transform_edt(pred_np, sampling=spacing).astype(bool)
        target_surface = target_np & ~distance_transform_edt(target_np, sampling=spacing).astype(bool)
        
        if not pred_surface.any() or not target_surface.any():
            return float('inf')
        
        dist_pred_to_target = pred_dt[target_surface]
        dist_target_to_pred = target_dt[pred_surface]
        
        hd95 = max(
            np.percentile(dist_pred_to_target, self.percentile),
            np.percentile(dist_target_to_pred, self.percentile)
        )
        
        return float(hd95)


class SurfaceDice:
    """Normalized Surface Dice (boundary-focused metric)."""
    
    def __init__(self, tolerance: float = 1.0):
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
        
        pred_dt = distance_transform_edt(pred_np, sampling=spacing)
        target_dt = distance_transform_edt(target_np, sampling=spacing)
        
        pred_surface = pred_np & (pred_dt <= 1)
        target_surface = target_np & (target_dt <= 1)
        
        if not pred_surface.any() or not target_surface.any():
            return 0.0
        
        pred_dt_to_target = distance_transform_edt(~target_np, sampling=spacing)
        target_dt_to_pred = distance_transform_edt(~pred_np, sampling=spacing)
        
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
        
        if 'hausdorff' in self.metrics_to_compute or 'surface_dice' in self.metrics_to_compute:
            pred_binary = (torch.sigmoid(pred[0, 0]) > 0.5).cpu()
            target_binary = target[0, 0].cpu()
            
            if 'hausdorff' in self.metrics_to_compute:
                results['hausdorff95'] = self.hausdorff(pred_binary, target_binary, self.spacing)
            
            if 'surface_dice' in self.metrics_to_compute:
                results['surface_dice'] = self.surface_dice(pred_binary, target_binary, self.spacing)
        
        return results


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================

def calculate_ged(samples):
    """Calculate Generalized Energy Distance"""
    n_samples = samples.shape[0]
    
    if n_samples < 2:
        return 0.0
    
    samples_flat = samples.reshape(n_samples, -1)
    
    ged_sum = 0
    count = 0
    max_pairs = min(n_samples, 15)
    
    for i in range(max_pairs):
        for j in range(i+1, max_pairs):
            dist = np.sqrt(np.mean((samples_flat[i] - samples_flat[j]) ** 2))
            ged_sum += dist
            count += 1
    
    ged = ged_sum / max(count, 1)
    return ged


def calculate_sample_diversity(samples):
    """Calculate diversity among samples using average pairwise correlation."""
    n_samples = samples.shape[0]
    
    if n_samples < 2:
        return 0.0
    
    samples_flat = samples.reshape(n_samples, -1)
    
    correlations = []
    for i in range(min(n_samples, 10)):
        for j in range(i+1, min(n_samples, 10)):
            corr = np.corrcoef(samples_flat[i], samples_flat[j])[0, 1]
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    diversity = 1 - abs(avg_correlation)
    
    return diversity


def calculate_uncertainty_metrics(samples, mean_pred, std_pred):
    """
    Calculate comprehensive uncertainty metrics including GED
    
    Args:
        samples: MC samples [n_samples, batch, channels, H, W, D]
        mean_pred: Mean prediction [B, C, D, H, W]
        std_pred: Standard deviation [B, C, D, H, W]
        
    Returns:
        Dictionary of uncertainty metrics
    """
    eps = 1e-8
    
    samples_squeezed = samples[:, 0, 0]
    mean_squeezed = mean_pred[0, 0]
    std_squeezed = std_pred[0, 0]
    
    # 1. Predictive Entropy
    p = mean_squeezed
    predictive_entropy = -(p * np.log(p + eps) + (1-p) * np.log(1-p + eps))
    
    # 2. Mutual Information
    entropy_of_mean = -(p * np.log(p + eps) + (1-p) * np.log(1-p + eps))
    
    mean_entropy = 0
    for sample in samples_squeezed:
        sample_entropy = -(sample * np.log(sample + eps) + (1-sample) * np.log(1-sample + eps))
        mean_entropy += sample_entropy
    mean_entropy /= len(samples_squeezed)
    
    mutual_info = entropy_of_mean - mean_entropy
    
    # 3. Variance and others
    variance = np.var(samples_squeezed, axis=0)
    cv = std_squeezed / (mean_squeezed + eps)
    diversity = calculate_sample_diversity(samples_squeezed)
    ged = calculate_ged(samples_squeezed)
    
    return {
        'predictive_entropy': predictive_entropy,
        'mutual_information': mutual_info,
        'variance': variance,
        'std': std_squeezed,
        'ged': ged,
        'coefficient_variation': cv,
        'sample_diversity': diversity
    }


# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

class RegionalSegmentationAnalyzer:
    """Analyze segmentation metrics across anatomical regions."""
    
    def __init__(self, spacing: Optional[tuple] = None):
        self.spacing = spacing
        self.metrics = SegmentationMetrics(
            metrics=['dice', 'hausdorff'],
            spacing=spacing
        )
    
    def split_into_regions(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split 3D volume into frontal and central regions.
        
        Args:
            volume: [D, H, W] or [B, C, D, H, W]
            
        Returns:
            frontal: First 25% of slices
            central: Remaining 75% of slices
        """
        # Handle different dimensions
        if volume.dim() == 5:  # [B, C, D, H, W]
            d = volume.shape[2]
        else:  # [D, H, W]
            d = volume.shape[0]
        
        split_idx = int(np.ceil(d * 0.25))
        
        if volume.dim() == 5:
            frontal = volume[:, :, :split_idx, :, :]
            central = volume[:, :, split_idx:, :, :]
        else:
            frontal = volume[:split_idx, :, :]
            central = volume[split_idx:, :, :]
        
        return frontal, central
    
    def evaluate_region(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        region_name: str = "region"
    ) -> Dict[str, float]:
        """Evaluate metrics for a specific region."""
        results = self.metrics(pred, target)
        results['region'] = region_name
        return results
    
    def analyze_subject(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Dict]:
        """
        Comprehensive analysis of a subject across regions.
        
        Args:
            predictions: [B, C, D, H, W]
            targets: [B, C, D, H, W]
            
        Returns:
            Dictionary with 'global', 'frontal', 'central' results
        """
        results = {}
        
        # Global metrics
        results['global'] = self.evaluate_region(
            predictions, targets, region_name='global'
        )
        
        # Regional split
        pred_frontal, pred_central = self.split_into_regions(predictions)
        target_frontal, target_central = self.split_into_regions(targets)
        
        # Frontal metrics
        results['frontal'] = self.evaluate_region(
            pred_frontal, target_frontal, region_name='frontal'
        )
        
        # Central metrics
        results['central'] = self.evaluate_region(
            pred_central, target_central, region_name='central'
        )
        
        return results


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Orchestrate the complete evaluation pipeline."""
    
    def __init__(self, spacing: Optional[tuple] = None):
        self.analyzer = RegionalSegmentationAnalyzer(spacing=spacing)
        self.all_results = []
    
    def run_evaluation(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        subject_id: str
    ):
        """
        Run evaluation for all model variants.
        
        Args:
            predictions: Dict with keys like 'unet', 'prob_unet', 'prob_unet_freebits', 'prob_unet_noise'
                        Each value is [B, C, D, H, W]
            targets: [B, C, D, H, W]
            subject_id: Subject identifier
        """
        for model_name, pred in predictions.items():
            regional_results = self.analyzer.analyze_subject(pred, targets)
            
            for region, metrics in regional_results.items():
                row = {
                    'subject': subject_id,
                    'method': model_name,
                    'region': region,
                    **metrics
                }
                self.all_results.append(row)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return all results as a structured DataFrame."""
        return pd.DataFrame(self.all_results)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate the summary table matching your figure."""
        df = self.get_results_dataframe()
        
        # Pivot to get methods vs metrics
        summary_parts = []
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            row = {'Method': method}
            for region in ['global', 'frontal', 'central']:
                region_data = method_data[method_data['region'] == region]
                if not region_data.empty:
                    row[f'{region}_dice'] = region_data['dice'].mean()
                    row[f'{region}_hd95'] = region_data['hausdorff95'].mean()
            
            # Calculate performance drop
            global_dice = row.get('global_dice', 0)
            frontal_dice = row.get('frontal_dice', 0)
            row['perf_drop'] = global_dice - frontal_dice
            
            summary_parts.append(row)
        
        return pd.DataFrame(summary_parts)

