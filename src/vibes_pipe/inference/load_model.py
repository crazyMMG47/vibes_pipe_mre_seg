# Run inference AND incorporate functions to plot 
# ... the hub to run inference, save results, and plot figures
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import warnings
warnings.filterwarnings('ignore')

# import other modules
from models.metric.uncertainty_metric import calculate_uncertainty_metrics
from models.inference.monte_carlo_sampling import monte_carlo_sampling

class InferenceWithDataLoader:
    """
    Complete pipeline to run inference with uncertainty quantification. 
    
    """
    
    def __init__(self, model, test_loader, device='cuda', save_dir='inference_results'):
        """
        Initialize inference pipeline
        
        Args:
            model: Trained model
            test_loader: DataLoader for test set
            device: Device for computation
            save_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for results
        self.predictions = []
        self.uncertainties = []
        self.ground_truths = []
        self.latent_samples = []
        self.metrics = []
        self.prediction_samples = []
        
        print(f"Initialized inference pipeline")
        print(f"Device: {self.device}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Results will be saved to: {self.save_dir}")
        
        # Reset to eval mode
        self.model.eval()
        
        # Stack samples
        samples = np.stack(samples)  # [n_samples, batch, channels, H, W, D]
        
        # Calculate statistics
        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self.calculate_uncertainty_metrics(samples, mean_pred, std_pred)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': samples,
            'uncertainty': uncertainty_metrics,
        }
    
    
    def run_inference(self, n_mc_samples=5, window_size=(96, 96, 48), 
                     calculate_metrics=True, save_predictions=True):
        """
        Run inference on entire test set with uncertainty quantification
        
        Args:
            n_mc_samples: Number of Monte Carlo samples
            window_size: Sliding window size
            calculate_metrics: Whether to calculate Dice and other metrics
            save_predictions: Whether to save predictions to disk
            
        Returns:
            Summary of results
        """
        print(f"\nStarting inference with {n_mc_samples} MC samples...")
        print("="*60)
        
        # Metrics calculators
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        post_pred = AsDiscrete(argmax=False, threshold=0.5)
        post_label = AsDiscrete(argmax=False, threshold=0.5)
        
        # Progress bar
        pbar = tqdm(self.test_loader, desc="Processing batches")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Get data
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device) if 'label' in batch_data else None
            else:
                images = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device) if len(batch_data) > 1 else None
            
            # Monte Carlo sampling
            mc_results = self.monte_carlo_sampling(
                images, 
                n_samples=n_mc_samples,
                window_size=window_size
            )
            
            # Store results
            self.predictions.append(mc_results['mean'])             # [B,C,H,W,D] averaged
            self.prediction_samples.append(mc_results['samples'])   # [K,B,C,H,W,D] 
            self.uncertainty_maps.append(mc_results['uncertainty']) 
                        
            if labels is not None:
                self.ground_truths.append(labels.cpu().numpy())
            
            if mc_results['latent'] is not None:
                self.latent_samples.extend(mc_results['latent'])
            
            # Calculate metrics if ground truth available
            if calculate_metrics and labels is not None:
                # Binarize predictions
                pred_binary = post_pred(torch.from_numpy(mc_results['mean']))
                label_binary = post_label(labels.cpu())
                
                # Dice score
                dice_metric(pred_binary, label_binary)
                dice_score = dice_metric.aggregate().item()
                dice_metric.reset()
                
                # Store metrics
                batch_metrics = {
                    'batch_idx': batch_idx,
                    'dice': dice_score,
                    'mean_entropy': np.mean(mc_results['uncertainty']['predictive_entropy']),
                    'mean_std': np.mean(mc_results['uncertainty']['std']),
                    'ged': mc_results['uncertainty']['ged'],
                    'sample_diversity': mc_results['uncertainty']['sample_diversity']
                }
                self.metrics.append(batch_metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'Dice': f"{dice_score:.4f}",
                    'GED': f"{mc_results['uncertainty']['ged']:.4f}"
                })
            
            # Save individual predictions if requested
            if save_predictions:
                np.save(self.save_dir / f"prediction_{batch_idx:04d}.npy", mc_results['mean'])
                np.save(self.save_dir / f"uncertainty_{batch_idx:04d}.npy", 
                       mc_results['uncertainty']['predictive_entropy'])
        
        print("\nInference completed!")
        
        # Generate summary
        summary = self.generate_summary()
        
        return summary
    
    def generate_summary(self):
        """
        Generate summary statistics of inference results
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Basic info
        summary['n_samples'] = len(self.predictions)
        summary['timestamp'] = datetime.now().isoformat()
        
        # Metrics summary if available
        if self.metrics:
            dice_scores = [m['dice'] for m in self.metrics]
            ged_scores = [m['ged'] for m in self.metrics]
            entropy_scores = [m['mean_entropy'] for m in self.metrics]
            diversity_scores = [m['sample_diversity'] for m in self.metrics]
            
            summary['metrics'] = {
                'dice': {
                    'mean': float(np.mean(dice_scores)),
                    'std': float(np.std(dice_scores)),
                    'min': float(np.min(dice_scores)),
                    'max': float(np.max(dice_scores))
                },
                'ged': {
                    'mean': float(np.mean(ged_scores)),
                    'std': float(np.std(ged_scores)),
                    'min': float(np.min(ged_scores)),
                    'max': float(np.max(ged_scores))
                },
                'entropy': {
                    'mean': float(np.mean(entropy_scores)),
                    'std': float(np.std(entropy_scores))
                },
                'sample_diversity': {
                    'mean': float(np.mean(diversity_scores)),
                    'std': float(np.std(diversity_scores))
                }
            }
        
        # Save summary
        with open(self.save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"Total samples processed: {summary['n_samples']}")
        
        if 'metrics' in summary:
            print(f"\nDice Score: {summary['metrics']['dice']['mean']:.4f} ± {summary['metrics']['dice']['std']:.4f}")
            print(f"GED: {summary['metrics']['ged']['mean']:.4f} ± {summary['metrics']['ged']['std']:.4f}")
            print(f"Mean Entropy: {summary['metrics']['entropy']['mean']:.4f} ± {summary['metrics']['entropy']['std']:.4f}")
            print(f"Sample Diversity: {summary['metrics']['sample_diversity']['mean']:.4f} ± {summary['metrics']['sample_diversity']['std']:.4f}")
        
        print(f"\nResults saved to: {self.save_dir}")
        print("="*60)
        
        return summary
    
