import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Literal


class DicePerSlice:
    """Compute Dice score for each slice independently."""
    
    def __init__(self, smooth: float = 1e-6, prob_threshold: float = 0.5):
        self.smooth = smooth
        self.prob_threshold = prob_threshold
    
    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [B, C, D H, W] predictions (logits or probabilities)
            target: [B, C, D, H, W] ground truth
        Returns:
            dice_scores: [B, D] Dice score per slice
        """
        B, C, D, H, W = preds.shape
        
        # Ensure binary predictions
        if preds.max() > 1.0:  # Assume logits
            preds = torch.sigmoid(preds)
        preds = (preds > self.prob_threshold).float()
        
        # Flatten spatial dimensions per slice
        preds_flat = preds.view(B, C, D, -1)      # [B, C, D, H*W]
        target_flat = target.view(B, C, D, -1)    # [B, C, D, H*W]
        
        # Compute intersection and sums per slice
        intersection = (preds_flat * target_flat).sum(dim=-1)  # [B, C, D]
        preds_sum = preds_flat.sum(dim=-1)                     # [B, C, D]
        target_sum = target_flat.sum(dim=-1)                   # [B, C, D]
        
        # Dice formula
        dice = (2.0 * intersection + self.smooth) / (preds_sum + target_sum + self.smooth)
        
        # Average over channels
        dice = dice.mean(dim=1)  # [B, D]
        
        return dice


class UncertaintyAwareSliceKL:
    """
    Reweights KL per slice based on segmentation difficulty.
    Uses Dice score to identify hard slices and upweight their KL contribution.
    """
    
    def __init__(
        self,
        worst_slice_percentage: float = 0.20,
        w_hard: float = 0.45,
        w_other: float = 1.0,
        mode: Literal["slice", "volume"] = "slice",
        prob_threshold: float = 0.5,
        free_bits: float = 0.02,  # nats per latent dimension
    ):
        assert 0.0 < worst_slice_percentage <= 1.0, "worst_slice_percentage must be in (0, 1]"
        assert w_hard <= w_other, "Hard slices should have equal or lower weight"
        
        self.worst_slice_percentage = worst_slice_percentage
        self.w_hard = float(w_hard)
        self.w_other = float(w_other)
        self.mode = mode
        self.free_bits = float(free_bits)
        
        self.dice_metric = DicePerSlice(prob_threshold=prob_threshold)
    
    def __call__(
        self,
        logits: torch.Tensor,        # [B, C, D, H, W]
        target: torch.Tensor,        # [B, C, D, H, W]
        mu_prior: torch.Tensor,      # [B, D, Z]
        logvar_prior: torch.Tensor,  # [B, D, Z]
        mu_post: torch.Tensor,       # [B, D, Z]
        logvar_post: torch.Tensor,   # [B, D, Z]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute uncertainty-aware KL loss with diagnostics.
        
        Returns:
            loss: Weighted KL loss scalar
            diagnostics: Dictionary of metrics
        """
        from .kl_metrics import KLDivergence
        
        B, C, D, H, W = logits.shape
        device = logits.device
        
        # Validate shapes
        assert mu_post.shape == (B, D, mu_post.shape[-1]), \
            f"Expected mu_post shape [B, D, Z], got {mu_post.shape}"
        
        # =====================================================================
        # 1. Compute per-slice difficulty (lower Dice = harder)
        # =====================================================================
        with torch.no_grad():
            dice = self.dice_metric(logits, target)  # [B, D]
        
        # =====================================================================
        # 2. Build per-slice weights based on difficulty
        # =====================================================================
        n_worst = max(1, int(round(D * self.worst_slice_percentage)))
        worst_idx = dice.argsort(dim=1, descending=False)[:, :n_worst]  # [B, n_worst]
        
        weights = torch.full((B, D), self.w_other, device=device, dtype=torch.float32)
        weights.scatter_(1, worst_idx, self.w_hard)
        
        # =====================================================================
        # 3. Compute KL divergence with free-bits
        # =====================================================================
        kl_computer = KLDivergence(free_bits=self.free_bits)
        kl_per_slice, kl_per_dim = kl_computer(
            mu_post, logvar_post,
            mu_prior, logvar_prior
        )  # [B, D], [B, D, Z]
        
        # =====================================================================
        # 4. Apply weights
        # =====================================================================
        if self.mode == "slice":
            weighted_kl = kl_per_slice * weights  # [B, D]
            loss = weighted_kl.mean()
            volume_weights = weights.mean(dim=1)  # [B]
        elif self.mode == "volume":
            volume_weights = weights.mean(dim=1)  # [B]
            volume_kl = kl_per_slice.mean(dim=1)  # [B]
            loss = (volume_weights * volume_kl).mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # =====================================================================
        # 5. Compute diagnostics
        # =====================================================================
        diagnostics = self._compute_diagnostics(
            loss, kl_per_slice, kl_per_dim, dice, weights
        )
        
        return loss, diagnostics
    
    def _compute_diagnostics(
        self,
        loss: torch.Tensor,
        kl_per_slice: torch.Tensor,
        kl_per_dim: torch.Tensor,
        dice: torch.Tensor,
        weights: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute detailed diagnostics for monitoring."""
        with torch.no_grad():
            hard_mask = (weights < self.w_other)
            easy_mask = ~hard_mask
            
            # KL statistics
            kl_hard = kl_per_slice[hard_mask].mean() if hard_mask.any() else 0.0
            kl_easy = kl_per_slice[easy_mask].mean() if easy_mask.any() else 0.0
            
            # Dice statistics
            dice_hard = dice[hard_mask].mean() if hard_mask.any() else 0.0
            dice_easy = dice[easy_mask].mean() if easy_mask.any() else 0.0
            
            # Correlation between difficulty and KL
            difficulty = (1.0 - dice).flatten()
            kl_flat = kl_per_slice.flatten()
            x_c = difficulty - difficulty.mean()
            y_c = kl_flat - kl_flat.mean()
            correlation = (x_c * y_c).sum() / (
                x_c.pow(2).sum().sqrt() * y_c.pow(2).sum().sqrt() + 1e-12
            )
            
            diagnostics = {
                # Loss
                "train/kl_weighted": float(loss.item()),
                "train/kl_unweighted": float(kl_per_slice.mean().item()),
                
                # KL statistics
                "train/kl_min": float(kl_per_slice.min().item()),
                "train/kl_median": float(kl_per_slice.median().item()),
                "train/kl_max": float(kl_per_slice.max().item()),
                "train/kl_mean": float(kl_per_slice.mean().item()),
                "train/kl_std": float(kl_per_slice.std().item()),
                
                # Per-dimension KL
                "train/kl_dim_mean": float(kl_per_dim.mean().item()),
                "train/kl_dim_min": float(kl_per_dim.min().item()),
                "train/kl_dim_max": float(kl_per_dim.max().item()),
                "train/kl_dim_std": float(kl_per_dim.std().item()),
                
                # Hard vs Easy slices
                "train/kl_hard": float(kl_hard),
                "train/kl_easy": float(kl_easy),
                "train/dice_hard": float(dice_hard),
                "train/dice_easy": float(dice_easy),
                
                # Weights
                "train/weight_hard": self.w_hard,
                "train/weight_other": self.w_other,
                "train/hard_slice_pct": 100.0 * self.worst_slice_percentage,
                
                # Dice statistics
                "train/dice_min": float(dice.min().item()),
                "train/dice_median": float(dice.median().item()),
                "train/dice_max": float(dice.max().item()),
                "train/dice_mean": float(dice.mean().item()),
                
                # Correlation
                "train/corr_difficulty_kl": float(correlation),
                
                # Config
                "train/free_bits": self.free_bits,
                "train/mode": self.mode,
            }
        
        return diagnostics