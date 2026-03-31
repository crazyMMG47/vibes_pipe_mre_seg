import torch
from typing import Tuple, Dict, Any, Optional


class KLDivergence:
    def __init__(self, free_bits: float = 0.0, epsilon: float = 1e-12, logvar_clip: float = 12.0, eduction: str = "mean"):
        self.free_bits = float(free_bits)     # nats per-dim
        self.epsilon = epsilon
        self.logvar_clip = float(logvar_clip)

    def __call__(self, mu_q, logvar_q, mu_p, logvar_p, return_per_dim: bool = True):
        # stabilize
        logvar_q = torch.clamp(logvar_q, -self.logvar_clip, self.logvar_clip)
        logvar_p = torch.clamp(logvar_p, -self.logvar_clip, self.logvar_clip)
        var_q = torch.exp(logvar_q) + self.epsilon
        var_p = torch.exp(logvar_p) + self.epsilon

        # per-dim KL(q||p)
        kl_per_dim = 0.5 * (
            (logvar_p - logvar_q)
            + (var_q + (mu_q - mu_p) ** 2) / var_p
            - 1.0
        )
        
        if self.free_bits > 0:
            kl_per_dim = torch.relu(kl_per_dim - self.free_bits)

        if self.reduction == "mean":
            kl_per_slice = kl_per_dim.mean(dim=-1)   # [B,D] or [B]
        else:
            kl_per_slice = kl_per_dim.sum(dim=-1)

        return (kl_per_slice, kl_per_dim) if return_per_dim else (kl_per_slice, None)

    @staticmethod
    def standard_normal_kl(mu, logvar, logvar_clip: float = 12.0, reduction: str = "mean"):
        logvar = torch.clamp(logvar, -logvar_clip, logvar_clip)
        per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return per_dim.mean(dim=-1) if reduction == "mean" else per_dim.sum(dim=-1)


class KLMonitor:
    """Monitor KL divergence and active dimensions during training."""
    
    def __init__(self, active_threshold: float = 0.01):
        """
        Args:
            active_threshold: Minimum KL (in nats) for a dimension to be considered active
        """
        self.active_threshold = active_threshold
        self.kl_computer = KLDivergence()
    
    def compute_metrics(
        self,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute comprehensive KL metrics including active dimensions.
        
        Returns:
            Dictionary with KL statistics and active dimension info
        """
        kl_per_slice, kl_per_dim = self.kl_computer(
            mu_q, logvar_q, mu_p, logvar_p, return_per_dim=True
        )
        
        with torch.no_grad():
            # Active dimensions (per-dim KL averaged over B and D)
            kl_per_latent = kl_per_dim.mean(dim=tuple(range(kl_per_dim.ndim - 1)))  # [Z]
            active_dims = (kl_per_latent > self.active_threshold).sum().item()
            
            # Distribution statistics
            posterior_std = torch.exp(0.5 * logvar_q).mean().item()
            prior_std = torch.exp(0.5 * logvar_p).mean().item()
            
            metrics = {
                "kl/total": float(kl_per_slice.mean().item()),
                "kl/min": float(kl_per_slice.min().item()),
                "kl/max": float(kl_per_slice.max().item()),
                "kl/std": float(kl_per_slice.std().item()),
                
                "kl/active_dims": active_dims,
                "kl/total_dims": kl_per_dim.shape[-1],
                "kl/active_ratio": active_dims / kl_per_dim.shape[-1],
                
                "kl/per_dim_mean": float(kl_per_dim.mean().item()),
                "kl/per_dim_min": float(kl_per_dim.min().item()),
                "kl/per_dim_max": float(kl_per_dim.max().item()),
                
                "dist/posterior_std": posterior_std,
                "dist/prior_std": prior_std,
                "dist/mu_post_mean": float(mu_q.mean().item()),
                "dist/mu_prior_mean": float(mu_p.mean().item()),
            }
        
        return metrics