from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceComboLoss(nn.Module):
    def __init__(
        self,
        alpha_combo: float = 0.20,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha_combo = alpha_combo
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        spatial_dims = tuple(range(2, logits.ndim))
        intersection = (probs * targets).sum(dim=spatial_dims)
        union = probs.sum(dim=spatial_dims) + targets.sum(dim=spatial_dims)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating_factor = (1.0 - p_t).pow(self.focal_gamma)
        alpha_weight = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1 - targets)
        return (alpha_weight * modulating_factor * bce_loss).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        loss_focal = self.focal_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.alpha_combo * loss_focal + (1.0 - self.alpha_combo) * loss_dice
    
    
class ProbUNetLoss(nn.Module):
    def __init__(
        self,
        recon_loss: nn.Module,
        beta_final: float = 0.1,
        beta_warmup: int = 20,
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.recon_loss = recon_loss
        self.beta_final = beta_final
        self.beta_warmup = beta_warmup
        self.free_bits = free_bits

    def get_beta(self, epoch: int) -> float:
        return self.beta_final * min(1.0, epoch / max(self.beta_warmup, 1))

    def forward(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        beta = self.get_beta(epoch)

        is_deterministic = hasattr(model, "inject_latent") and not model.inject_latent

        if is_deterministic:
            logits = model(x)
            loss_recon = self.recon_loss(logits, y)
            loss_kl = torch.tensor(0.0, device=x.device)
            total_loss = loss_recon
        else:
            logits, (mu_prior, logvar_prior), (mu_post, logvar_post) = model(x, y)

            loss_recon = self.recon_loss(logits, y)

            kl_per_dim = -0.5 * (
                1 + logvar_post - logvar_prior
                - (logvar_post.exp() + (mu_post - mu_prior).pow(2)) / logvar_prior.exp()
            )
            kl_meanZ = kl_per_dim.mean(dim=1)

            if self.free_bits > 0:
                kl_meanZ = torch.clamp(kl_meanZ - self.free_bits, min=0.0)

            loss_kl = kl_meanZ.mean()
            total_loss = loss_recon + beta * loss_kl

        return {
            "loss": total_loss,
            "recon": loss_recon,
            "kl": loss_kl,
            "beta": torch.tensor(beta, device=x.device),
        }