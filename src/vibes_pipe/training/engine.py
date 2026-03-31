from __future__ import annotations
from typing import Optional, Dict
import json
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from monai.metrics import DiceMetric


class TrainEngine:
    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer,
        device: str = "cpu",
        num_epochs: int = 100,
        grad_clip: Optional[float] = 1.0,
        fast_val: bool = True,
        log_every: int = 1,
        save_path: str = "best_model.pt",
        final_save_path: str = "last_model.pt",
        history_save_path: str = "train_history.json",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.fast_val = fast_val
        self.log_every = log_every

        self.save_path = save_path
        self.final_save_path = final_save_path
        self.history_save_path = history_save_path

        self.scaler = GradScaler()
        self.val_dice_metric = DiceMetric(include_background=True, reduction="mean")

    def _prepare_save_dirs(self):
        for path in [self.save_path, self.final_save_path, self.history_save_path]:
            save_dir = os.path.dirname(path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

    def _save_checkpoint(self, path: str, epoch: int, val_dice: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": val_dice,
            },
            path,
        )

    def fit(self, train_loader, val_loader) -> Dict[str, list]:
        self.model.to(self.device)
        self._prepare_save_dirs()

        history = {
            "train/loss_total": [],
            "train/loss_recon": [],
            "train/loss_kl": [],
            "val/dice": [],
            "val/loss": [],
            "train/beta": [],
            "train/lr": [],
        }

        best_val_dice = -1.0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            epoch_metrics = {
                "loss": 0.0,
                "recon": 0.0,
                "kl": 0.0,
                "n": 0,
            }

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
            )

            last_beta = 0.0

            for batch in train_pbar:
                x = batch["image"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True).float()

                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    loss_dict = self.criterion(self.model, x, y, epoch)
                    total_loss = loss_dict["loss"]
                    loss_recon = loss_dict["recon"]
                    loss_kl = loss_dict["kl"]
                    last_beta = float(loss_dict["beta"])

                self.scaler.scale(total_loss).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_metrics["loss"] += total_loss.item()
                epoch_metrics["recon"] += loss_recon.item()
                epoch_metrics["kl"] += loss_kl.item()
                epoch_metrics["n"] += 1

                train_pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "recon": f"{loss_recon.item():.4f}",
                        "kl": f"{loss_kl.item():.4f}",
                    }
                )

            n = max(epoch_metrics["n"], 1)

            history["train/loss_total"].append(epoch_metrics["loss"] / n)
            history["train/loss_recon"].append(epoch_metrics["recon"] / n)
            history["train/loss_kl"].append(epoch_metrics["kl"] / n)
            history["train/beta"].append(last_beta)
            history["train/lr"].append(self.optimizer.param_groups[0]["lr"])

            avg_val_loss, avg_val_dice = self.validate(val_loader)
            history["val/loss"].append(avg_val_loss)
            history["val/dice"].append(avg_val_dice)

            if epoch % self.log_every == 0:
                print(
                    f"[Epoch {epoch:03d}/{self.num_epochs}] "
                    f"L:{history['train/loss_total'][-1]:.4f} "
                    f"Recon:{history['train/loss_recon'][-1]:.4f} "
                    f"KL:{history['train/loss_kl'][-1]:.4f} | "
                    f"Val Loss:{avg_val_loss:.4f} "
                    f"Val Dice:{avg_val_dice:.4f}"
                )

            # Save best checkpoint
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                self._save_checkpoint(
                    path=self.save_path,
                    epoch=epoch,
                    val_dice=best_val_dice,
                )
                if epoch % self.log_every == 0:
                    print(f"  ✓ New best checkpoint saved: {self.save_path}")

        # Save final checkpoint at end of training
        final_val_dice = history["val/dice"][-1] if history["val/dice"] else -1.0
        self._save_checkpoint(
            path=self.final_save_path,
            epoch=self.num_epochs,
            val_dice=final_val_dice,
        )

        with open(self.history_save_path, "w") as f:
            json.dump(history, f)

        print(f"Training history saved to {self.history_save_path}")
        print(f"Final checkpoint saved to {self.final_save_path}")

        return history

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        val_loss_accum = 0.0
        val_batches = 0

        for val_batch in val_loader:
            x_val = val_batch["image"].to(self.device, non_blocking=True)
            y_val = val_batch["label"].to(self.device, non_blocking=True).float()

            logits_val = self.model(x_val)
            recon_loss = self.criterion.recon_loss(logits_val, y_val)

            val_loss_accum += recon_loss.item()
            val_batches += 1

            self.val_dice_metric(
                y_pred=(torch.sigmoid(logits_val) > 0.5).float(),
                y=y_val,
            )

            if self.fast_val:
                break

        avg_val_loss = val_loss_accum / max(val_batches, 1)
        avg_val_dice = self.val_dice_metric.aggregate().item()
        self.val_dice_metric.reset()

        return avg_val_loss, avg_val_dice