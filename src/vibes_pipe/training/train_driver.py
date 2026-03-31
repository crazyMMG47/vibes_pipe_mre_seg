from pathlib import Path
import yaml
import torch

from src.vibes_pipe.training.builders import build_training_pipeline
from src.vibes_pipe.training.engine import train_one_epoch, validate_one_epoch


def load_yaml(path):
    path = Path(path).expanduser().resolve()
    return yaml.safe_load(path.read_text())


def run_training(cfg_path):
    cfg = load_yaml(cfg_path)

    pipeline = build_training_pipeline(cfg)

    model = pipeline["model"]
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    criterion = pipeline["criterion"]
    optimizer = pipeline["optimizer"]
    scheduler = pipeline["scheduler"]
    device = pipeline["device"]
    save_dir = pipeline["save_dir"]
    epochs = pipeline["epochs"]

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        ) if val_loader is not None else None

        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"] if val_stats is not None else None

        if scheduler is not None:
            if val_loss is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss is None:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        }

        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_dir / "last.pt")

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")

    return pipeline