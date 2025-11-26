import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from data_utils import prepare_diffusion_dataloaders
import yaml
import numpy as np

def train_from_arrays(cfg, data, text_embeds, save_path="best_diffusion_model.pt"):
    """Train numpy arrays on a single GPU (or CPU)"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Take first num_samples if provided
    num_samples = cfg.get("num_samples", None)
    if num_samples is not None:
        data = data[:num_samples]
        text_embeds = text_embeds[:num_samples]

    # Map data from 0-1 to -1..1 if needed
    if cfg.get("data_scale_from_zero_one", True):
        data = (data * 2) - 1

    # Prepare dataloaders
    train_loader, test_loader = prepare_diffusion_dataloaders(
        data, text_embeds, cfg["data_utils"], device
    )

    # Initialize model
    model = DiffusionModel(cfg["model"]).to(device)

    # Trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg["trainer"],
        device=device
    )

    train_losses, val_losses = trainer.train()

    # Save checkpoint
    torch.save({
        "epoch": trainer.num_epochs - 1,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    parser.add_argument("vectors_path")
    parser.add_argument("embeds_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    data = np.load(args.vectors_path)
    text_embeds = np.load(args.embeds_path)

    train_from_arrays(cfg, data, text_embeds)
