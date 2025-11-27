import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from data_utils import prepare_diffusion_dataloaders
import yaml
import pickle
import argparse

def train_from_pkl(cfg, pkl_path, save_path="best_diffusion_model.pt"):
    """Train using a pickle file containing samples and text embeddings"""

    # Load data from pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    samples = data["samples"]
    text_embeds = data["text_embeds"]

    assert samples.shape[0] == text_embeds.shape[0], "Mismatch between samples and text embeddings"

    # Take first num_samples if provided in config
    num_samples = cfg.get("num_samples", None)
    if num_samples is not None:
        samples = samples[:num_samples]
        text_embeds = text_embeds[:num_samples]

    # Scale data from 0-1 to -1..1 if needed
    if cfg.get("data_scale_from_zero_one", True):
        samples = (samples * 2) - 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Prepare dataloaders
    train_loader, test_loader = prepare_diffusion_dataloaders(
        samples, text_embeds, cfg["data_utils"], device
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

    # Train
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
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to YAML config file")
    parser.add_argument("pkl_path", help="Path to .pkl file containing samples and embeddings")
    args = parser.parse_args()

    # Load config
    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Train from pickle
    train_from_pkl(cfg, args.pkl_path)
