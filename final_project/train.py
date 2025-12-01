import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from data_utils import prepare_diffusion_dataloaders
import yaml
import pickle
import argparse
import numpy as np

def train_from_pkl(cfg, samples, text_embeds, save_path="best_diffusion_model.pt", resume_ckpt=None):
    """Train using a pickle file containing samples and text embeddings"""
    # SKIP NORMALIZATION FOR NOW
    # samples = torch.from_numpy(samples)
    # oldest_age = torch.max(samples[:, 1])
    # max_admissions = torch.max(samples[:, 3])
    # samples[:, 1] = samples[:, 1] / oldest_age
    # samples[:, 3] = samples[:, 3] / max_admissions
    # samples = (samples * 2) - 1
    # assert (samples.max() <= 1.0) and (samples.min() >= -1.0), "Samples are not in a normalized range"
    text_embeds = torch.from_numpy(text_embeds)
    # print(f"Oldest age: {oldest_age}")
    # print(f"Max admissions: {max_admissions}")
    print(f"SAMPLES SHAPE {samples.shape} | TEXT EMBEDS SHAPE {text_embeds.shape}")
    print(f"SAMPLE EXAMPLE: {samples[0]}")
    print(f"EMBED EXAMPLE: {text_embeds[0]}")


    assert samples.shape[0] == text_embeds.shape[0], "Mismatch between samples and text embeddings"

    # Take first num_samples if provided in config
    num_samples = cfg.get("num_samples", None)
    if num_samples is not None:
        samples = samples[:num_samples]
        text_embeds = text_embeds[:num_samples]


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

    if resume_ckpt is not None:
        trainer.load_checkpoint(resume_ckpt)

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
