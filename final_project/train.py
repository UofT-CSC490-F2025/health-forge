import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from torch.utils.data import TensorDataset, DataLoader
from autoencoder import EHRLatentAutoencoder
from data_utils import prepare_diffusion_dataloaders
import yaml
import pickle
import argparse
import numpy as np
import boto3

def train_from_pkl(cfg, samples, text_embeds, autoencoder_path, save_path="best_diffusion_model.pt",latent_mean_path = 'latent_mean.npy', latent_std_path = 'latent_std.npy', resume_ckpt=None):
    """
    Train a latent-diffusion model using EHR samples and text embeddings.

    This function loads a pretrained EHR autoencoder, encodes the full dataset into
    latent space, normalizes latents and text embeddings, builds dataloaders, and
    trains a diffusion model specified by `cfg`. Latent normalization statistics
    are saved for later use during sampling, and a final model checkpoint is written
    to disk.

    Parameters
    ----------
    cfg : dict
        Configuration dict containing model, trainer, and dataloader settings.
    samples : np.ndarray
        Raw EHR feature vectors (shape: N × 1806). Age and admission count are normalized.
    text_embeds : np.ndarray
        L2-normalized text embeddings aligned with the samples.
    autoencoder_path : str
        Path to a trained `EHRLatentAutoencoder` checkpoint.
    save_path : str
        Output path for the trained diffusion model checkpoint.
    latent_mean_path : str
        Where to save the latent mean vector.
    latent_std_path : str
        Where to save the latent std vector.
    resume_ckpt : str or None
        Optional checkpoint to resume diffusion training.

    Returns
    -------
    save_path : str
        Location of the saved diffusion model.
    latent_mean_path : str
        Saved latent mean file.
    latent_std_path : str
        Saved latent std file.
    """

    #Load Autoencoder
    INPUT_DIM = 1806
    LATENT_DIM = 1024

    autoencoder = EHRLatentAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to("cuda")
    state_dict = torch.load(autoencoder_path, map_location="cuda")
   
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    # Freeze AE weights
    for p in autoencoder.parameters():
        p.requires_grad = False

    print("Autoencoder loaded.")

    #Normalize age and number of admissions
    oldest_age = samples[:, 1].max()
    max_admissions = samples[:, 3].max()

    samples[:, 1] = samples[:, 1] / oldest_age
    samples[:, 3] = samples[:, 3] / max_admissions
    assert (samples.max() <= 1.0) and (samples.min() >= 0), "Samples are not in a normalized range"


    #Normalize text embedding
    norms = np.linalg.norm(text_embeds, axis=1, keepdims=True) + 1e-8
    text_embeds = text_embeds / norms

    B = 2048  # Autoencoder Batch Size
    latents = []

    samples_t = torch.from_numpy(samples).float().to("cuda")  

    #Pass the entire dataset into the encoder to get the latent dataset
    with torch.no_grad():
        for i in range(0, samples_t.size(0), B):
            batch = samples_t[i:i+B]   
            _, z = autoencoder(batch)       

            latents.append(z.cpu())         

    latents = torch.cat(latents, dim=0)     
    latents = latents.detach().cpu().numpy()

    latent_mean = latents.mean(0)
    latent_std  = latents.std(0) + 1e-6
    latents = (latents - latent_mean) / latent_std

    #Save normalization numbers for denormalization during sampling.
    np.save(latent_mean_path, latent_mean)
    np.save(latent_std_path, latent_std)

    text_embeds = torch.from_numpy(text_embeds)
    print(f"SAMPLES SHAPE {samples.shape} | LATENTS SHAPE {latents.shape} | TEXT EMBEDS SHAPE {text_embeds.shape}")


    assert latents.shape[0] == text_embeds.shape[0], "Mismatch between samples and text embeddings"

    # Take first num_samples if provided in config
    num_samples = cfg.get("num_samples", None)
    if num_samples is not None:
        print("Truncating Samples...")
        latents = latents[:num_samples]
        text_embeds = text_embeds[:num_samples]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Prepare dataloaders
    train_loader, test_loader = prepare_diffusion_dataloaders(
        latents, text_embeds, cfg["data_utils"], device
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

    return save_path, latent_mean_path, latent_std_path


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
