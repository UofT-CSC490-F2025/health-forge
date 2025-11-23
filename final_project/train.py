import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from data_utils import prepare_diffusion_dataloaders
import argparse
import yaml
import numpy as np
import pickle


def train(cfg, data):
     # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare dataloaders
    train_loader, test_loader, noise_schedule = prepare_diffusion_dataloaders(
        data,
        text_embeds,
        cfg["data_utils"],
        device,
    )
    
    # Create model (you need to implement this)
    model = DiffusionModel(cfg["model"])
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        noise_schedule=noise_schedule,
        cfg=cfg["trainer"],
        device=device
    )
    
    # Train
    train_losses, val_losses = trainer.train()
    

    # Load best model
    trainer.load_checkpoint('best_diffusion_model.pt')
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    parser.add_argument("data_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    samples, descs, llm_descs, text_embeds = data["samples"], data["descs"], data["llm_decs"], data["text_embeds"]
    assert samples.shape[0] == text_embeds.shape[0], "Different number of samples and text embedddings"

    # data = (samples[0])[None,:] # TODO: Only one sample
    data = (data * 2) -1
    
    train(cfg, data)
   