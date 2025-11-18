import torch
from model import DiffusionModel
from trainer import DiffusionTrainer
from data_utils import prepare_diffusion_dataloaders
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path")
    parser.add_argument("--mode", choices=["train", "sample", "eval"])
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)


    # Your dataset
    # TODO: REPLACE WITH REAL DATA
    data = torch.randn(1000, cfg["model"]["input_dim"])  # [B=1000, D=256]
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare dataloaders
    train_loader, test_loader, noise_schedule = prepare_diffusion_dataloaders(
        data,
        cfg["data_utils"],
        device
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