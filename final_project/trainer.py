import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from data_utils import prepare_diffusion_dataloaders
from model import DiffusionModel

class DiffusionTrainer:
    """Trainer class for diffusion models"""
    def __init__(self, model, train_loader, test_loader, 
                 cfg, device=None):
        """
        model: diffusion model
        train_loader: training dataloader
        test_loader: validation dataloader
        cfg: configuration dict
        device: 'cpu' or 'cuda'; if None, auto-detect GPU
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        lr = cfg["lr"]
        self.num_epochs = cfg["num_epochs"]
        self.save_path = cfg["save_path"]

        # Move model to device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for z_l, text_embed, epsilon_true in tqdm(self.train_loader, desc='Training'):
            # Move batch to same device as model
            z_l = z_l.to(self.device)
            text_embed = text_embed.to(self.device)
            epsilon_true = epsilon_true.to(self.device)

            self.optimizer.zero_grad()
            epsilon_pred = self.model(z_l, text_embed)
            loss = self.criterion(epsilon_pred, epsilon_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        if len(self.test_loader) == 0:
            return 0

        device = self.device
        with torch.no_grad():
            for z_l, text_embed, epsilon_true in tqdm(self.test_loader, desc='Validation'):
                z_l = z_l.to(device)
                text_embed = text_embed.to(device)
                epsilon_true = epsilon_true.to(device)

                epsilon_pred = self.model(z_l, text_embed)
                loss = self.criterion(epsilon_pred, epsilon_true)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        """Train the model for multiple epochs"""
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')

            train_loss = self.train_epoch()
            val_loss = self.validate()
            curr_lr = self.optimizer.param_groups[0]['lr']

            print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {curr_lr:.6f}')

            # Save best model based on validation loss
            if self.save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.save_path)
                print(f'Saved best model to {self.save_path}')

        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, self.save_path)
        print(f'Saved final model to {self.save_path}')
        return self.train_losses, self.val_losses

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
