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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Use gradient scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.startswith("cuda")))
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma=0.5)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        # tqdm on the dataloader
        for z_l, text_embed, l, epsilon_true in tqdm(self.train_loader, desc='Training', leave=False):

            # Move batch to same device as model
            z_l = z_l.to(self.device, non_blocking=True)
            text_embed = text_embed.to(self.device, non_blocking=True)
            epsilon_true = epsilon_true.to(self.device, non_blocking=True)
            l = l.to(self.device, dtype=torch.float, non_blocking=True)
            l = l.unsqueeze(-1)


            self.optimizer.zero_grad()

            # Mixed precision forward/backward
            with torch.cuda.amp.autocast(enabled=(self.device.startswith("cuda"))):
                epsilon_pred = self.model(z_l, text_embed, l)
                loss = self.criterion(epsilon_pred, epsilon_true)

            # Scaled backward
            if self.device.startswith("cuda"):
                self.scaler.scale(loss).backward()
                # gradient clipping on the unscaled gradients requires scaler.unscale_
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
            for z_l, text_embed, l, epsilon_true in tqdm(self.test_loader, desc='Validation', leave=False):
                z_l = z_l.to(device, non_blocking=True)
                text_embed = text_embed.to(device, non_blocking=True)
                epsilon_true = epsilon_true.to(device, non_blocking=True)
                l = l.to(self.device, dtype=torch.float, non_blocking=True)
                l = l.unsqueeze(-1)


                with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                    epsilon_pred = self.model(z_l, text_embed, l)
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
            self.lr_scheduler.step()

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
