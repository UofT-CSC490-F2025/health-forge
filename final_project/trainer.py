import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from data_utils import prepare_diffusion_dataloaders
from model import DiffusionModel

class DiffusionTrainer:
    """Trainer class for diffusion models"""
    def __init__(self, model, train_loader, test_loader, noise_schedule, 
                 cfg, device='cpu'):
        """
        model: diffusion model
        train_loader: training dataloader
        test_loader: validation dataloader
        noise_schedule: dict with noise schedule parameters
        lr: learning rate
        device: 'cpu' or 'cuda'
        """
        lr = cfg["lr"]
        self.num_epochs = cfg["num_epochs"]
        self.save_path = cfg["save_path"]
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.noise_schedule = noise_schedule
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for x_t, t, epsilon_true in tqdm(self.train_loader, desc='Training'):
            x_t = x_t.to(self.device)
            t = t.to(self.device)
            epsilon_true = epsilon_true.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            epsilon_pred = self.model(x_t, t)
            
            # Loss
            loss = self.criterion(epsilon_pred, epsilon_true)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_t, t, epsilon_true in tqdm(self.test_loader, desc='Validation'):
                x_t = x_t.to(self.device)
                t = t.to(self.device)
                epsilon_true = epsilon_true.to(self.device)
                
                # Forward pass
                epsilon_pred = self.model(x_t, t)
                
                # Loss
                loss = self.criterion(epsilon_pred, epsilon_true)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self):
        """
        Train the model for multiple epochs
        
        num_epochs: number of epochs to train
        save_path: path to save best model (optional)
        """
        best_val_loss = float('inf')
        num_epochs = self.num_epochs
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Save best model
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
        
        return self.train_losses, self.val_losses
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

