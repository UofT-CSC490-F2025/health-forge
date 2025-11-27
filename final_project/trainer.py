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
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for z_l, text_embed, epsilon_true in tqdm(self.train_loader, desc='Training'):
            
            self.optimizer.zero_grad()
            
            # Forward pass
            epsilon_pred = self.model(z_l, text_embed)
            
            # Loss
            loss = self.criterion(epsilon_pred, epsilon_true)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #TODO: Keep?
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        if len(self.test_loader) == 0: return 0
        
        with torch.no_grad():
            for z_l, text_embed, epsilon_true in tqdm(self.test_loader, desc='Validation'):
                # Forward pass
                epsilon_pred = self.model(z_l, text_embed)
                
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
        best_train_loss = float('inf')

        num_epochs = self.num_epochs
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            # self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {curr_lr:.6f}')
            
            # Save best model
            # if True: #TODO: !!! DONT KEEP THIS !!!
            if self.save_path and val_loss < best_val_loss:
            # if self.save_path and train_loss < best_train_loss:

                best_val_loss = val_loss
                # best_train_loss = train_loss

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.save_path)
                print(f'Saved best model to {self.save_path}')
        print(f'Saved best model to {self.save_path}')
        
        return self.train_losses, self.val_losses
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

