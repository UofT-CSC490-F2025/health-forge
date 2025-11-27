import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import math
import random

class DiffusionDataset(Dataset):
    """Dataset for diffusion model training"""
    def __init__(self, data, text_embeds, T, noise_a, noise_b, embed_drop_prob, device):
        """
        data: [N, D] tensor - clean data
        alpha_bar: [T] tensor - cumulative product of alphas
        """
        self.data = data
        self.text_embeds = text_embeds
        self.T = T
        self.noise_a = noise_a
        self.noise_b = noise_b
        self.embed_drop_prob = embed_drop_prob
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # original tensor
        text_embed = self.text_embeds[idx]
        # Drop text embedding embed_drop_prob % of time
        text_embed = (1 - ((torch.rand((1,)) <= self.embed_drop_prob)).to(dtype=torch.int).item()) * text_embed 

        u = random.random()
        l = -2 * math.log(math.tan(self.noise_a * u + self.noise_b))

        signal_coeff = math.sqrt(1 / (1 + math.e ** (-l)))     # alpha
        noise_coeff = math.sqrt(1 - (signal_coeff ** 2))       # sigma

        epsilon = torch.randn_like(x)

        z_l = (signal_coeff * x) + (noise_coeff * epsilon)
        
        return z_l, text_embed, epsilon


def prepare_diffusion_dataloaders(data, text_embeds, cfg, device):
    """
    Prepare train and test dataloaders for diffusion model
    
    data: [B, D] tensor or numpy array
    test_split: fraction for test set
    T: number of diffusion timesteps
    schedule: 'linear' or 'cosine'
    batch_size: batch size
    device: 'cpu' or 'cuda'
    num_workers: number of dataloader workers
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        noise_schedule: dict with beta, alpha, alpha_bar
    """
    test_split = cfg['test_split']
    T = cfg['T']
    num_workers = cfg['num_workers']
    batch_size = cfg['batch_size']
    lambda_min = cfg["lambda_min"]
    lambda_max = cfg["lambda_max"]
    embed_drop_prob = cfg["embed_drop_prob"]

    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if isinstance(text_embeds, np.ndarray):
        text_embeds = torch.from_numpy(text_embeds).float()
    data = data.to(device)
    text_embeds = text_embeds.to(device)
    
    # Train/test split
    dataset_size = len(data)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size
    print(f"Train samples: {train_size} | Test samples {test_size}")
    
    indices = torch.randperm(dataset_size)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_data, test_data = data[train_indices], data[test_indices]
    train_embeds, test_embeds = text_embeds[train_indices], text_embeds[test_indices]

    noise_b = math.atan(math.exp(-lambda_max / 2))
    noise_a = math.atan(math.exp(-lambda_min / 2)) - noise_b
    
    # Create datasets
    train_dataset = DiffusionDataset(
        train_data,
        train_embeds,
        T,
        noise_a,
        noise_b,
        embed_drop_prob,
        device
    )
    test_dataset = DiffusionDataset(
        test_data,
        test_embeds,
        T,
        noise_a,
        noise_b,
        embed_drop_prob,
        device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    return train_loader, test_loader

