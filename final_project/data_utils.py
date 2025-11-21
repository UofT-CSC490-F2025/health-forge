import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training"""
    def __init__(self, data, text_embeds, alpha_bar):
        """
        data: [N, D] tensor - clean data
        alpha_bar: [T] tensor - cumulative product of alphas
        """
        self.data = data
        self.text_embeds = text_embeds
        self.alpha_bar = alpha_bar
        self.T = len(alpha_bar)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x0 = self.data[idx]  # [D]
        text_embed = self.text_embeds[idx]
        
        # Sample random timestep
        t = torch.randint(0, self.T, (1,)).item()
        
        # Sample noise
        epsilon = torch.randn_like(x0)  # [D]
        
        # Add noise: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
        alpha_bar_t = self.alpha_bar[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
        
        return x_t, t, epsilon, text_embed


def create_noise_schedule(T=1000, schedule='linear', device='cpu'):
    """
    Create noise schedule for diffusion
    
    T: number of timesteps
    schedule: 'linear' or 'cosine'
    
    Returns:
        beta: [T] - noise schedule
        alpha: [T] - 1 - beta
        alpha_bar: [T] - cumulative product of alpha
    """
    if schedule == 'linear':
        beta = torch.linspace(1e-4, 0.02, T, device=device)
    
    elif schedule == 'cosine':
        s = 0.008
        steps = torch.arange(T + 1, device=device) / T
        alpha_bar = torch.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = torch.clip(beta, 0, 0.999)
    
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    return beta, alpha, alpha_bar


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
    schedule = cfg['schedule']
    num_workers = cfg['num_workers']
    batch_size = cfg['batch_size']


    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if isinstance(text_embeds, np.ndarray):
        text_embeds = torch.from_numpy(text_embeds).float()
    data = data.to(device)
    text_embeds = text_embeds.to(device)
    
    # Create noise schedule
    beta, alpha, alpha_bar = create_noise_schedule(T, schedule, device)
    
    # Train/test split
    dataset_size = len(data)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size
    
    # train_data, test_data = random_split(
    #     data, 
    #     [train_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    indices = torch.randperm(dataset_size)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_data, test_data = data[train_indices], data[test_indices]
    train_embeds, test_embeds = text_embeds[train_indices], text_embeds[test_indices]
    
    # Create datasets
    train_dataset = DiffusionDataset(
        train_data,
        train_embeds,
        alpha_bar
    )
    test_dataset = DiffusionDataset(
        test_data,
        test_embeds,
        alpha_bar
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
    
    noise_schedule = {
        'beta': beta,
        'alpha': alpha,
        'alpha_bar': alpha_bar,
        'T': T
    }
    
    return train_loader, test_loader, noise_schedule

