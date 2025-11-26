import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np

class DiffusionDataset(Dataset):
    """Dataset for diffusion model training with on-the-fly noise generation"""
    def __init__(self, data, text_embeds, T, noise_a, noise_b, embed_drop_prob, device):
        """
        data: [N, D] tensor - clean data
        text_embeds: [N, text_dim] tensor
        T: number of diffusion steps (unused here, kept for consistency)
        noise_a, noise_b: noise schedule parameters
        embed_drop_prob: probability to drop text embeddings
        device: 'cpu' or 'cuda'
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(text_embeds, np.ndarray):
            text_embeds = torch.from_numpy(text_embeds).float()

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
        x = self.data[idx]  # clean sample
        text_embed = self.text_embeds[idx]

        # Drop text embedding some of the time
        if random.random() < self.embed_drop_prob:
            text_embed = torch.zeros_like(text_embed)

        # Sample u and compute l
        u = random.random()
        l = -2 * math.log(math.tan(self.noise_a * u + self.noise_b))

        signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))  # alpha
        noise_coeff = math.sqrt(1 - signal_coeff**2)      # sigma

        epsilon = torch.randn_like(x)
        z_l = signal_coeff * x + noise_coeff * epsilon

        return z_l, text_embed, epsilon


def prepare_diffusion_dataloaders(data, text_embeds, cfg, device):
    """
    Prepare train and test dataloaders
    """
    test_split = cfg['test_split']
    T = cfg['T']
    num_workers = cfg['num_workers']
    batch_size = cfg['batch_size']
    lambda_min = cfg["lambda_min"]
    lambda_max = cfg["lambda_max"]
    embed_drop_prob = cfg["embed_drop_prob"]

    # Train/test split
    dataset_size = len(data)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size

    indices = torch.randperm(dataset_size)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_data, test_data = data[train_indices], data[test_indices]
    train_embeds, test_embeds = text_embeds[train_indices], text_embeds[test_indices]

    # Noise schedule
    noise_b = math.atan(math.exp(-lambda_max / 2))
    noise_a = math.atan(math.exp(-lambda_min / 2)) - noise_b

    # Datasets
    train_dataset = DiffusionDataset(train_data, train_embeds, T, noise_a, noise_b, embed_drop_prob, device)
    test_dataset = DiffusionDataset(test_data, test_embeds, T, noise_a, noise_b, embed_drop_prob, device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    return train_loader, test_loader
