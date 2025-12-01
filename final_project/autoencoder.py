import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int = 1024 ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512, latent_dim)
        self.ln3 = nn.LayerNorm(latent_dim)

        self.res_fc = nn.Linear(latent_dim, latent_dim)
        self.res_ln = nn.LayerNorm(latent_dim)
    

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.ln2(x)

        x = self.fc3(x)
        x = F.gelu(x)
        x = self.ln3(x)

        # latent residual block
        residual = x
        h = self.res_fc(x)
        h = F.gelu(h)
        h = self.res_ln(h)
        z = h + residual         

        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 1024, output_dim: int = 1804):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 1024)
        self.ln2 = nn.LayerNorm(1024)

        self.fc3 = nn.Linear(1024, output_dim)  # logits for each code

    def forward(self, z):
        x = self.fc1(z)
        x = F.gelu(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.ln2(x)

        logits = self.fc3(x)      # (B, output_dim), no sigmoid here
        return logits
        
class EHRLatentAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 1804, latent_dim: int = 1024):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z