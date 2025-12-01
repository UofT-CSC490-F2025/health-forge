import tempfile
import boto3
import torch
from autoencoder import EHRLatentAutoencoder
from model import DiffusionModel
import yaml
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def sample_from_checkpoint(cfg, checkpoint_path, text_desc: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = cfg["data_utils"]["T"]
    lambda_min = cfg["data_utils"]["lambda_min"]
    lambda_max = cfg["data_utils"]["lambda_max"]
    guidance_scale = cfg["sampler"]["guidance_scale"]
    var_interp_coeff = cfg["sampler"]["var_interpolation_coeff"]
    text_embed_dim = cfg["model"]["text_embed_dim"]

    text_desc = "This is a young male who is married and asian"

    embed_model = SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")
    text_embed = embed_model.encode([text_desc])
    text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)
    text_embed = torch.from_numpy(text_embed).to(device=device)

    assert text_embed.shape[-1] == text_embed_dim

    model = DiffusionModel(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # checkpoint = torch.load("best_diffusion_model_truedata_4096h_3l.pt", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device=device)

    z_t = torch.randn((100, cfg["model"]["input_dim"]))
    z_t = z_t.to(device=device)
    empty_embed = text_embed * 0

    noise_b = math.atan(math.exp(-lambda_max / 2))
    noise_a = math.atan(math.exp(-lambda_min / 2)) - noise_b

    lambda_schedule = torch.linspace(lambda_min, lambda_max, T)
    with torch.no_grad():
        # for t in tqdm(range(T)):
        for t in tqdm(reversed(range(T))):

            # l = lambda_schedule[t]  # lambda
            l = -2 * math.log(math.tan(noise_a * (t / T) + noise_b))

            l_scaled = math.tanh(l / (lambda_max - lambda_min))
            l_tensor = torch.full((z_t.shape[0], 1), l_scaled, device=device)


            signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))      # alpha
            noise_coeff = math.sqrt(1 - (signal_coeff ** 2))        # sigma

            cond_epsilon = model(z_t, text_embed, l_tensor)
            uncond_epsilon = model(z_t, empty_embed, l_tensor)


            epsilon_pred = ((1 + guidance_scale) * cond_epsilon) - (guidance_scale * uncond_epsilon)
            x_t = (z_t - (noise_coeff * epsilon_pred)) / signal_coeff
            
            # if t < T - 1 :
            if t > 0:
                # TODO: Algorithm 2: Line 5
                # l_next = lambda_schedule[t+1]
                l_next = -2 * math.log(math.tan(noise_a * ((t-1) / T) + noise_b))

                signal_coeff_next =  math.sqrt(1 / (1 + math.exp(-l_next)))         # alpha
                noise_coeff_next = math.sqrt(1 - (signal_coeff_next ** 2))          # sigma

                # Equation 3
                exp_diff = math.exp(l - l_next)


                mu = exp_diff * (signal_coeff_next / signal_coeff) * z_t + (1 - exp_diff) * signal_coeff_next * x_t
                var_bar = (1 - exp_diff) * (noise_coeff_next ** 2)
                var = (1 - exp_diff) * (noise_coeff ** 2)

                sigma_squared = (var_bar ** (1 - var_interp_coeff)) * (var ** var_interp_coeff)
                sigma = math.sqrt(sigma_squared)

                sample = torch.randn_like(z_t).to(device=device)
                z_t = (sigma * sample) + mu

            else:
                z_t = x_t


    print("z_t stats BEFORE denorm:", z_t.mean().item(), z_t.std().item())            
    INPUT_DIM = 1806
    LATENT_DIM = 1024

    autoencoder = EHRLatentAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to("cuda")
    state_dict = torch.load("best_autoencoder_model.pt", map_location="cuda")
   
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

     # Freeze AE weights
    for p in autoencoder.parameters():
        p.requires_grad = False

    print("Autoencoder loaded.")


    latent_mean = np.load("latent_mean.npy")

    latent_std = np.load("latent_std.npy")
  
    latent_mean = torch.from_numpy(latent_mean).to(device=z_t.device, dtype=z_t.dtype)
    latent_std  = torch.from_numpy(latent_std).to(device=z_t.device, dtype=z_t.dtype)

    print("Loading samples...")
    
    z_t = z_t * latent_std + latent_mean

    print("z_t stats AFTER denorm:", z_t.mean().item(), z_t.std().item())

    z_t = autoencoder.decoder(z_t)

    print("decoder logits stats:", z_t.mean().item(), z_t.std().item())

    cont_idx = [1, 3]
    binary_idx = [i for i in range(1806) if i not in cont_idx]

    temperature = 5.0   # try 2.0 to 5.0
    threshold = 0.8     # try 0.6–0.9

    probs = torch.sigmoid(z_t / temperature)
    x_bin = (probs[:, binary_idx] > threshold).float()

    MAX_ADMISSIONS = 238
    MAX_AGE= 91
    #    reconstruct full vector
    formatted = probs.clone()
    formatted[:, binary_idx] = x_bin

    formatted[:, 1] = formatted[:, 1]*MAX_AGE
    formatted[:, 3] = formatted[:, 3]*MAX_ADMISSIONS
    print("GENERATED SAMPLE: ", formatted)

    return formatted.cpu().numpy()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    parser.add_argument("ckpt_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    out = sample_from_checkpoint(cfg, args.ckpt_path, "diabetes")
    print(out)

   
    count = 0
    for row in out:
        print(row[367 : 372])
        print(np.sum(row >= 1.0))

    print(count)
