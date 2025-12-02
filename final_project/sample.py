import torch
from autoencoder import EHRLatentAutoencoder
from model import DiffusionModel
import yaml
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def sample_from_checkpoint(cfg, checkpoint_path, text_desc: str = None):
    """
    Generate synthetic EHR samples from a trained diffusion checkpoint.

    Loads a diffusion model checkpoint, constructs (or embeds) a text condition,
    performs reverse diffusion to produce latent vectors, and decodes them using a
    pretrained autoencoder. Handles classifier-free guidance, variance interpolation,
    latent de-normalization, and postprocessing of binary/continuous features.

    Args:
        cfg (dict): Full experiment configuration containing model, sampler,
            and data-utils parameters.
        checkpoint_path (str): Path to the saved diffusion model checkpoint.
        text_desc (str, optional): Optional conditioning text. If None, a default
            demographic description is used.

    Returns:
        np.ndarray: Generated EHR samples of shape (N, feature_dim), with binary
        features thresholded and continuous features de-normalized.
    """
    INPUT_DIM = 1806
    LATENT_DIM = 1024
    MAX_ADMISSIONS = 238
    MAX_AGE= 91

    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = cfg["data_utils"]["T"]
    lambda_min = cfg["data_utils"]["lambda_min"]
    lambda_max = cfg["data_utils"]["lambda_max"]
    guidance_scale = cfg["sampler"]["guidance_scale"]
    var_interp_coeff = cfg["sampler"]["var_interpolation_coeff"]
    text_embed_dim = cfg["model"]["text_embed_dim"]

    if text_desc == None:
        print("Using Default")
        text_desc = "This is a young male who is married and asian"

    #Embedding model
    embed_model = SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")
    text_embed = embed_model.encode([text_desc])
    text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)
    text_embed = torch.from_numpy(text_embed).to(device=device)

    assert text_embed.shape[-1] == text_embed_dim

    text_embed = text_embed.repeat(100, 1)  

    # Load diffusion model
    model = DiffusionModel(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device=device)

    z_t = torch.randn((100, cfg["model"]["input_dim"]))
    z_t = z_t.to(device=device)
    empty_embed = text_embed * 0

    noise_b = math.atan(math.exp(-lambda_max / 2))
    noise_a = math.atan(math.exp(-lambda_min / 2)) - noise_b

    with torch.no_grad():
        for t in tqdm(reversed(range(T))):

            l = -2 * math.log(math.tan(noise_a * (t / T) + noise_b))

            l_scaled = math.tanh(l / (lambda_max - lambda_min))
            l_tensor = torch.full((z_t.shape[0], 1), l_scaled, device=device)


            signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))      # alpha
            noise_coeff = math.sqrt(1 - (signal_coeff ** 2))        # sigma

            cond_epsilon = model(z_t, text_embed, l_tensor)
            uncond_epsilon = model(z_t, empty_embed, l_tensor)


            epsilon_pred = ((1 + guidance_scale) * cond_epsilon) - (guidance_scale * uncond_epsilon)
            x_t = (z_t - (noise_coeff * epsilon_pred)) / signal_coeff
            
            if t > 0:
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

    z_t = z_t*latent_std + latent_mean

    print("Loading samples...")
    
    print(z_t)
    


    logits = autoencoder.decoder(z_t)

    probs = torch.sigmoid(logits)

    cont_idx = [1, 3]
    binary_idx = [i for i in range(1806) if i not in cont_idx]

    # Start from probs (safe for binary)
    formatted = probs.clone()

    # 1. Hard threshold binary dims
    formatted[:, binary_idx] = (probs[:, binary_idx] > 0.5).float()

    # 2. Continuous dims should use raw decoder output, not sigmoid!
    formatted[:, cont_idx] = logits[:, cont_idx].clamp(0, 1)

    # 3. Denormalize continuous values
    formatted[:, 1] *= MAX_AGE
    formatted[:, 3] *= MAX_ADMISSIONS
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

    out = sample_from_checkpoint(cfg, args.ckpt_path, "early 30s")
    print(out)

   
    count = 0
    for row in out:
        print(row[0:3])
        diabetes = row[367 : 371]
        if (diabetes == 1.0).any():
            count+=1
    
    print(count)