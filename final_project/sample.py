import torch
from model import DiffusionModel
import yaml
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MAX_AGE = 91
MAX_ADMISSIONS = 238

def sample_from_checkpoint(cfg, checkpoint_path, text_desc_list=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if text_desc_list is None:
        text_desc_list = ["female, married, dead, asian ethnicity"]

    # Load embedding model
    embed_model = SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")
    text_embeds = embed_model.encode(text_desc_list)
    text_embeds = torch.from_numpy(text_embeds).to(device=device)

    text_embed_dim = cfg["model"]["text_embed_dim"]
    assert text_embeds.shape[-1] == text_embed_dim

    # Load diffusion model
    model = DiffusionModel(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device=device)

    T = cfg["data_utils"]["T"]
    lambda_min = cfg["data_utils"]["lambda_min"]
    lambda_max = cfg["data_utils"]["lambda_max"]
    guidance_scale = cfg["sampler"]["guidance_scale"]
    var_interp_coeff = cfg["sampler"]["var_interpolation_coeff"]

    all_samples = []

    for text_embed in tqdm(text_embeds, desc="Prompts"):
        z_t = torch.randn((1, cfg["model"]["input_dim"]), device=device)
        empty_embed = text_embed.unsqueeze(0) * 0
        text_embed = text_embed.unsqueeze(0)  # add batch dimension

        noise_b = math.atan(math.exp(-lambda_max / 2))
        noise_a = math.atan(math.exp(-lambda_min / 2)) - noise_b
        lambda_schedule = torch.linspace(lambda_min, lambda_max, T)

        with torch.no_grad():
            for t in reversed(range(T)):
                l = -2 * math.log(math.tan(noise_a * (t / T) + noise_b))
                l_scaled = math.tanh(l / (lambda_max - lambda_min))
                l_tensor = torch.full((z_t.shape[0], 1), l_scaled, device=device)

                signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))
                noise_coeff = math.sqrt(1 - (signal_coeff ** 2))

                cond_epsilon = model(z_t, text_embed, l_tensor)
                uncond_epsilon = model(z_t, empty_embed, l_tensor)
                epsilon_pred = ((1 + guidance_scale) * cond_epsilon) - (guidance_scale * uncond_epsilon)

                x_t = (z_t - (noise_coeff * epsilon_pred)) / signal_coeff

                if t > 0:
                    l_next = -2 * math.log(math.tan(noise_a * ((t-1) / T) + noise_b))
                    signal_coeff_next = math.sqrt(1 / (1 + math.exp(-l_next)))
                    noise_coeff_next = math.sqrt(1 - (signal_coeff_next ** 2))
                    exp_diff = math.exp(l - l_next)

                    mu = exp_diff * (signal_coeff_next / signal_coeff) * z_t + (1 - exp_diff) * signal_coeff_next * x_t
                    var_bar = (1 - exp_diff) * (noise_coeff_next ** 2)
                    var = (1 - exp_diff) * (noise_coeff ** 2)
                    sigma_squared = (var_bar ** (1 - var_interp_coeff)) * (var ** var_interp_coeff)
                    sigma = math.sqrt(sigma_squared)

                    sample = torch.randn_like(z_t).to(device=device)
                    z_t = (sigma * sample) + mu
                else:
                    z_t = torch.tanh(x_t)

        z_t = (z_t + 1) / 2
        z_t[:, 1] *= MAX_AGE
        z_t[:, 3] *= MAX_ADMISSIONS
        z_t = z_t.round()
        all_samples.append(z_t.cpu().numpy()[0])

    return np.array(all_samples)  # shape: (num_prompts, feature_dim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    parser.add_argument("ckpt_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    out = sample_from_checkpoint(cfg, args.ckpt_path)
    print(out)
