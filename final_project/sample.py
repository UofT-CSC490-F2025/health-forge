import torch
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

    # if text_desc is None:
    #     text_desc = "A 62 year old female individual who has recently completed her journey through life without major health issues"
    text_desc = "male with diabetes and cancer and depression"

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embed = embed_model.encode([text_desc])
    text_embed = torch.from_numpy(text_embed).to(device=device)

    assert text_embed.shape[-1] == text_embed_dim

    model = DiffusionModel(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device=device)

    z_t = torch.randn((100, cfg["model"]["input_dim"]))
    z_t = z_t.to(device=device)
    empty_embed = text_embed * 0


    lambda_schedule = torch.linspace(lambda_min, lambda_max, T)
    with torch.no_grad():
        for t in tqdm(range(T)):
            l = lambda_schedule[t]  # lambda
            signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))      # alpha
            noise_coeff = math.sqrt(1 - (signal_coeff ** 2))        # sigma

            cond_epsilon = model(z_t, text_embed)
            uncond_epsilon = model(z_t, empty_embed)


            epsilon_pred = ((1 + guidance_scale) * cond_epsilon) - (guidance_scale * uncond_epsilon)
            x_t = (z_t - (noise_coeff * epsilon_pred)) / signal_coeff
            
            if t < T - 1 :
                # TODO: Algorithm 2: Line 5
                l_next = lambda_schedule[t+1]
                signal_coeff_next =  math.sqrt(1 / (1 + math.exp(-l_next)))         # alpha
                noise_coeff_next = math.sqrt(1 - (signal_coeff_next ** 2))          # sigma

                # Equation 3
                exp_diff = math.exp(l - l_next)


                mu = exp_diff * (signal_coeff_next / signal_coeff) * z_t + (1 - exp_diff) * signal_coeff_next * x_t
                var_bar = (1 - exp_diff) * (noise_coeff_next ** 2)
                var = (1 - exp_diff) * (noise_coeff ** 2)


                sigma = (var_bar ** (1 - var_interp_coeff)) * (var ** var_interp_coeff)
                # sigma = max(sigma, 1e-10)  # Prevent numerical issues

                sample = torch.randn_like(z_t).to(device=device)
                z_t = (sigma * sample) + mu

            else:
                z_t = x_t

    z_t = (z_t + 1) / 2

    formatted_sample = z_t
    formatted_sample[:, 1] *= 100
    formatted_sample = formatted_sample.round()
    print(formatted_sample)

    return formatted_sample.cpu().numpy()

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
