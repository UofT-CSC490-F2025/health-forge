import torch
from .model import DiffusionModel
import argparse
import yaml
from tqdm import tqdm
import math
from sentence_transformers import SentenceTransformer



def sample(cfg):
    # Start with pure noise
    # Starting at timestep = T, predict how much noise is in the vector using trained model
    # Given epsilon noise, get new vector by removing that much noise from the previously noisy vector
    # Use new vector in next timestep
    # Repeat until timestep=0, at which point return the vector
    T = cfg["data_utils"]["T"]
    lambda_min = cfg["data_utils"]["lambda_min"]
    lambda_max = cfg["data_utils"]["lambda_max"]
    guidance_scale = cfg["sampler"]["guidance_scale"]
    var_interp_coeff = cfg["sampler"]["var_interpolation_coeff"]
    text_embed_dim = cfg["model"]["text_embed_dim"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # text_desc = input("Enter a text description of the person: ")
    #TODO: Make this dynamic
    text_desc = "A 62-year-old female individual who has recently completed her journey through life's challenges and successfully navigated through various stages of retirement without experiencing any major health issues such as diabetes, cancer, or depression or schizophrenia."
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embed = embed_model.encode([text_desc])
    text_embed = torch.from_numpy(text_embed).to(device=device)
    assert text_embed.shape[-1] == text_embed_dim, "Got a text embedding of a shape incompatible with model"
    empty_embed = text_embed * 0

    model = DiffusionModel(cfg["model"])
    model_state_dict = torch.load("/Users/abdus/School/CSC490/health-forge/final_project/best_diffusion_model.pt")["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device=device)

    z_t = torch.randn((1, 8)).to(device=device)
    lambda_schedule = torch.linspace(lambda_min, lambda_max, T)

    for t in tqdm(range(T)):
        l = lambda_schedule[t]  # lambda
        signal_coeff = math.sqrt(1 / (1 + math.exp(-l)))      # alpha
        noise_coeff = math.sqrt(1 - (signal_coeff ** 2))        # sigma

        with torch.no_grad():
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

    # data = (data * 2) -1
    z_t = (z_t + 1) / 2
    print(z_t)
    return z_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    sample(cfg)