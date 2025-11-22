import torch
from data_utils import create_noise_schedule
from model import DiffusionModel
import argparse
import yaml
from tqdm import tqdm

def sample(cfg):
    # Start with pure noise
    # Starting at timestep = T, predict how much noise is in the vector using trained model
    # Given epsilon noise, get new vector by removing that much noise from the previously noisy vector
    # Use new vector in next timestep
    # Repeat until timestep=0, at which point return the vector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DiffusionModel(cfg["model"])
    model_state_dict = torch.load("/Users/abdus/School/CSC490/health-forge/final_project/best_diffusion_model.pt")["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device=device)

    T = cfg["data_utils"]["T"]
    x_t = torch.randn((1, 8)).to(device=device)
    beta, alpha, alpha_bar = create_noise_schedule(T)
    beta, alpha, alpha_bar = beta.to(device=device), alpha.to(device=device), alpha_bar.to(device=device)

    # Loop from T down to 1 (the final step)
    for t in tqdm(range(T, 0, -1)): # CRITICAL CHANGE: Stop at 1
        
        t_index = t - 1 # CRITICAL CHANGE: Use t-1 index for schedule tensors
        
        # Get schedule values at timestep t (using t-1 index)
        alpha_t = alpha[t_index]
        # For the mean calculation, we need alpha_bar_{t-1} and alpha_bar_t
        # Since your model is trained with the simplified DDPM noise variance:
        alpha_bar_t = alpha_bar[t_index]
        beta_t = beta[t_index]

        # Calculate x_{t-1}
        sqrt_alpha_t = torch.sqrt(alpha_t)
        # Note: sqrt_one_minus_alpha_bar_t is sqrt(1 - alpha_bar_t), which is used correctly for the mean
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        # Pass t (the step number, 1 to T) to the model
        # Use .float() for best practice, though PyTorch handles the cast
        t_tensor = torch.tensor([t], device=device).float() 
        epsilon = model(x_t, t_tensor, None) 

        # Mean of the reverse distribution (Correct)
        mean = (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon)

        # Add noise (The final step is t=1, which produces x_0 and has no noise added)
        if t > 1: # Noise is added for steps t=T down to t=2
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(x_t)
            x_t_minus_1 = mean + sigma_t * z
        else: # When t=1, we are calculating x_0, which is just the mean
            x_t_minus_1 = mean

        # x_t_minus_1 is now your denoised sample at timestep t-1
        x_t = x_t_minus_1
    x_t = (x_t + 1) / 2 #Unnormalize
    print(x_t)
    return x_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    sample(cfg)