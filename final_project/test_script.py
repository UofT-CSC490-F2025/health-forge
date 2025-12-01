import numpy as np
import torch

from autoencoder import EHRLatentAutoencoder

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_ADMISSIONS = 238
    MAX_AGE= 91

    INPUT_DIM = 1806
    LATENT_DIM = 1024
    # 1. Load first row and keep it as a batch of size 1
    original_vectors = np.load("original_vectors_gemma.npy")   # shape (N, 1806)
    first_row = original_vectors[0:1]                          # shape (1, 1806)

    first_row[:, 1] = first_row[:, 1] / MAX_AGE
    first_row[:, 3] = first_row[:, 3] / MAX_AGE

    # Optionally save the original row for comparison
    np.savetxt("first_row.txt", first_row, fmt="%.6f")

    # 2. Convert to torch tensor on the right device
    x = torch.from_numpy(first_row).float().to(device)         # shape (1, 1806)

    # 3. Encode + decode through the autoencoder
    with torch.no_grad():
        autoencoder = EHRLatentAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to("cuda")
        state_dict = torch.load("best_autoencoder_model.pt", map_location="cuda")
    
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        # If your AE has encoder/decoder modules (this matches your earlier code)
        z = autoencoder.encoder(x)             # shape (1, latent_dim)
        logits = autoencoder.decoder(z)        # shape (1, 1806)

        # If instead you have a forward that returns (logits, z), use:
        # logits, z = autoencoder(x)

        # 4. Convert logits → probabilities
        probs = torch.sigmoid(logits)          # shape (1, 1806), in (0,1)

        cont_idx = [1, 3]
        D = logits.shape[1]
        binary_idx = [i for i in range(D) if i not in cont_idx]

        # 5. Start from probs (safe for binary)
        formatted = probs.clone()

        # Hard-threshold binary dims
        formatted[:, binary_idx] = (probs[:, binary_idx] > 0.5).float()

        # Use raw logits (clamped to [0,1]) for continuous dims
        formatted[:, cont_idx] = logits[:, cont_idx].clamp(0, 1)

        # 6. Move back to CPU + numpy and drop batch dimension
        formatted_np = formatted.squeeze(0).cpu().numpy()      # shape (1806,)

    # 7. Save reconstructed row
    np.savetxt("final.txt", formatted_np, fmt="%.6f")