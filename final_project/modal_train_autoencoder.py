import modal
import boto3
import tempfile
import os
import random
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket"
MERGED_KEY = "data/final_merged.pkl"
DATA_KEY = "original_vectors_gemma.npy"
EMBEDS_KEY = "vector_tag_embeddings_gemma.npy"

MODEL_OUTPUT_KEY = "autoencoder/best_autoencoder_model.pt"

RESUME = False

app = modal.App("diffusion-training-app")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "numpy",
        "boto3",
        "pyyaml",
        "sentence-transformers",
        "tqdm",
        "transformers",
        "scipy",
        "scikit-learn",
        "pandas"
    ])
    .add_local_file("autoencoder.py", "/root/autoencoder.py"))
  


aws_secret = modal.Secret.from_name("aws-secret")

# ---------------------------
# GPU TRAINING WORKER
# ---------------------------
@app.function(
    gpu=["H100"],       # single-GPU container
    timeout=20*60*60,
    image=image,
    secrets=[aws_secret]
)
def train_worker():
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from autoencoder import EHRLatentAutoencoder

    print("GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    
    s3 = boto3.client("s3")

    data_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, DATA_KEY, data_tmp)
    data_tmp.close()


    print("Loading samples...")
    samples = np.load(data_tmp.name)

    oldest_age = samples[:, 1].max()
    max_admissions = samples[:, 3].max()

    samples[:, 1] = samples[:, 1] / oldest_age
    samples[:, 3] = samples[:, 3] / max_admissions
    assert (samples.max() <= 1.0) and (samples.min() >= 0.0), "Samples are not in a normalized range"


    print("Loaded dataset:")
    print(" samples:", samples.shape)

    num_features = samples.shape[1]
    cont_idx = [1, 3]
    binary_idx = [i for i in range(num_features) if i not in cont_idx]
    
   # train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(samples, test_size=0.1, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=512, shuffle=False)

    #Train
    X_all = torch.from_numpy(samples)  # (N, D)
    X_bin_all = X_all[:, binary_idx]   # only binary features

    pos = X_bin_all.sum()
    neg = X_bin_all.numel() - pos
    pos_weight = neg / pos

    print("pos_weight:", pos_weight.item())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cont_idx_t   = torch.tensor(cont_idx,   device=device)
    binary_idx_t = torch.tensor(binary_idx, device=device)   

    model = EHRLatentAutoencoder(input_dim=samples.shape[1], latent_dim=1024).to(device)

    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    num_epochs = 100
    best_val = float("inf")
    patience = 30
    bad_epochs = 0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        total_train_loss = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device) 

            x_bin   = batch_x[:, binary_idx_t]
            x_cont  = batch_x[:, cont_idx_t]

            optimizer.zero_grad()
            logits, z = model(batch_x)

            logits_bin  = logits[:, binary_idx_t]  # (B, #bin)
            pred_cont   = logits[:, cont_idx_t]                

            loss_bin  = bce_loss(logits_bin, x_bin)
            loss_cont = mse_loss(pred_cont, x_cont)

            lambda_cont = 10 

            loss = loss_bin + lambda_cont * loss_cont
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ---- val ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_bin   = batch_x[:, binary_idx_t]
                x_cont  = batch_x[:, cont_idx_t]

                logits, z = model(batch_x)

                logits_bin  = logits[:, binary_idx_t]  
                pred_cont   = logits[:, cont_idx_t]                

                loss_bin  = bce_loss(logits_bin, x_bin)
                loss_cont = mse_loss(pred_cont, x_cont)

                lambda_cont = 10 

                loss = loss_bin + lambda_cont * loss_cont

                total_val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        # ---- early stopping ----
        if avg_val_loss < best_val - 1e-4:
            torch.save(model.state_dict(), "ehr_autoencoder_best.pt")
            best_val = avg_val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break


    # Upload the trained model
    best_model_path = "/root/ehr_autoencoder_best.pt"
    if os.path.exists(best_model_path):
        print("Uploading best model to S3...")
        s3.upload_file(best_model_path, BUCKET, MODEL_OUTPUT_KEY)
        print("Upload complete.")
    else:
        print("ERROR: best model file not found")

  
# ---------------------------
# ENTRYPOINT
# ---------------------------
@app.local_entrypoint()
def main():
    h = train_worker.spawn()
    h.get()
    print("Training complete.")

if __name__ == "__main__":
    with app.run():
        main()
