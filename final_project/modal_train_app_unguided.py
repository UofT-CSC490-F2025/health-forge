import modal
import boto3
import tempfile
import numpy as np
import yaml
import os
from sample import sample_from_checkpoint

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket-1"
MODEL_OUTPUT_KEY = "results/best_diffusion_model_WITH_ENCODER_BIG.pt"
SAMPLE_OUTPUT_KEY = "results/sample_output.npy"

# Local filenames you said you have
LOCAL_DIFFUSION = "rip"
LOCAL_AUTOENCODER = "best_autoencoder_model.pt"   # local name
# inside container we will expose it as best_autoencoder_model.pt to match sample.py
CONTAINER_AUTOENCODER = "/root/best_autoencoder_model.pt"
LOCAL_LATENT_MEAN = "latent_mean_BIG.npy"
LOCAL_LATENT_STD = "latent_std_BIG.npy"
LOCAL_CONFIG = "configs.yaml"

# ---------------------------
# Modal App
# ---------------------------
app = modal.App("diffusion-sampling-app")

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
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("autoencoder.py", "/root/autoencoder.py")
    .add_local_file("sample.py", "/root/sample.py")
    .add_local_file(LOCAL_AUTOENCODER, CONTAINER_AUTOENCODER)
    .add_local_file(LOCAL_LATENT_MEAN, "/root/latent_mean.npy")
    .add_local_file(LOCAL_LATENT_STD, "/root/latent_std.npy")
    .add_local_file(LOCAL_CONFIG, "/root/configs_og.yaml")
)

aws_secret = modal.Secret.from_name("aws-secret")

# ============================================================
# SAMPLING FUNCTION
# ============================================================
@app.function(
    gpu="H100",
    timeout=4 * 60 * 60,
    image=image,
    secrets=[aws_secret],
)
def sample_model():
    """
    Modal script to generate a bunch of unguided data points to gauge dataset adherence/similarity.
    """
    s3 = boto3.client("s3")

    # Load config
    cfg_path_candidates = ["/root/configs_og.yaml", "/root/configs.yaml"]
    cfg = None
    for p in cfg_path_candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                cfg = yaml.safe_load(f)
            break
    if cfg is None:
        raise FileNotFoundError("No config found")

    
    model_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, MODEL_OUTPUT_KEY, model_tmp)
    model_tmp.close()
    diffusion_ckpt_path = model_tmp.name

    # Ensure autoencoder + latent stats exist
    # ---------------------------------------------------------
    if not os.path.exists(CONTAINER_AUTOENCODER):
        ae_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        s3.download_fileobj(BUCKET, "ae/best_autoencoder_model.pt", ae_tmp)
        ae_tmp.close()
        os.replace(ae_tmp.name, CONTAINER_AUTOENCODER)

    if not (os.path.exists("/root/latent_mean.npy") and os.path.exists("/root/latent_std.npy")):
        mean_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        std_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        s3.download_fileobj(BUCKET, "ae/latent_mean.npy", mean_tmp)
        s3.download_fileobj(BUCKET, "ae/latent_std.npy", std_tmp)
        mean_tmp.close()
        std_tmp.close()
        os.replace(mean_tmp.name, "/root/latent_mean.npy")
        os.replace(std_tmp.name, "/root/latent_std.npy")

    # ---------------------------------------------------------
    # Generate samples
    # ---------------------------------------------------------
    all_outputs = []

    for i in range(0, 10):
        print(f"Generating unguided batch {i}")
        sample = sample_from_checkpoint(cfg, diffusion_ckpt_path, "")

        # sample is probably shape [4096] or [seq_len, dim]
        all_outputs.extend(sample)

    # Convert to array
    all_outputs = np.array(all_outputs)

    # Save one big file
    out_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(out_tmp, all_outputs)

    s3.upload_file(out_tmp, BUCKET, "results/all_samples_unguided.npy")

    os.unlink(out_tmp)

    return {
        "status": "complete",
        "output_s3_key": "results/all_samples_unguided.npy",
        "shape": str(all_outputs.shape),
    }


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================
@app.local_entrypoint()
def main():
    print("Launching batch sampling on Modal GPU...")
    res = sample_model.remote()
    print(res)

if __name__ == "__main__":
    with app.run():
        main()
