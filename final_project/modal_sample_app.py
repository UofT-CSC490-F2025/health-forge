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
DIFFUSION_CKPT_KEY = "results/best_diffusion_model_WITH_ENCODER_BIG.pt"
AUTOENCODER_CKPT_KEY = "autoencoder/best_autoencoder_model.pt"
LATENT_STD_KEY = "results/latent_std_BIG.npy"
LATENT_MEAN_KEY = "results/latent_mean_BIG.npy"
SAMPLE_OUTPUT_KEY = "results/sample_output.npy"
CONTAINER_AUTOENCODER = "/root/best_autoencoder_model.pt"
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
    # code files
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("autoencoder.py", "/root/autoencoder.py")
    .add_local_file("sample.py", "/root/sample.py")
    .add_local_file("patient_text_prompts.txt", "/root/patient_text_prompts.txt")
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
    Reads text prompts from /root/patient_text_prompts.txt
    Generates one sample per line
    Combines all samples into one array
    Uploads ONE final .npy file to S3.
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

    # Diffusion checkpoint resolution
    model_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, DIFFUSION_CKPT_KEY, model_tmp)
    model_tmp.close()
    diffusion_ckpt_path = model_tmp.name

    # Ensure autoencoder + latent stats exist
    # ---------------------------------------------------------
    print("Loading autoencoder")
    ae_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, AUTOENCODER_CKPT_KEY, ae_tmp)
    ae_tmp.close()
    os.replace(ae_tmp.name, CONTAINER_AUTOENCODER)

    print("Loading autoencoder mean and std stats")
    mean_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    std_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, LATENT_MEAN_KEY, mean_tmp)
    s3.download_fileobj(BUCKET, LATENT_STD_KEY, std_tmp)
    mean_tmp.close()
    std_tmp.close()
    os.replace(mean_tmp.name, "/root/latent_mean.npy")
    os.replace(std_tmp.name, "/root/latent_std.npy")

    # ---------------------------------------------------------
    # Read text prompts
    # ---------------------------------------------------------
    with open("/root/patient_text_prompts.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # ---------------------------------------------------------
    # Generate samples
    # ---------------------------------------------------------
    all_outputs = []

    for idx, text_prompt in enumerate(prompts):
        print(f"Generating for prompt {idx}: {text_prompt}")
        sample = sample_from_checkpoint(cfg, diffusion_ckpt_path, text_prompt)

        # sample is probably shape [4096] or [seq_len, dim]
        all_outputs.append(sample)

    # Convert to array
    all_outputs = np.array(all_outputs)

    # Save one big file
    out_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(out_tmp, all_outputs)

    s3.upload_file(out_tmp, BUCKET, SAMPLE_OUTPUT_KEY)

    os.unlink(out_tmp)

    return {
        "status": "complete",
        "num_prompts": len(prompts),
        "output_s3_key": SAMPLE_OUTPUT_KEY,
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
