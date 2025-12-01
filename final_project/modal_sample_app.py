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
MODEL_OUTPUT_KEY = "results/best_diffusion_model_truedata_4096h_3l.pt"
SAMPLE_OUTPUT_KEY = "results/sample_output.npy"

# Local filenames you said you have
LOCAL_DIFFUSION = "best_diffusion_model.pt"
LOCAL_AUTOENCODER = "best_autoencoder_model.pt"   # local name
# inside container we will expose it as best_autoencoder_model.pt to match sample.py
CONTAINER_AUTOENCODER = "/root/best_autoencoder_model.pt"
LOCAL_LATENT_MEAN = "latent_mean.npy"
LOCAL_LATENT_STD = "latent_std.npy"
LOCAL_CONFIG = "configs_og.yaml"

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

    # optional local model files, mapped into container paths sample.py expects
    # diffusion ckpt
    .add_local_file(LOCAL_DIFFUSION, "/root/{}".format(LOCAL_DIFFUSION))
    # autoencoder local file put at container filename sample.py expects
    .add_local_file(LOCAL_AUTOENCODER, CONTAINER_AUTOENCODER)
    # latent stats and config
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
    local_diff_path = f"/root/{LOCAL_DIFFUSION}"
    if os.path.exists(local_diff_path):
        diffusion_ckpt_path = local_diff_path
    else:
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

    s3.upload_file(out_tmp, BUCKET, "results/all_samples.npy")

    os.unlink(out_tmp)

    return {
        "status": "complete",
        "num_prompts": len(prompts),
        "output_s3_key": "results/all_samples.npy",
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
