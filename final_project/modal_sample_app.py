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
BUCKET = "healthforge-final-bucket"
MODEL_OUTPUT_KEY = "results/best_diffusion_model.pt"
SAMPLE_OUTPUT_KEY = "results/sample_output.npy"

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
    .add_local_file("sample.py", "/root/sample.py")
    .add_local_file("configs.yaml", "/root/configs.yaml")
)

aws_secret = modal.Secret.from_name("aws-secret")

# ============================================================
# SAMPLING FUNCTION
# ============================================================
@app.function(
    gpu="A100",
    timeout=2 * 60 * 60,
    image=image,
    secrets=[aws_secret],
)
def sample_model(text_prompt: str = None):
    """
    Loads trained model and produces a sample, uploads output to S3.
    """
    s3 = boto3.client("s3")

    # Load configs
    cfg_path = "/root/configs.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Download trained model
    model_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, MODEL_OUTPUT_KEY, model_tmp)
    model_tmp.close()

    # Generate sample
    output = sample_from_checkpoint(cfg, model_tmp.name, text_prompt)

    # Save and upload output
    out_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(out_tmp, output)
    s3.upload_file(out_tmp, BUCKET, SAMPLE_OUTPUT_KEY)

    os.unlink(model_tmp.name)
    os.unlink(out_tmp)

    return {"status": "sampling_complete", "sample_s3_key": SAMPLE_OUTPUT_KEY}


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================
@app.local_entrypoint()
def main():
    print("Launching sampling on Modal GPU...")
    sample_res = sample_model.remote("example description")
    print("Sampling submitted, result handle:", sample_res)


if __name__ == "__main__":
    with app.run():
        main()
