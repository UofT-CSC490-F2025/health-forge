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
MODEL_OUTPUT_KEY = "checkpoints/best_diffusion_model_truedata_4096h_3l_GLU.pt"
PROMPT_FILE_KEY = "patient_text_prompts.txt"
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
    gpu="H100",
    timeout=4 * 60 * 60,
    image=image,
    secrets=[aws_secret],
)
def sample_model():
    s3 = boto3.client("s3")

    # Load configs
    cfg_path = "/root/configs.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Download trained model
    model_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, MODEL_OUTPUT_KEY, model_tmp)
    model_tmp.close()

    # Download text prompts
    prompt_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    s3.download_fileobj(BUCKET, PROMPT_FILE_KEY, prompt_tmp)
    prompt_tmp.close()

    with open(prompt_tmp.name, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Generate samples
    output = sample_from_checkpoint(cfg, model_tmp.name, prompts)

    # Save and upload output
    out_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(out_tmp, output)
    s3.upload_file(out_tmp, BUCKET, SAMPLE_OUTPUT_KEY)

    os.unlink(model_tmp.name)
    os.unlink(out_tmp)
    os.unlink(prompt_tmp.name)

    return {"status": "sampling_complete", "sample_s3_key": SAMPLE_OUTPUT_KEY}


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================
@app.local_entrypoint()
def main():
    print("Launching sampling on Modal GPU...")
    sample_res = sample_model.remote()
    print("Sampling submitted, result handle:", sample_res)


if __name__ == "__main__":
    with app.run():
        main()
