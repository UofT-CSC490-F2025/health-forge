import modal
import boto3
import tempfile
import pickle
import os
import torch
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket"
PKL_KEY_PREFIX = "data/dummy_data_chunk_"   # prefix of chunk files
FINAL_MERGED_KEY = "data/final_merged.pkl"

app = modal.App("merge-only-app")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "boto3",
        "numpy"
    ])
)

aws_secret = modal.Secret.from_name("aws-secret")


# ---------------------------
# MERGE FUNCTION — STANDALONE
# ---------------------------
@app.function(
    image=image,
    secrets=[aws_secret],
    timeout=60 * 30,   # 30 minutes
)
def merge_s3_chunks(bucket: str, prefix: str, output_key: str):

    print("============================================")
    print("  STARTING S3 MERGE JOB")
    print("============================================")

    s3 = boto3.client("s3")

    # List all keys under prefix
    print(f"[MERGE] Listing S3 keys with prefix: '{prefix}'")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in resp:
        raise RuntimeError(f"No files found under prefix: {prefix}")

    keys = [obj["Key"] for obj in resp["Contents"]]
    print(f"[MERGE] Found {len(keys)} chunk files:")
    for k in keys:
        print("   -", k)

    # Prepare merge containers
    merged = {
        "samples": [],
        "descs": [],
        "llm_descs": [],
        "text_embeds": []
    }

    # Download and merge one by one
    for key in keys:
        print(f"\n[MERGE] Downloading {key}")
        tmp = tempfile.NamedTemporaryFile(delete=False)
        s3.download_file(bucket, key, tmp.name)

        with open(tmp.name, "rb") as f:
            data = pickle.load(f)

        print(f"[MERGE] Extending data from {key}")

        # Merge samples tensor
        merged["samples"].append(data["samples"])

        # Merge description lists
        merged["descs"].extend(data["descs"])
        merged["llm_descs"].extend(data["llm_descs"])

        # Merge embeddings
        # They might be list-of-lists or numpy array or tensor
        embeds = data["text_embeds"]

        if isinstance(embeds, torch.Tensor):
            merged["text_embeds"].append(embeds)
        else:
            merged["text_embeds"].append(torch.tensor(embeds))

        os.unlink(tmp.name)

    print("\n[MERGE] Concatenating final tensors…")

    merged["samples"] = torch.cat(merged["samples"], dim=0)
    merged["text_embeds"] = torch.cat(merged["text_embeds"], dim=0)

    print("[MERGE] Final shapes:")
    print("  samples:     ", merged["samples"].shape)
    print("  text_embeds: ", merged["text_embeds"].shape)
    print("  descs:       ", len(merged["descs"]))
    print("  llm_descs:   ", len(merged["llm_descs"]))

    # Save merged file
    tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    with open(tmp_out.name, "wb") as f:
        pickle.dump(merged, f)

    print(f"[MERGE] Uploading final merged file → s3://{bucket}/{output_key}")
    s3.upload_file(tmp_out.name, bucket, output_key)

    os.unlink(tmp_out.name)

    print("\n============================================")
    print("  MERGE COMPLETE →", output_key)
    print("============================================")

    return output_key


# ---------------------------
# ENTRYPOINT FOR MANUAL EXECUTION
# ---------------------------
@app.local_entrypoint()
def main():
    merge_s3_chunks.remote(BUCKET, PKL_KEY_PREFIX, FINAL_MERGED_KEY)
    print("Done.")


if __name__ == "__main__":
    with app.run():
        main()