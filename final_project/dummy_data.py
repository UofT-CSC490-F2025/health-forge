import modal
import boto3
import torch
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pickle
import tempfile
import os
from typing import List

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket"
PKL_KEY_PREFIX = "data/dummy_data_chunk_"
FINAL_PKL_KEY = "data/dummy_data_100k.pkl"
NUM_WORKERS = 8

app = modal.App("dummy-data-app")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "sentence-transformers",
        "boto3"
    ])
)

aws_secret = modal.Secret.from_name("aws-secret")

# ---------------------------
# GPU Worker for a chunk
# ---------------------------
@app.function(image=image, secrets=[aws_secret], gpu="H100", timeout=24*60*60)
def generate_chunk(start_idx: int, chunk_size: int, max_age: int = 100, log_every: int = 1000):
    print(f"[Worker] Processing samples {start_idx} → {start_idx+chunk_size}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Worker] Using device: {device}")

    # ---------------------------
    # Generate dummy samples
    # ---------------------------
    def create_dummy_samples(sample_size, max_age=100):
        print(f"[Samples] Generating {sample_size} dummy samples...")
        samples = []
        for i in range(sample_size):
            male = random.randint(0, 1)
            age = round(random.randrange(0, max_age) / max_age, 2)
            married = random.randint(0, 1)
            dead = random.randint(0, 1)
            diabetes = random.randint(0, 1)
            cancer = random.randint(0, 1)
            depression = random.randint(0, 1)
            schizophrenia = random.randint(0, 1)
            curr_sample = torch.tensor([male, age, married, dead, diabetes, cancer, depression, schizophrenia])
            samples.append(curr_sample)
            if (i + 1) % log_every == 0:
                print(f"[Samples] Generated {i+1}/{sample_size} samples")
        return torch.stack(samples, dim=0)

    # ---------------------------
    # Create textual descriptions
    # ---------------------------
    def data_descriptions(samples, max_age=100):
        print(f"[Descriptions] Generating textual descriptions...")
        vec_to_word = {
            0: {0: "is female", 1: "is male"},
            1: lambda x: f"is {int(x * max_age)} years old",
            2: {0: "is not married", 1: "is married"},
            3: {0: "is not dead", 1: "is dead"},
            4: {0: "does not have diabetes", 1: "has diabetes"},
            5: {0: "does not have cancer", 1: "has cancer"},
            6: {0: "does not have depression", 1: "has depression"},
            7: {0: "does not have schizophrenia", 1: "has schizophrenia"},
        }
        B, D = samples.shape
        descriptions = []
        for i in range(B):
            curr_sample = samples[i]
            curr_desc = []
            for j in range(D):
                if isinstance(vec_to_word[j], dict):
                    prop = curr_sample[j].item()
                    curr_desc.append(vec_to_word[j][prop])
                else:
                    curr_desc.append(vec_to_word[j](curr_sample[j]))
            descriptions.append(curr_desc)
            if (i + 1) % log_every == 0:
                print(f"[Descriptions] Created descriptions for {i+1}/{B} samples")
        return descriptions

    # ---------------------------
    # Batched LLM generation
    # ---------------------------
    def create_llm_descs(descs, batch_size=10):
        print(f"[LLM] Initializing text-generation pipeline on {device}...")
        pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=0)
        llm_descs = []
        total = len(descs)
        for i in range(0, total, batch_size):
            batch = descs[i:i+batch_size]
            messages = [
                [{"role": "user",
                  "content": f"Write a concise, direct English description with no filler of a person with the following properties: {', '.join(d)}"}]
                for d in batch
            ]
            print(f"[LLM] Generating batch {i//batch_size + 1} ({len(batch)} descriptions)...")
            results = pipe(messages)
            for j in range(len(results)):
                llm_descs.append(results[j][0]["generated_text"][1]["content"])
            print(f"[LLM] Generated {min(i+batch_size, total)}/{total} LLM descriptions")
        return llm_descs

    # ---------------------------
    # Sentence embeddings
    # ---------------------------
    def get_text_embeds(descs):
        print(f"[Embeddings] Encoding LLM descriptions into embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(descs, show_progress_bar=True)
        print(f"[Embeddings] Finished encoding {len(descs)} descriptions")
        return embeddings

    # ---------------------------
    # Run chunk
    # ---------------------------
    samples = create_dummy_samples(chunk_size, max_age)
    descs = data_descriptions(samples, max_age)
    llm_descs = create_llm_descs(descs)
    text_embeds = get_text_embeds(llm_descs)

    data = {
        "samples": samples,
        "descs": descs,
        "llm_descs": llm_descs,
        "text_embeds": text_embeds
    }

    # Save to temp and upload
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    with open(tmp_file.name, "wb") as f:
        pickle.dump(data, f)

    s3_key = f"{PKL_KEY_PREFIX}{start_idx}_{start_idx+chunk_size}.pkl"
    print(f"[S3] Uploading chunk to s3://{BUCKET}/{s3_key} ...")
    s3 = boto3.client("s3")
    s3.upload_file(tmp_file.name, BUCKET, s3_key)
    os.unlink(tmp_file.name)
    print(f"[S3] Uploaded chunk to s3://{BUCKET}/{s3_key}")

    return s3_key

# ---------------------------
# Orchestrator
# ---------------------------
@app.local_entrypoint()
def main():
    total_samples = 100000
    chunk_size = total_samples // NUM_WORKERS
    handles = []
    print(f"[Orchestrator] Launching {NUM_WORKERS} workers, chunk size={chunk_size}")

    for i in range(NUM_WORKERS):
        start_idx = i * chunk_size
        size = chunk_size if i < NUM_WORKERS-1 else total_samples - start_idx
        print(f"[Orchestrator] Spawning worker {i} for samples {start_idx} → {start_idx+size}")
        h = generate_chunk.spawn(start_idx, size)
        handles.append(h)

    # Wait for all workers and collect results
    results = []
    for i, h in enumerate(handles):
        print(f"[Orchestrator] Waiting for worker {i} to finish...")
        res = h.get()
        print(f"[Orchestrator] Worker {i} finished, S3 key: {res}")
        results.append(res)

    print("[Orchestrator] All chunks uploaded:", results)

    # ---------------------------
    # Merge all chunks into a final .pkl
    # ---------------------------
    def merge_chunks(s3_keys: list, final_s3_key: str):
        import pickle
        import tempfile
        import os
        import torch

        print(f"[Merger] Starting merge of {len(s3_keys)} chunks...")
        merged_data = {
            "samples": [],
            "descs": [],
            "llm_descs": [],
            "text_embeds": []
        }

        s3 = boto3.client("s3")

        for idx, key in enumerate(s3_keys):
            tmp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_file.close()
            print(f"[Merger] Downloading chunk {idx} from s3://{BUCKET}/{key} ...")
            s3.download_file(BUCKET, key, tmp_file.name)

            with open(tmp_file.name, "rb") as f:
                chunk_data = pickle.load(f)

            merged_data["samples"].append(chunk_data["samples"])
            merged_data["descs"].extend(chunk_data["descs"])
            merged_data["llm_descs"].extend(chunk_data["llm_descs"])
            merged_data["text_embeds"].append(chunk_data["text_embeds"])

            os.unlink(tmp_file.name)
            print(f"[Merger] Finished processing chunk {idx}")

        # Concatenate torch tensors
        merged_data["samples"] = torch.cat(merged_data["samples"], dim=0)
        merged_data["text_embeds"] = torch.cat(merged_data["text_embeds"], dim=0)

        # Save final merged .pkl
        tmp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp_file.close()
        with open(tmp_file.name, "wb") as f:
            pickle.dump(merged_data, f)

        s3.upload_file(tmp_file.name, BUCKET, final_s3_key)
        os.unlink(tmp_file.name)
        print(f"[Merger] Uploaded final merged file to s3://{BUCKET}/{final_s3_key}")

    merge_chunks(results, FINAL_PKL_KEY)
    print(f"[Orchestrator] Final merged .pkl available at s3://{BUCKET}/{FINAL_PKL_KEY}")


if __name__ == "__main__":
    with app.run():
        main()
