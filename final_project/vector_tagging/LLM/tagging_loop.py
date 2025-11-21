import csv
import modal
import logging
import numpy as np
from vector_tagger import BioMistralVectorTagger

BUCKET_NAME = "healthforge-final-bucket"
S3_BATCH_PREFIX = "vector_batches/"
FINAL_S3_KEY = "merged_patient_vectors.npy"
VECTOR_DEFINITION_KEY = "patient_vector_columns.csv"
ORIGINAL_VECTOR_KEY = "original_vectors.npy"
VECTOR_TAG_KEY = "vector_tags.npy"
VECTOR_TAG_EMBEDDING_KEY = "vector_tag_embeddings.npy"
TEMP_DIR = "/tmp"

BATCH_SIZE = 512    # patients per GPU task
NUM_WORKERS = 8     # concurrent GPU workers

INFERENCE_BATCH_SIZE = 16
VECTORIZING_BATCH_SIZE = 128

app = modal.App("gpu-ehr-vector-tagging")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rds-parallel")


#Model
llm = None

# ---------------------------
# GPU batch worker
# ---------------------------
@gpu_batch_worker := app.function(
    gpu="H100",
    timeout=5 * 60 * 60,
    image=modal.Image.debian_slim().pip_install([
        "boto3", "numpy", "torch", "psycopg2-binary", "transformers", "accelerate"
    ]).add_local_file("vector_tagger.py", remote_path="/root/vector_tagger.py"),
    secrets=[
        modal.Secret.from_name("aws-secret")
    ],
)
def batch_worker(vector_batch: np.ndarray, batch_index: int, definition_array: np.ndarray):
    
    global llm
    if llm is None:
        logger.info("LLM not defined for this container, initiating LLM cold start...")
        llm = BioMistralVectorTagger(vector_definitions=definition_array)

    
    batches = [vector_batch[i:i+INFERENCE_BATCH_SIZE] for i in range(0, len(vector_batch), INFERENCE_BATCH_SIZE)]
    
    output = []
    for batch in batches:
        descriptions = llm.tag_vectors(batch)
        encodings = llm.encode_text([descriptions[i][1] for i in range(0, len(descriptions))])

        output.extend([[descriptions[i][0], descriptions[i][1], encodings[i]] for i in range(0, len(descriptions))])
 
    return {"index": batch_index, "result": output }




# ---------------------------
# Orchestrator
# ---------------------------
@app.function(
        timeout=5 * 60 * 60,
        image=modal.Image.debian_slim().pip_install(["psycopg2-binary", "boto3", "numpy", "torch", "transformers", "accelerate"]).add_local_file("vector_tagger.py", remote_path="/root/vector_tagger.py"),
         secrets=[
        modal.Secret.from_name("aws-secret")
    ],)

def tag_ehr_vectors():
    import boto3
    import os
    import time
    import numpy as np

    logger.info("Starting vector tagging orchestrator...")

    s3 = boto3.client("s3")
    local_training_data_path = os.path.join(TEMP_DIR, FINAL_S3_KEY)
    s3.download_file(BUCKET_NAME, FINAL_S3_KEY, local_training_data_path)

    local_vector_definition_path = os.path.join(TEMP_DIR, VECTOR_DEFINITION_KEY)
    s3.download_file(BUCKET_NAME, VECTOR_DEFINITION_KEY, local_vector_definition_path)

    all_training_vectors = np.load(local_training_data_path)
    
    
    with open(local_vector_definition_path, newline='') as f:
        reader = csv.reader(f)
        definition_vector = list(reader)[0]

    batches = [all_training_vectors[i:i+BATCH_SIZE] for i in range(0, 1)]

    calls = []
    results = []

    for idx, vector_batch in enumerate(batches):
        # Throttle to NUM_WORKERS concurrent calls
        while len(calls) >= NUM_WORKERS:
            for call in list(calls):
                try:
                    res = call.get(timeout=0)  # non-blocking check
                except TimeoutError:
                    continue  # still running
                else:
                    results.extend(res["result"])
                    calls.remove(call)

            if len(calls) >= NUM_WORKERS:
                time.sleep(0.5)

        # Spawn a new remote job – returns FunctionCall, not result
        call = batch_worker.spawn(vector_batch, idx, definition_vector)
        calls.append(call)

    # Drain remaining calls
    for call in calls:
        res = call.get()  # block until done
        results.extend(res["result"])

    original_vectors = np.array([results[i][0] for i in range(len(results))])
    print(original_vectors.shape)
    vector_tags = np.array([results[i][1] for i in range(len(results))])
    print(vector_tags.shape)
    vector_tag_embeddings = np.array([results[i][2] for i in range(len(results))])
    print(vector_tag_embeddings.shape)


    orig_local = os.path.join(TEMP_DIR, ORIGINAL_VECTOR_KEY)
    tag_local = os.path.join(TEMP_DIR, VECTOR_TAG_KEY)
    embed_local = os.path.join(TEMP_DIR, VECTOR_TAG_EMBEDDING_KEY)
    np.save(orig_local, original_vectors)
    np.save(tag_local, vector_tags)
    np.save(embed_local, vector_tag_embeddings)
    s3.upload_file(orig_local, BUCKET_NAME, ORIGINAL_VECTOR_KEY)
    s3.upload_file(tag_local, BUCKET_NAME, VECTOR_TAG_KEY)
    s3.upload_file(embed_local, BUCKET_NAME, VECTOR_TAG_EMBEDDING_KEY)

    print("Files uploaded to S3")


    


@app.local_entrypoint()
def main():
    tag_ehr_vectors.remote()


    



