import modal
import boto3
import psycopg2
import os
import time
import logging
from typing import List
import numpy as np
import torch

# ---------------------------
# CONFIG
# ---------------------------
BUCKET_NAME = "healthforge-final-bucket"
S3_BATCH_PREFIX = "vector_batches/"
FINAL_S3_KEY = "patient_vectors.npy"
TEMP_DIR = "/tmp"

BATCH_SIZE = 256    # patients per GPU task
NUM_WORKERS = 1     # concurrent GPU workers

# ---------------------------
# Modal App
# ---------------------------
app = modal.App("gpu-ehr-processing")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rds-parallel")

# ---------------------------
# GPU batch worker
# ---------------------------
@gpu_batch_worker := app.function(
    gpu="A100",
    image=modal.Image.debian_slim().pip_install([
        "boto3", "numpy", "torch", "psycopg2-binary"
    ]),
    secrets=[
        modal.Secret.from_name("aws-secret"),       # AWS access key + secret
        modal.Secret.from_name("postgres-secret")   # RDS credentials
    ],
)
def batch_worker(subject_ids: List[int], batch_index: int):
    import psycopg2
    import torch
    import numpy as np
    import boto3
    print("Worker running on device:", torch.cuda.get_device_name(0))
    logger = logging.getLogger(f"batch-{batch_index}")
    logger.setLevel(logging.INFO)

    # -------------------------
    # RDS creds
    # -------------------------
    host = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
    port = 5432
    dbname = "raw_training_data"
    user = "healthforgedb"
    password = "mimicpassword"

    conn = psycopg2.connect(host=host, port=port, dbname=dbname,
                            user=user, password=password, sslmode="require")
    cur = conn.cursor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Worker {batch_index} running on {device}, processing {len(subject_ids)} patients")

    # -------------------------
    # Fetch all patient info in batch
    # -------------------------
    cur.execute(f"""
        SELECT subject_id, gender, anchor_age, dod
        FROM patients
        WHERE subject_id = ANY(%s)
    """, (subject_ids,))
    patient_rows = {r[0]: r[1:] for r in cur.fetchall()}

    # Admissions
    cur.execute(f"""
        SELECT subject_id, marital_status, race
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime DESC) as rn
            FROM admissions
            WHERE subject_id = ANY(%s)
        ) t
        WHERE rn = 1
    """, (subject_ids,))
    admissions_rows = {r[0]: r[1:] for r in cur.fetchall()}

    cur.execute(f"""
        SELECT subject_id, COUNT(*) as total
        FROM admissions
        WHERE subject_id = ANY(%s)
        GROUP BY subject_id
    """, (subject_ids,))
    total_admissions_map = {r[0]: r[1] for r in cur.fetchall()}

    # Diagnoses
    cur.execute(f"""
        SELECT subject_id, icd_code
        FROM diagnoses_icd
        WHERE subject_id = ANY(%s)
    """, (subject_ids,))
    diagnoses_map = {}
    for sid, icd in cur.fetchall():
        diagnoses_map.setdefault(sid, []).append(icd)

    # -------------------------
    # Build enums
    # -------------------------
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd WHERE icd_code IS NOT NULL;")
    icd_codes = sorted([r[0] for r in cur.fetchall()])
    icd_map = {v: i for i, v in enumerate(icd_codes)}

    cur.execute("SELECT DISTINCT marital_status FROM admissions;")
    marital_statuses = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])
    marital_map = {v: i for i, v in enumerate(marital_statuses)}

    cur.execute("SELECT DISTINCT race FROM admissions;")
    races = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])
    race_map = {v: i for i, v in enumerate(races)}

    # -------------------------
    # Construct vectors
    # -------------------------
    vectors = []
    for sid in subject_ids:
        gender_val, age_val, dod_val = patient_rows.get(sid, ("M", 0, False))
        gender_tensor = torch.tensor([0 if gender_val == 'M' else 1], dtype=torch.float32)
        age_tensor = torch.tensor([age_val if age_val is not None else 0.0], dtype=torch.float32)
        dod_tensor = torch.tensor([1 if dod_val else 0], dtype=torch.float32)

        marital_status, race = admissions_rows.get(sid, ("N/A", "N/A"))
        total_adm = total_admissions_map.get(sid, 0)

        marital_vec = torch.zeros(len(marital_map), dtype=torch.float32)
        race_vec = torch.zeros(len(race_map), dtype=torch.float32)
        marital_vec[marital_map.get(marital_status, 0)] = 1.0
        race_vec[race_map.get(race, 0)] = 1.0

        admissions_vector = torch.cat([torch.tensor([total_adm], dtype=torch.float32),
                                       marital_vec, race_vec])

        diagnoses_vec = torch.zeros(len(icd_map), dtype=torch.float32)
        for code in diagnoses_map.get(sid, []):
            if code in icd_map:
                diagnoses_vec[icd_map[code]] = 1.0

        final_tensor = torch.cat([gender_tensor, age_tensor, dod_tensor, admissions_vector, diagnoses_vec])
        vectors.append(final_tensor)

    # Move whole batch to GPU at once
    batch_tensor = torch.stack(vectors).to(device)
    arr = batch_tensor.cpu().numpy().astype("float32")

    # Save and upload to S3
    local_path = f"/tmp/batch_{batch_index}.npy"
    np.save(local_path, arr)
    s3 = boto3.client("s3")
    s3_key = f"{S3_BATCH_PREFIX}batch_{batch_index}.npy"
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    logger.info(f"Uploaded batch {batch_index} to s3://{BUCKET_NAME}/{s3_key}")

    cur.close()
    conn.close()

    return {"batch_index": batch_index, "s3_key": s3_key, "num_subjects": len(subject_ids)}


# ---------------------------
# Orchestrator
# ---------------------------
@app.function(image=modal.Image.debian_slim().pip_install(["psycopg2-binary", "boto3", "numpy", "torch"]))
def process_ehr_data():
    import psycopg2
    import boto3
    import os
    import time
    import numpy as np

    logger.info("Starting orchestrator...")

    host = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
    port = 5432
    dbname = "raw_training_data"
    user = "healthforgedb"
    password = "mimicpassword"

    conn = psycopg2.connect(host=host, port=port, dbname=dbname,
                            user=user, password=password, sslmode="require")
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT subject_id FROM patients ORDER BY subject_id;")
    subject_ids = [r[0] for r in cur.fetchall()]
    total = len(subject_ids)

    logger.info(f"Total patients={total}, batch_size={BATCH_SIZE}, num_batches={(total + BATCH_SIZE - 1)//BATCH_SIZE}")

    batches = [subject_ids[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    # Launch workers in parallel (limit concurrent to NUM_WORKERS)
    handles = []

    for idx, batch in enumerate(batches):
        # throttle so we don't launch more than NUM_WORKERS concurrently
        while len(handles) >= NUM_WORKERS:
            for h in handles:
                if h.ready():       # check if the future is done
                    handles.remove(h)
                    break
            else:
                time.sleep(0.5)  # wait a bit before checking again

        res = batch_worker.remote(batch, idx)   # directly get the dict
        handles.append(res)
        logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")

    # Wait for all remaining workers
    for h in handles:
        res = h.wait()  # now this works
        logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")


    # Aggregate final arrays
    s3 = boto3.client("s3")
    arrays = []
    for idx in range(len(batches)):
        local_path = os.path.join(TEMP_DIR, f"agg_batch_{idx}.npy")
        key = f"{S3_BATCH_PREFIX}batch_{idx}.npy"
        s3.download_file(BUCKET_NAME, key, local_path)
        arrays.append(np.load(local_path))

    final_arr = np.vstack(arrays) if arrays else np.zeros((0, len(arrays[0][0])))
    final_local = os.path.join(TEMP_DIR, FINAL_S3_KEY)
    np.save(final_local, final_arr)
    s3.upload_file(final_local, BUCKET_NAME, FINAL_S3_KEY)
    logger.info(f"Uploaded final vectors to s3://{BUCKET_NAME}/{FINAL_S3_KEY}")

    cur.close()
    conn.close()
    return {"num_patients": total, "num_batches": len(batches), "final_s3_key": FINAL_S3_KEY}


# ---------------------------
# Local entrypoint
# ---------------------------
@app.local_entrypoint()
def main():
    import math
    import time
    import numpy as np
    import boto3
    import logging
    import psycopg2
    import os

    logger = logging.getLogger("orchestrator")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ---------------------------
    # Postgres credentials
    # ---------------------------
    host = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
    port = 5432
    dbname = "raw_training_data"
    user = "healthforgedb"
    password = "mimicpassword"

    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password, sslmode="require")
    cur = conn.cursor()

    # fetch all patient IDs
    cur.execute("SELECT DISTINCT subject_id FROM patients ORDER BY subject_id;")
    subject_rows = cur.fetchall()
    subject_ids = [r[0] for r in subject_rows]
    total = len(subject_ids)
    if total == 0:
        logger.warning("No patients found.")
        return

    batch_size = 256  # or tune
    NUM_WORKERS = 4   # max number of parallel GPU workers
    batches = [subject_ids[i:i+batch_size] for i in range(0, total, batch_size)]
    logger.info(f"Total patients={total}, batch_size={batch_size}, num_batches={len(batches)}")

    # ---------------------------
    # Launch workers in parallel using submit()
    # ---------------------------
    handles = []
    s3 = boto3.client("s3")

    for idx, batch in enumerate(batches):
        # throttle parallel workers
        while len(handles) >= NUM_WORKERS:
            # wait for at least one to finish
            done_handles = [h for h in handles if h.done()]
            for dh in done_handles:
                res = dh.result()
                logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")
                handles.remove(dh)
            time.sleep(0.5)

        # submit a new worker
        h = batch_worker.submit(batch, idx)
        handles.append(h)
        logger.info(f"Launched batch {idx} ({len(batch)} subjects)")

    # wait for remaining workers
    for h in handles:
        res = h.result()
        logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")

    # ---------------------------
    # Aggregate batch files
    # ---------------------------
    s3 = boto3.client("s3")
    local_files = []
    for idx in range(len(batches)):
        key = f"{S3_BATCH_PREFIX}batch_{idx}.npy"
        local_path = os.path.join(TEMP_DIR, f"agg_batch_{idx}.npy")
        logger.info(f"Downloading s3://{BUCKET_NAME}/{key} -> {local_path}")
        s3.download_file(BUCKET_NAME, key, local_path)
        local_files.append(local_path)

    # load and concatenate arrays
    arrays = [np.load(p) for p in local_files]

    if arrays:
        final_arr = np.vstack(arrays)  # stack vertically to get full patient x vector_dim array
    else:
        # fallback: create an empty array with the expected vector dimension
        # You can compute the vector dimension from your enums if needed
        arrays = [np.load(p) for p in local_files]
            
        if arrays:
            final_arr = np.vstack(arrays)
        else:
            # fallback: assume a default vector_dim if no batches (rare)
            vector_dim = 100  # replace with the dimension you expect per patient
            final_arr = np.zeros((0, vector_dim), dtype=np.float32)
        final_arr = np.zeros((0, vector_dim), dtype=np.float32)

    # save final numpy array and upload
    final_local = os.path.join(TEMP_DIR, "patient_vectors.npy")
    np.save(final_local, final_arr)
    s3.upload_file(final_local, BUCKET_NAME, FINAL_S3_KEY)
    logger.info(f"Uploaded final vectors to s3://{BUCKET_NAME}/{FINAL_S3_KEY}")

    cur.close()
    conn.close()
    logger.info("Processing complete.")

if __name__ == "__main__":
    with app.run():
        main()
