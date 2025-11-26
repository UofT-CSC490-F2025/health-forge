import modal
import boto3
import psycopg2
import os
import time
import logging
from typing import List
import numpy as np
import torch
import re
import csv

# ---------------------------
# CONFIG
# ---------------------------
BUCKET_NAME = "healthforge-final-bucket"
S3_BATCH_PREFIX = "vector_batches/"
FINAL_S3_KEY = "patient_vectors.npy"
TEMP_DIR = "/tmp"

BATCH_SIZE = 512    # patients per GPU task
NUM_WORKERS = 8     # concurrent GPU workers


# ---------------------------
# Modal App
# ---------------------------
app = modal.App("gpu-ehr-processing")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rds-parallel")


# ---------------------------
# Helper functions for GEM loading & ICD normalization
# ---------------------------
def load_gem_from_s3(bucket, key):
    """Load GEM mapping file directly from S3."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8", errors="ignore")

    mapping = {}
    reg = re.compile(r'(\S+)\s+(\S+)\s+(\S+)')

    for line in text.splitlines():
        m = reg.search(line.strip())
        if not m:
            continue
        src, tgt, flag = m.groups()
        src_n = src.upper().replace(".", "").replace(" ", "")
        tgt_n = tgt.upper().replace(".", "").replace(" ", "")
        mapping.setdefault(src_n, set()).add(tgt_n)

    return {k: list(v) for k, v in mapping.items()}



def normalize_code_strip(code: str):
    """Return uppercased code with dot/spaces removed (safe for lookup)."""
    if code is None:
        return None
    return str(code).upper().replace('.', '').replace(' ', '')


def collapse_to_icd10_3(code_no_dot: str):
    """
    Given an ICD10 code without dot (e.g., 'E1165' or 'E11'), return the 3-character category:
      - take first 3 characters (letter + 2 digits), i.e. 'E11'
    If code_no_dot is shorter than 3, return it as-is.
    """
    if code_no_dot is None:
        return None
    s = str(code_no_dot)
    if len(s) <= 3:
        return s
    return s[:3]


# ---------------------------
# GPU batch worker
# ---------------------------
@gpu_batch_worker := app.function(
    gpu="H100",
    timeout=5 * 60 * 60,
    image=modal.Image.debian_slim().pip_install([
        "boto3", "numpy", "torch", "psycopg2-binary"
    ]),
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("postgres-secret")
    ],
)
def batch_worker(subject_ids: List[int], batch_index: int):
    import psycopg2
    import torch
    import numpy as np
    import boto3
    import logging

    # Load GEMs from S3
    I9_KEY = "2018_I9gem.txt"      # update to your actual key
    I10_KEY = "2018_I10gem.txt"   # update to your actual key

    gem_i9_to_i10 = load_gem_from_s3(BUCKET_NAME, I9_KEY)
    gem_i10_to_i9 = load_gem_from_s3(BUCKET_NAME, I10_KEY)


    print("Worker running on device:", torch.cuda.get_device_name(0))
    logger = logging.getLogger(f"batch-{batch_index}")
    logger.setLevel(logging.INFO)

    # -------------------------
    # RDS creds (keep your originals)
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

    # -------------------------
    # Diagnoses: fetch icd_code + icd_version
    # and convert ICD-9 -> ICD-10, collapse to 3-char ICD-10 categories
    # -------------------------
    cur.execute(f"""
        SELECT subject_id, icd_code, icd_version
        FROM diagnoses_icd
        WHERE subject_id = ANY(%s)
    """, (subject_ids,))

    diagnoses_map = {}  # subject_id -> set of collapsed 3-digit ICD10 codes
    for sid, icd_raw, icd_version in cur.fetchall():
        if icd_raw is None:
            continue
        # Normalize (strip spaces)
        code_stripped = str(icd_raw).strip()
        # Handle ICD-9 input: convert to ICD-10 via GEMs (include all)
        if icd_version == 9 or str(icd_version) == '9':
            key = normalize_code_strip(code_stripped)  # ICD9 key without dots
            mapped = gem_i9_to_i10.get(key, [])
            # for each mapped ICD10 target, collapse to 3-digit
            for tgt in mapped:
                collapsed = collapse_to_icd10_3(tgt)  # tgt is already no-dot in loader
                if collapsed:
                    diagnoses_map.setdefault(sid, set()).add(collapsed)
            # if there were no GEM mappings, we skip (you could alternatively log)
        else:
            # icd_version == 10 -> normalize and collapse
            tgt = normalize_code_strip(code_stripped)  # removes dot
            collapsed = collapse_to_icd10_3(tgt)
            if collapsed:
                diagnoses_map.setdefault(sid, set()).add(collapsed)

    # convert sets to lists for stable iteration
    for k in list(diagnoses_map.keys()):
        diagnoses_map[k] = list(diagnoses_map[k])

    # -------------------------
    # Build enums
    # -------------------------
    # We must build icd_map (columns) from all DISTINCT collapsed 3-digit ICD10 categories across the ENTIRE DB,
    # not just the current batch, to ensure column indices are consistent between batches.
    # So query full table of distinct icd_code + icd_version and convert them the same way.
    cur.execute("SELECT DISTINCT icd_code, icd_version FROM diagnoses_icd WHERE icd_code IS NOT NULL;")
    all_codes = cur.fetchall()

    collapsed_set = set()
    for icd_raw, icd_version in all_codes:
        if icd_raw is None:
            continue
        s = str(icd_raw).strip()
        if icd_version == 9 or str(icd_version) == '9':
            key = normalize_code_strip(s)
            mapped = gem_i9_to_i10.get(key, [])
            for tgt in mapped:
                collapsed_set.add(collapse_to_icd10_3(tgt))
        else:
            tgt = normalize_code_strip(s)
            collapsed_set.add(collapse_to_icd10_3(tgt))

    # Remove None if any slipped in
    collapsed_set.discard(None)
    # Sort for deterministic ordering
    collapsed_icd10_3_list = sorted([c for c in collapsed_set if c is not None])

    icd_map = {v: i for i, v in enumerate(collapsed_icd10_3_list)}

    # -------------------------
    # Other enums unchanged
    # -------------------------
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
        # Now diagnoses_map contains collapsed ICD-10 3-digit codes (strings)
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

    # batch to resume from
    START_BATCH = 378  # change this to whatever batch you want to start from

    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password, sslmode="require")
    cur = conn.cursor()

    # fetch all patient IDs
    cur.execute("SELECT DISTINCT subject_id FROM patients ORDER BY subject_id;")
    subject_rows = cur.fetchall()
    cur.close()
    conn.close()
    subject_ids = [r[0] for r in subject_rows]
    total = len(subject_ids)
    if total == 0:
        logger.warning("No patients found.")
        return

    batch_size = 512
    NUM_WORKERS = 8
    batches = [subject_ids[i:i+batch_size] for i in range(0, total, batch_size)]
    logger.info(f"Total patients={total}, batch_size={batch_size}, num_batches={len(batches)}")

    # ---------------------------
    # Launch workers from START_BATCH
    # ---------------------------
    handles = []
    for idx in range(START_BATCH, len(batches)):
        batch = batches[idx]

        # throttle workers
        while len(handles) >= NUM_WORKERS:
            done = []
            still_running = []

            for h in handles:
                try:
                    res = h.get(timeout=0)
                    done.append((h, res))
                except TimeoutError:
                    still_running.append(h)
                except Exception as e:
                    logger.error(f"Unexpected error in get(timeout=0): {e}")
                    still_running.append(h)

            for h, res in done:
                logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")

            handles = still_running
            if len(done) == 0:
                time.sleep(0.3)

        h = batch_worker.spawn(batch, idx)
        handles.append(h)
        logger.info(f"Launched batch {idx} ({len(batch)} subjects)")

    # wait for remaining workers
    for h in handles:
        res = h.get()
        logger.info(f"Batch {res['batch_index']} finished, s3_key={res['s3_key']}")

    cur.close()
    conn.close()
    logger.info("Processing complete.")



if __name__ == "__main__":
    with app.run():
        main()