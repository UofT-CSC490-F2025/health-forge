import modal
import boto3
import psycopg2
import re
import logging
import os
import csv

# ---------------------------
# CONFIG
# ---------------------------
BUCKET_NAME = "healthforge-final-bucket"
FINAL_S3_KEY = "patient_vector_columns.csv"  # changed to CSV
TEMP_PATH = "/tmp/columns.csv"

# Create Modal app
app = modal.App("gpu-ehr-processing")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("column-export")

# ---------------------------
# Helper functions
# ---------------------------
def load_gem_from_s3(bucket, key):
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
    if code is None:
        return None
    return str(code).upper().replace('.', '').replace(' ', '')

def collapse_to_icd10_3(code_no_dot: str):
    if code_no_dot is None:
        return None
    s = str(code_no_dot)
    return s[:3] if len(s) > 3 else s

# ---------------------------
# Modal function
# ---------------------------
@app.function(
    timeout=60*60,
    image=modal.Image.debian_slim().pip_install(["boto3", "psycopg2-binary"]),
    secrets=[modal.Secret.from_name("aws-secret"), modal.Secret.from_name("postgres-secret")]
)
def export_column_labels():
    logger.info("Connecting to Postgres...")
    host = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
    port = 5432
    dbname = "raw_training_data"
    user = "healthforgedb"
    password = "mimicpassword"

    conn = psycopg2.connect(host=host, port=port, dbname=dbname,
                            user=user, password=password, sslmode="require")
    cur = conn.cursor()

    # ---------------------------
    # Basic fixed columns
    # ---------------------------
    columns = ["gender", "age", "dod"]

    # ---------------------------
    # Admissions columns
    # ---------------------------
    cur.execute("SELECT DISTINCT marital_status FROM admissions;")
    marital_statuses = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])
    cur.execute("SELECT DISTINCT race FROM admissions;")
    races = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])

    columns.append("total_admissions")
    columns += [f"marital_{m}" for m in marital_statuses]
    columns += [f"race_{r}" for r in races]

    # ---------------------------
    # Diagnoses columns
    # ---------------------------
    I9_KEY = "2018_I9gem.txt"
    I10_KEY = "2018_I10gem.txt"
    gem_i9_to_i10 = load_gem_from_s3(BUCKET_NAME, I9_KEY)

    cur.execute("SELECT DISTINCT icd_code, icd_version FROM diagnoses_icd WHERE icd_code IS NOT NULL;")
    all_codes = cur.fetchall()

    collapsed_set = set()
    for icd_raw, icd_version in all_codes:
        s = str(icd_raw).strip()
        if icd_version == 9 or str(icd_version) == '9':
            key = normalize_code_strip(s)
            mapped = gem_i9_to_i10.get(key, [])
            for tgt in mapped:
                collapsed_set.add(collapse_to_icd10_3(tgt))
        else:
            collapsed_set.add(collapse_to_icd10_3(normalize_code_strip(s)))

    collapsed_set.discard(None)
    collapsed_icd10_3_list = sorted([c for c in collapsed_set if c is not None])
    columns += [f"icd_{c}" for c in collapsed_icd10_3_list]

    # ---------------------------
    # Save as CSV instead of NPY
    # ---------------------------
    os.makedirs(os.path.dirname(TEMP_PATH), exist_ok=True)
    with open(TEMP_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

    s3 = boto3.client("s3")
    s3.upload_file(TEMP_PATH, BUCKET_NAME, FINAL_S3_KEY)
    logger.info(f"Uploaded column labels to s3://{BUCKET_NAME}/{FINAL_S3_KEY}")

    cur.close()
    conn.close()
    return {"s3_key": FINAL_S3_KEY, "num_columns": len(columns)}

# ---------------------------
# Local run
# ---------------------------
def main():
    with app.run():
        handle = export_column_labels.spawn()
        result = handle.get()
        print(result)

if __name__ == "__main__":
    main()
