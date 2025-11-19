import modal
import boto3
import psycopg2
import re
import logging
import os
import csv
import io
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
BUCKET_NAME = "healthforge-final-bucket"
FINAL_S3_KEY = "patient_vector_columns.csv"
TEMP_PATH = "/tmp/columns.csv"

DXCCSR_CSV_KEY = "DXCCSR-Reference.csv"


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
    return str(code).upper().replace(".", "").replace(" ", "")


def collapse_to_icd10_3(code_no_dot: str):
    if code_no_dot is None:
        return None
    s = str(code_no_dot)
    return s[:3] if len(s) >= 3 else None

def sanitize_description(desc: str):
    """Convert human text into a safe column name."""
    desc = desc.lower()
    desc = re.sub(r"[^a-z0-9]+", "_", desc)
    desc = re.sub(r"_+", "_", desc)
    return desc.strip("_")


def load_dxccsr_mapping():
    """Loads DXCCSR reference CSV from S3 and returns mapping:
       3-digit ICD → CCSR Category Description"""
    logger.info("Loading DXCCSR CSV file from S3...")

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=DXCCSR_CSV_KEY)
    data = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(io.StringIO(data))

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # REQUIRED columns (you confirmed these exist in your CSV)
    required_cols = ["ICD-10-CM Code", "CCSR Category Description"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DXCCSR CSV missing required column: {col}")

    mapping = {}

    for full_icd, desc in zip(df["ICD-10-CM Code"], df["CCSR Category Description"]):
        icd_clean = normalize_code_strip(full_icd)
        prefix = collapse_to_icd10_3(icd_clean)

        if prefix and prefix not in mapping:
            mapping[prefix] = str(desc)

    logger.info(f"DXCCSR mapping loaded: {len(mapping)} prefixes mapped.")
    return mapping




# ---------------------------
# Modal function
# ---------------------------
@app.function(
    timeout=60 * 60,
    image=modal.Image.debian_slim().pip_install(
        ["boto3", "psycopg2-binary", "pandas"]
    ),
    secrets=[modal.Secret.from_name("aws-secret"),
             modal.Secret.from_name("postgres-secret")]
)
def export_column_labels():

    logger.info("Connecting to Postgres...")

    host = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
    port = 5432
    dbname = "raw_training_data"
    user = "healthforgedb"
    password = "mimicpassword"

    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password, sslmode="require"
    )
    cur = conn.cursor()

    # ---------------------------
    # Load DXCCSR (Option A)
    # ---------------------------
    dxccsr_map = load_dxccsr_mapping()

    # ---------------------------
    # Basic fixed columns
    # ---------------------------
    columns = ["isMale", "age", "isDead"]

    # ---------------------------
    # Admissions columns
    # ---------------------------
    cur.execute("SELECT DISTINCT marital_status FROM admissions;")
    marital_statuses = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])

    cur.execute("SELECT DISTINCT race FROM admissions;")
    races = sorted([r[0] if r[0] else "N/A" for r in cur.fetchall()])

    # rename total admissions
    columns.append("total admission number")

    # marital: remove prefix, keep underscores
    columns += [sanitize_description(m) for m in marital_statuses]

    # race: remove prefix, keep underscores
    columns += [sanitize_description(r) for r in races]

    # ---------------------------
    # Diagnoses columns: keep ICD prefix + description
    # ---------------------------
    I9_KEY = "2018_I9gem.txt"
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
                pref = collapse_to_icd10_3(tgt)
                if pref:
                    collapsed_set.add(pref)
        else:
            pref = collapse_to_icd10_3(normalize_code_strip(s))
            if pref:
                collapsed_set.add(pref)

    collapsed_set.discard(None)

    # For each 3-digit ICD prefix, produce a column named:
    # <PREFIX>_<sanitized-description>  (e.g. "A01_acute_myocardial_infarction")
    for prefix in sorted(collapsed_set):
        if prefix in dxccsr_map:
            desc = dxccsr_map[prefix]
            clean_desc = sanitize_description(desc).replace("_", " ")
            columns.append(clean_desc)
        else:
            columns.append("unknown")



    # ---------------------------
    # Save to CSV
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
