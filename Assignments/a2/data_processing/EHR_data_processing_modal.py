import modal
import boto3
import sqlite3
import numpy as np
import tempfile
import os
import re
from pathlib import Path
import torch

# ---------------------------
# AWS S3 CONFIG
# ---------------------------
BUCKET_NAME = "health-forge-data-processing"
INPUT_DB_FILENAME = "MIMIC_IV.sqlite"

# ---------------------------
# CREATE MODAL APP
# ---------------------------
app = modal.App("gpu-ehr-processing")  # App name



# ---------------------------
# DEFINE GPU FUNCTION WITH SECRET
# ---------------------------
@gpu_function := app.function(
    gpu="A100",
    image=modal.Image.debian_slim().pip_install(["boto3", "numpy", "pandas", "torch"]),
    secrets=[modal.Secret.from_name("aws-secret")],  # use the secret you created
)
def process_ehr_data():
    """
    Downloads EHR database from S3 and processes it with merge_database_gpu() using PyTorch.
    """

    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp_db_file:
        print(f"Downloading {INPUT_DB_FILENAME} from S3 bucket {BUCKET_NAME}...")
        s3.download_file(BUCKET_NAME, INPUT_DB_FILENAME, tmp_db_file.name)
        print(f"Downloaded to temporary path: {tmp_db_file.name}")

        # Call your GPU processing function
        merge_database_gpu(tmp_db_file.name)

        # Upload processed vector store
        s3.upload_file("/tmp/vector_store.sqlite", BUCKET_NAME, "vector_store.sqlite")
        print("Uploaded processed vector store to S3")

# ---------------------------
# DATABASE PROCESSING LOGIC
# ---------------------------
def merge_database_gpu(db_input_path: str):
    """
    GPU-accelerated version of merge_database using PyTorch.
    Converts all CuPy operations to torch tensors for GPU computation.
    """

    output_db_path = "/tmp/vector_store.sqlite"
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_conn = sqlite3.connect(db_input_path)
    input_conn.row_factory = sqlite3.Row
    output_conn = sqlite3.connect(output_db_path)
    input_cur = input_conn.cursor()
    output_cur = output_conn.cursor()

    # --- CREATE vectors TABLE ---
    output_cur.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        subject_id INTEGER PRIMARY KEY,
        vec BLOB
    )
    """)
    output_conn.commit()

    enums = get_enums_gpu(input_conn)  # You will also need to rewrite get_enums_gpu with torch

    print("Starting GPU-based merge process...")

    for patient in input_cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]

        # --- BASIC PATIENT FEATURES (torch tensors on GPU) ---
        subject_id_tensor = torch.tensor([subject_id], dtype=torch.float32, device=device)
        gender = torch.tensor([0 if patient["gender"] == 'M' else 1], dtype=torch.float32, device=device)
        age = torch.tensor([patient["anchor_age"]], dtype=torch.float32, device=device)
        dod = torch.tensor([1 if patient["dod"] else 0], dtype=torch.float32, device=device)

        # --- Retrieve and compute all patient-related vectors (torch tensors) ---
        gsn_vector = get_prescriptions_gpu(input_conn, subject_id, enums["prescriptions"]).to(device)
        icd_codes_vector = get_diagnoses_icd_gpu(input_conn, subject_id, enums["diagnoses_icd"]).to(device)
        proc_vector = get_procedures_icd_gpu(input_conn, subject_id, enums["procedures_icd"]["icd_codes"]).to(device)
        poe_vector = get_poe_gpu(input_conn, subject_id, enums["poe"]).to(device)
        svc_vector = get_services_gpu(input_conn, subject_id, enums["services"]).to(device)

        total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector = \
            get_admissions_gpu(input_conn, subject_id, enums["admissions"])

        admission_vector = torch.cat((
            torch.tensor([total_admissions], dtype=torch.float32, device=device),
            admission_type_vector.to(device).float(),
            admission_location_vector.to(device).float(),
            discharge_location_vector.to(device).float()
        ))

        result_mapping, vector_length = get_omr_result_names_gpu(input_conn)
        hcpcs_map = get_all_hcpcs_codes_gpu(input_conn)
        medication_map = get_all_drugs_gpu(input_conn)

        omr_vector = get_omr_gpu(input_conn, subject_id, result_mapping, vector_length).to(device)
        hcpcs_vector = get_hcpcsevents_gpu(input_conn, subject_id, hcpcs_map).to(device)
        pharm_vector = get_pharmacy_gpu(input_conn, subject_id, medication_map).to(device)

        # --- Concatenate everything on GPU ---
        final_tensor = torch.cat((
            subject_id_tensor,
            gender,
            age,
            dod,
            gsn_vector,
            icd_codes_vector,
            proc_vector,
            poe_vector,
            svc_vector,
            admission_vector,
            omr_vector,
            hcpcs_vector,
            pharm_vector
        ))

        # --- Move to CPU for database write ---
        vector_blob = final_tensor.cpu().numpy().astype("float32").tobytes()
        output_cur.execute(
            "INSERT INTO vectors (subject_id, vec) VALUES (?, ?);",
            (subject_id, vector_blob)
        )

    output_conn.commit()
    output_conn.close()
    input_conn.close()
    print(f"✅ GPU vector store created at {output_db_path}")

# ---------------------------
# ALL OTHER HELPER FUNCTIONS (get_enums, get_poe, get_prescriptions, etc.)
# Copy your existing helper functions here, unchanged
# ---------------------------
def get_enums_gpu(db_conn: sqlite3.Connection) -> dict:
    print("Starting to extract enums from database (GPU-ready)...")
    cur = db_conn.cursor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enums = {
        "prescriptions": {},
        "diagnoses_icd": {},
        "poe": {},
        "procedures_icd": {},
        "services": {},
        "admissions": {}
    }

    # --- prescriptions ---
    print("Processing prescriptions...")
    cur.execute("SELECT DISTINCT gsn FROM prescriptions;")
    gsn_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["prescriptions"]["gsn"] = {v: i for i, v in enumerate(gsn_codes)}
    enums["prescriptions"]["gsn_gpu"] = torch.arange(len(gsn_codes), dtype=torch.int32, device=device)

    # --- diagnoses ICD codes ---
    print("Processing diagnoses ICD codes...")
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["diagnoses_icd"]["icd_codes"] = {v: i for i, v in enumerate(icd_codes)}
    enums["diagnoses_icd"]["icd_codes_gpu"] = torch.arange(len(icd_codes), dtype=torch.int32, device=device)

    # --- POE (orders) ---
    print("Processing POE (orders)...")
    cur.execute("SELECT DISTINCT order_type FROM poe;")
    order_types = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["poe"]["order_type"] = {v: i for i, v in enumerate(order_types)}
    enums["poe"]["order_type_gpu"] = torch.arange(len(order_types), dtype=torch.int32, device=device)

    # --- procedures ICD ---
    print("Processing procedures ICD...")
    cur.execute("SELECT DISTINCT icd_code, icd_version FROM procedures_icd;")
    proc_codes = cur.fetchall()
    proc_norm = [f"{version}_{(code.replace('.', ''))}" for code, version in proc_codes if code is not None]
    enums["procedures_icd"]["icd_codes"] = {v: i for i, v in enumerate(proc_norm)}
    enums["procedures_icd"]["icd_codes_gpu"] = torch.arange(len(proc_norm), dtype=torch.int32, device=device)

    # --- services ---
    print("Processing services...")
    cur.execute("SELECT DISTINCT curr_service FROM services;")
    services = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["services"]["curr_service"] = {v: i for i, v in enumerate(services)}
    enums["services"]["curr_service_gpu"] = torch.arange(len(services), dtype=torch.int32, device=device)

    # --- admissions ---
    print("Processing admissions...")
    admission_types = [
        'AMBULATORY OBSERVATION', 'DIRECT EMER.', 'DIRECT OBSERVATION',
        'ELECTIVE', 'EU OBSERVATION', 'EW EMER.',
        'OBSERVATION ADMIT', 'SURGICAL SAME DAY ADMISSION', 'URGENT'
    ]
    admission_type_map = {t: i for i, t in enumerate(admission_types)}

    cur.execute("SELECT DISTINCT admission_location FROM admissions;")
    admission_locations = [row[0] for row in cur.fetchall()]
    admission_location_map = {loc: i for i, loc in enumerate(admission_locations)}

    cur.execute("SELECT DISTINCT discharge_location FROM admissions;")
    discharge_locations = [row[0] for row in cur.fetchall()]
    discharge_location_map = {loc: i for i, loc in enumerate(discharge_locations)}

    enums["admissions"]["admission_type"] = admission_type_map
    enums["admissions"]["admission_location"] = admission_location_map
    enums["admissions"]["discharge_location"] = discharge_location_map
    enums["admissions"]["admission_type_gpu"] = torch.arange(len(admission_types), dtype=torch.int32, device=device)
    enums["admissions"]["admission_location_gpu"] = torch.arange(len(admission_locations), dtype=torch.int32, device=device)
    enums["admissions"]["discharge_location_gpu"] = torch.arange(len(discharge_locations), dtype=torch.int32, device=device)

    cur.close()
    print("Finished extracting enums (GPU-ready).")
    return enums


def get_poe_gpu(db_conn: sqlite3.Connection, subject_id: int, poe_enum: dict) -> torch.Tensor:
    print(f"Processing POE for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    query = f"SELECT order_type FROM poe WHERE subject_id={subject_id};"
    order_type_enum = poe_enum['order_type']
    n = len(order_type_enum)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poe_vector = torch.zeros(n, dtype=torch.int32, device=device)

    order_types = [row[0] for row in cur.execute(query)]
    cur.close()

    indices = [order_type_enum[o] for o in order_types if o in order_type_enum]
    if indices:
        poe_vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device), torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished POE for subject_id={subject_id}")
    return poe_vector


def get_prescriptions_gpu(db_conn: sqlite3.Connection, subject_id: int, prescription_enums: dict) -> torch.Tensor:
    print(f"Processing prescriptions for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    query = f"SELECT gsn FROM prescriptions WHERE subject_id={subject_id};"
    gsn_enum = prescription_enums['gsn']
    n = len(gsn_enum)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsn_vector = torch.zeros(n, dtype=torch.int32, device=device)

    gsns = [row[0] for row in cur.execute(query)]
    cur.close()

    indices = [gsn_enum[g] for g in gsns if g in gsn_enum]
    if indices:
        gsn_vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device), torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished prescriptions for subject_id={subject_id}")
    return gsn_vector


import torch
import re
import sqlite3

def get_procedures_icd_gpu(db_conn: sqlite3.Connection, subject_id: int, proc_enums: dict) -> torch.Tensor:
    print(f"Processing procedures ICD for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    cur.execute("SELECT icd_code, icd_version FROM procedures_icd WHERE subject_id=?;", (subject_id,))
    proc_records = cur.fetchall()
    cur.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(proc_enums)
    vector = torch.zeros(n, dtype=torch.int32, device=device)

    indices = []
    for code, version in proc_records:
        if code:
            norm_code = f"{version}_{code.replace('.', '')}"
            if norm_code in proc_enums:
                indices.append(proc_enums[norm_code])

    if indices:
        vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                          torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished procedures ICD for subject_id={subject_id}")
    return vector


def get_services_gpu(db_conn: sqlite3.Connection, subject_id: int, services_enum: dict) -> torch.Tensor:
    print(f"Processing services for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    cur.execute("SELECT curr_service FROM services WHERE subject_id=?;", (subject_id,))
    services = [row[0] for row in cur.fetchall()]
    cur.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(services_enum["curr_service"])
    vec = torch.zeros(n, dtype=torch.int32, device=device)

    indices = [services_enum["curr_service"][svc] for svc in services if svc in services_enum["curr_service"]]
    if indices:
        vec.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                       torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished services for subject_id={subject_id}")
    return vec


def get_diagnoses_icd_gpu(db_conn: sqlite3.Connection, subject_id: int, diagnoses_icd_enum: dict) -> torch.Tensor:
    print(f"Processing diagnoses ICD for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    cur.execute("SELECT icd_code FROM diagnoses_icd WHERE subject_id=?;", (subject_id,))
    rows = cur.fetchall()
    cur.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icd_codes_enum = diagnoses_icd_enum['icd_codes']
    n = len(icd_codes_enum)
    icd_codes_vector = torch.zeros(n, dtype=torch.int32, device=device)

    indices = [icd_codes_enum[code] for (code,) in rows if code in icd_codes_enum]
    if indices:
        icd_codes_vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                                    torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished diagnoses ICD for subject_id={subject_id}")
    return icd_codes_vector


def get_admissions_gpu(db_conn: sqlite3.Connection, subject_id: int, admission_maps: dict):
    print(f"Processing admissions for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    cur.execute("SELECT admission_type, admission_location, discharge_location FROM admissions WHERE subject_id=?;", (subject_id,))
    rows = cur.fetchall()
    cur.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_admissions = len(rows)

    admission_type_vector = torch.zeros(len(admission_maps['admission_type']), dtype=torch.int32, device=device)
    admission_location_vector = torch.zeros(len(admission_maps['admission_location']), dtype=torch.int32, device=device)
    discharge_location_vector = torch.zeros(len(admission_maps['discharge_location']), dtype=torch.int32, device=device)

    type_indices, loc_indices, disc_indices = [], [], []
    for a_type, a_loc, d_loc in rows:
        if a_type in admission_maps['admission_type']:
            type_indices.append(admission_maps['admission_type'][a_type])
        if a_loc in admission_maps['admission_location']:
            loc_indices.append(admission_maps['admission_location'][a_loc])
        if d_loc in admission_maps['discharge_location']:
            disc_indices.append(admission_maps['discharge_location'][d_loc])

    if type_indices:
        admission_type_vector.index_add_(0, torch.tensor(type_indices, dtype=torch.int64, device=device),
                                        torch.ones(len(type_indices), dtype=torch.int32, device=device))
    if loc_indices:
        admission_location_vector.index_add_(0, torch.tensor(loc_indices, dtype=torch.int64, device=device),
                                            torch.ones(len(loc_indices), dtype=torch.int32, device=device))
    if disc_indices:
        discharge_location_vector.index_add_(0, torch.tensor(disc_indices, dtype=torch.int64, device=device),
                                            torch.ones(len(disc_indices), dtype=torch.int32, device=device))

    print(f"Finished admissions for subject_id={subject_id}")
    return total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector


def get_omr_result_names_gpu(db_conn: sqlite3.Connection):
    print("Extracting OMR result names (GPU compatible)...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT result_name FROM omr ORDER BY result_name;")
    all_result_names = [result[0] for result in cur.fetchall()]
    cur.close()

    result_names_index_mapping = {}
    cur_index = 0
    for result in all_result_names:
        if 'Blood Pressure' in result:
            result_names_index_mapping[result] = (cur_index, cur_index + 1)
            cur_index += 2
        else:
            result_names_index_mapping[result] = (cur_index,)
            cur_index += 1
    print("Finished extracting OMR result names.")
    return result_names_index_mapping, cur_index + 1


def get_omr_gpu(db_conn: sqlite3.Connection, subject_id: int, result_mapping: dict, vector_length: int) -> torch.Tensor:
    print(f"Processing OMR for subject_id={subject_id} (GPU)...")
    cur = db_conn.cursor()
    query = """
        SELECT * FROM (
            SELECT omr_t.*, 
                   ROW_NUMBER() OVER (PARTITION BY result_name ORDER BY chartdate DESC, seq_num DESC) AS rn
            FROM omr omr_t
            WHERE subject_id = ?
        ) AS x
        WHERE rn = 1;
    """
    cur_omr = cur.execute(query, (subject_id,))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    omr_vector = torch.zeros(vector_length, dtype=torch.float32, device=device)

    for result in cur_omr:
        result_name, result_value = result[3], result[4]
        if result_name not in result_mapping:
            continue
        if 'Blood Pressure' in result_name and '/' in result_value:
            try:
                systolic, diastolic = map(float, result_value.split('/'))
                s_idx, d_idx = result_mapping[result_name]
                omr_vector[s_idx] = systolic
                omr_vector[d_idx] = diastolic
            except Exception:
                continue
        else:
            try:
                numeric_value = float(re.sub(r'[^0-9.]', '', result_value))
                r_idx = result_mapping[result_name][0]
                omr_vector[r_idx] = numeric_value
            except Exception:
                continue

    cur.close()
    print(f"Finished OMR for subject_id={subject_id}")
    return omr_vector

def get_all_hcpcs_codes_gpu(db_conn: sqlite3.Connection) -> dict:
    print("Extracting all HCPCS codes...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT hcpcs_cd FROM hcpcsevents ORDER BY hcpcs_cd;")
    all_hcpcs_codes = [result[0] for result in cur.fetchall()]
    hcpcs_index_mapping = {code: i for i, code in enumerate(all_hcpcs_codes)}
    cur.close()
    print("Finished extracting HCPCS codes.")
    return hcpcs_index_mapping


def get_hcpcsevents_gpu(db_conn: sqlite3.Connection, subject_id: int, hcpcs_map: dict) -> torch.Tensor:
    print(f"Processing HCPCS events for subject_id={subject_id}...")
    cur = db_conn.cursor()
    cur.execute("SELECT * FROM hcpcsevents WHERE subject_id = ?;", (subject_id,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hcpcs_vector = torch.zeros(len(hcpcs_map), dtype=torch.int32, device=device)
    events = cur.fetchall()
    cur.close()

    indices = [hcpcs_map[event[3]] for event in events if event[3] in hcpcs_map]
    if indices:
        hcpcs_vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                                torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished HCPCS events for subject_id={subject_id}")
    return hcpcs_vector


def get_all_drugs_gpu(db_conn: sqlite3.Connection) -> dict:
    print("Extracting all drugs from pharmacy...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT medication FROM pharmacy ORDER BY medication")
    all_medications = [med[0] for med in cur.fetchall()]
    medication_map = {med: i for i, med in enumerate(all_medications)}
    cur.close()
    print("Finished extracting all drugs.")
    return medication_map


def get_pharmacy_gpu(db_conn: sqlite3.Connection, subject_id: int, medication_map: dict) -> torch.Tensor:
    print(f"Processing pharmacy events for subject_id={subject_id}...")
    cur = db_conn.cursor()
    cur.execute("SELECT * FROM pharmacy WHERE subject_id = ?;", (subject_id,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pharm_vector = torch.zeros(len(medication_map), dtype=torch.int32, device=device)
    events = cur.fetchall()
    cur.close()

    indices = [medication_map[event[6]] for event in events if event[6] in medication_map]
    if indices:
        pharm_vector.index_add_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                                torch.ones(len(indices), dtype=torch.int32, device=device))

    print(f"Finished pharmacy events for subject_id={subject_id}")
    return pharm_vector

# ---------------------------
# ENTRY POINT
# ---------------------------
@app.local_entrypoint()
def main():
    # Trigger the Modal GPU function
    process_ehr_data.remote()

if __name__ == "__main__":
    with app.run():
        handle = process_ehr_data.remote()
        # Stream logs
        for log in handle.stream_logs():
            print(log, end="")
        # Wait for completion
        result = handle.wait()
        print("Finished processing, result:", result)