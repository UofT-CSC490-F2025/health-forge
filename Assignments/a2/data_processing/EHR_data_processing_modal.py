import modal
import boto3
import sqlite3
import numpy as np
import tempfile
import os
import re
from pathlib import Path

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
    image=modal.Image.debian_slim().pip_install(["boto3", "numpy", "pandas"]),    
    secrets=[modal.Secret.from_name("aws-secret")],  # use the secret you created
)
def process_ehr_data():
    """
    Downloads EHR database from S3 and processes it with merge_database().
    """

    s3 = boto3.client("s3")

    s3_client = boto3.client("s3")  # boto3 automatically picks up AWS credentials from environment
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp_db_file:
        print(f"Downloading {INPUT_DB_FILENAME} from S3 bucket {BUCKET_NAME}...")
        s3_client.download_file(BUCKET_NAME, INPUT_DB_FILENAME, tmp_db_file.name)
        print(f"Downloaded to temporary path: {tmp_db_file.name}")

        # Call your processing function
        merge_database(tmp_db_file.name)

        # after merge_database
        s3.upload_file("/tmp/vector_store.sqlite", BUCKET_NAME, "vector_store.sqlite")
        print("Uploaded processed vector store to S3")

# ---------------------------
# DATABASE PROCESSING LOGIC
# ---------------------------
def merge_database(db_input_path: str):
    """
    Your existing merge_database logic adapted to accept a path string.
    """


    
    output_db_path = "/tmp/vector_store.sqlite"
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

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

    enums = get_enums(input_conn)

    enums = get_enums(input_conn)

    for patient in input_cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]
        subject_id_array = np.array([subject_id])
        gender = np.array([0 if patient["gender"] == 'M' else 1])
        age = np.array([patient["anchor_age"]])
        dod = np.array([1 if patient["dod"] else 0])

        gsn_vector = get_prescriptions(input_conn, subject_id, enums["prescriptions"])
        icd_codes_vector = get_diagnoses_icd(input_conn, subject_id, enums['diagnoses_icd'])
        proc_vector = get_procedures_icd(input_conn, subject_id, enums["procedures_icd"]["icd_codes"])
        poe_vector = get_poe(input_conn, subject_id, enums["poe"])
        svc_vector = get_services(input_conn, subject_id, enums["services"])

        total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector = get_admissions(
            input_conn, subject_id, enums["admissions"]
        )

        admission_vector = np.concatenate(
            ([total_admissions], admission_type_vector, admission_location_vector, discharge_location_vector)
        )

        result_mapping, vector_length = get_omr_result_names(input_conn)
        hcpcs_map = get_all_hcpcs_codes(input_conn)
        medication_map = get_all_drugs(input_conn)
        omr_vector = get_omr(input_conn, subject_id, result_mapping, vector_length)
        hcpcs_vector = get_hcpcsevents(input_conn, subject_id, hcpcs_map)
        pharm_vector = get_pharmacy(input_conn, subject_id, medication_map)

        final = np.concatenate((
            subject_id_array,
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

        vector_blob = final.astype("float32").tobytes()
        output_cur.execute("INSERT INTO vectors (subject_id, vec) VALUES (?, ?);", (subject_id, vector_blob))

    output_conn.commit()
    output_conn.close()
    input_conn.close()
    print(f"Vector store created at {output_db_path}")


# ---------------------------
# ALL OTHER HELPER FUNCTIONS (get_enums, get_poe, get_prescriptions, etc.)
# Copy your existing helper functions here, unchanged
# ---------------------------
def get_enums(db_conn: sqlite3.Connection) -> dict:
    print("Starting to extract enums from database...")
    cur = db_conn.cursor()
    enums = {
        "prescriptions": {},
        "diagnoses_icd": {},
        "poe": {},
        "procedures_icd": {},
        "services": {},
        "admissions": {}
    }

    print("Processing prescriptions...")
    cur.execute("SELECT DISTINCT gsn FROM prescriptions;")
    gsn_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["prescriptions"]["gsn"] = {v: i for i, v in enumerate(gsn_codes)}

    print("Processing diagnoses ICD codes...")
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["diagnoses_icd"]["icd_codes"] = {v: i for i, v in enumerate(icd_codes)}

    print("Processing POE (orders)...")
    cur.execute("SELECT DISTINCT order_type FROM poe;")
    order_types = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["poe"]["order_type"] = {v: i for i, v in enumerate(order_types)}

    print("Processing procedures ICD...")
    cur.execute("SELECT DISTINCT icd_code, icd_version FROM procedures_icd;")
    proc_codes = cur.fetchall()
    proc_norm = [f"{version}_{(code.replace('.', ''))}" for code, version in proc_codes if code is not None]
    enums["procedures_icd"]["icd_codes"] = {v: i for i, v in enumerate(proc_norm)}

    print("Processing services...")
    cur.execute("SELECT DISTINCT curr_service FROM services;")
    services = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["services"]["curr_service"] = {v: i for i, v in enumerate(services)}

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

    cur.close()
    print("Finished extracting enums.")
    return enums


def get_poe(db_conn: sqlite3.Connection, subject_id: int, poe_enum: dict) -> np.ndarray:
    print(f"Processing POE for subject_id={subject_id}...")
    cur = db_conn.cursor()
    query = f"SELECT order_type FROM poe WHERE subject_id={subject_id};"
    order_type_enum = poe_enum['order_type']
    poe_vector = np.zeros(len(order_type_enum), dtype=np.int32)

    for (order_type,) in cur.execute(query):
        if order_type in order_type_enum:
            poe_vector[order_type_enum[order_type]] += 1

    cur.close()
    print(f"Finished POE for subject_id={subject_id}")
    return poe_vector


def get_prescriptions(db_conn: sqlite3.Connection, subject_id: int, prescription_enums: dict) -> np.ndarray:
    print(f"Processing prescriptions for subject_id={subject_id}...")
    cur = db_conn.cursor()
    query = f"SELECT gsn FROM prescriptions WHERE subject_id={subject_id};"
    gsn_enum = prescription_enums['gsn']
    gsn_vector = np.zeros(len(gsn_enum), dtype=np.int32)

    for (gsn,) in cur.execute(query):
        if gsn in gsn_enum:
            gsn_vector[gsn_enum[gsn]] += 1

    cur.close()
    print(f"Finished prescriptions for subject_id={subject_id}")
    return gsn_vector


def get_procedures_icd(db_conn, subject_id: int, proc_enums: dict) -> np.ndarray:
    print(f"Processing procedures ICD for subject_id={subject_id}...")
    cur = db_conn.cursor()
    cur.execute("SELECT icd_code, icd_version FROM procedures_icd WHERE subject_id=?;", (subject_id,))
    vector = np.zeros(len(proc_enums), dtype=np.int32)

    for code, version in cur.fetchall():
        if code:
            norm_code = f"{version}_{code.replace('.', '')}"
            if norm_code in proc_enums:
                vector[proc_enums[norm_code]] += 1

    cur.close()
    print(f"Finished procedures ICD for subject_id={subject_id}")
    return vector


def get_services(db_conn: sqlite3.Connection, subject_id: int, services_enum: dict) -> np.ndarray:
    print(f"Processing services for subject_id={subject_id}...")
    cur = db_conn.cursor()
    query = "SELECT curr_service FROM services WHERE subject_id=?;"
    vec = np.zeros(len(services_enum["curr_service"]), dtype=np.int32)

    for (svc,) in cur.execute(query, (subject_id,)):
        if svc in services_enum["curr_service"]:
            vec[services_enum["curr_service"][svc]] += 1

    cur.close()
    print(f"Finished services for subject_id={subject_id}")
    return vec


def get_diagnoses_icd(db_conn: sqlite3.Connection, subject_id: int, diagnoses_icd_enum: dict) -> np.ndarray:
    print(f"Processing diagnoses ICD for subject_id={subject_id}...")
    cur = db_conn.cursor()
    columns_str = 'icd_code'
    query = f"SELECT {columns_str} FROM diagnoses_icd WHERE subject_id={subject_id};"
    icd_codes_enum = diagnoses_icd_enum['icd_codes']
    icd_codes_vector = np.zeros(len(icd_codes_enum), dtype=np.int32)

    for diagnosis in cur.execute(query):
        code = diagnosis["icd_code"]
        icd_codes_vector[icd_codes_enum[code]] += 1

    cur.close()
    print(f"Finished diagnoses ICD for subject_id={subject_id}")
    return icd_codes_vector


def get_admissions(db_conn: sqlite3.Connection, subject_id: int, admission_maps: dict):
    print(f"Processing admissions for subject_id={subject_id}...")
    cur = db_conn.cursor()
    query = "SELECT admission_type, admission_location, discharge_location FROM admissions WHERE subject_id=?;"
    rows = cur.execute(query, (subject_id,))

    total_admissions = 0
    admission_type_vector = np.zeros(len(admission_maps['admission_type']), dtype=np.int32)
    admission_location_vector = np.zeros(len(admission_maps['admission_location']), dtype=np.int32)
    discharge_location_vector = np.zeros(len(admission_maps['discharge_location']), dtype=np.int32)

    for row in rows:
        total_admissions += 1
        admission_type_vector[admission_maps['admission_type'][row['admission_type']]] += 1
        admission_location_vector[admission_maps['admission_location'][row['admission_location']]] += 1
        discharge_location_vector[admission_maps['discharge_location'][row['discharge_location']]] += 1

    cur.close()
    print(f"Finished admissions for subject_id={subject_id}")
    return total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector

def get_omr_result_names(db_conn: sqlite3.Connection) -> dict:
    print("Extracting OMR result names...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT result_name FROM omr ORDER BY result_name;")
    all_result_names = [result[0] for result in cur.fetchall()]
    result_names_index_mapping = {}
    cur_index = 0
    for result in all_result_names:
        if 'Blood Pressure' in result:
            result_names_index_mapping[result] = (cur_index, cur_index + 1)
            cur_index += 2
        else:
            result_names_index_mapping[result] = (cur_index,)
            cur_index += 1
    cur.close()
    print("Finished extracting OMR result names.")
    return result_names_index_mapping, cur_index + 1


def get_omr(db_conn: sqlite3.Connection, subject_id: int, result_mapping: dict, vector_length: int) -> np.ndarray:
    print(f"Processing OMR for subject_id={subject_id}...")
    cur = db_conn.cursor()
    query = f"""SELECT * 
        FROM (
            SELECT omr_t.*, 
                ROW_NUMBER() OVER (PARTITION BY result_name ORDER BY chartdate DESC, seq_num DESC) AS rn
            FROM omr omr_t
            WHERE subject_id = ?
        ) AS x
        WHERE rn = 1;"""
    cur_omr = cur.execute(query, (subject_id,))
    omr_vector = np.zeros(vector_length)

    for result in cur_omr:
        if 'Blood Pressure' in result[3]:
            split = result[4].split('/')
            systolic = float(split[0])
            diastolic = float(split[1])
            systolic_index = result_mapping[result[3]][0]
            diastolic_index = result_mapping[result[3]][1]
            omr_vector[systolic_index] = systolic
            omr_vector[diastolic_index] = diastolic
        else:
            result_value = float(re.sub(r'[^0-9.]', '', result[4]))
            result_index = result_mapping[result[3]][0]
            omr_vector[result_index] = result_value

    cur.close()
    print(f"Finished OMR for subject_id={subject_id}")
    return omr_vector


def get_all_hcpcs_codes(db_conn: sqlite3.Connection) -> dict:
    print("Extracting all HCPCS codes...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT hcpcs_cd FROM hcpcsevents ORDER BY hcpcs_cd;")
    all_hcpcs_codes = [result[0] for result in cur.fetchall()]
    hcpcs_index_mapping = {code: i for i, code in enumerate(all_hcpcs_codes)}
    cur.close()
    print("Finished extracting HCPCS codes.")
    return hcpcs_index_mapping


def get_hcpcsevents(db_conn: sqlite3.Connection, subject_id: int, hcpcs_map: dict) -> np.ndarray:
    print(f"Processing HCPCS events for subject_id={subject_id}...")
    cur = db_conn.cursor()
    cur.execute("SELECT * FROM hcpcsevents WHERE subject_id = ?;", (subject_id,))
    hcpcs_vector = np.zeros(len(hcpcs_map))
    for event in cur.fetchall():
        hcpcs_vector[hcpcs_map[event[3]]] += 1
    cur.close()
    print(f"Finished HCPCS events for subject_id={subject_id}")
    return hcpcs_vector


def get_all_drugs(db_conn: sqlite3.Connection) -> dict:
    print("Extracting all drugs from pharmacy...")
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT medication FROM pharmacy ORDER BY medication")
    all_medications = [med[0] for med in cur.fetchall()]
    medication_map = {med: i for i, med in enumerate(all_medications)}
    cur.close()
    print("Finished extracting all drugs.")
    return medication_map


def get_pharmacy(db_conn: sqlite3.Connection, subject_id: int, medication_map: dict) -> np.ndarray:
    print(f"Processing pharmacy events for subject_id={subject_id}...")
    cur = db_conn.cursor()
    cur.execute("SELECT * FROM pharmacy WHERE subject_id = ?;", (subject_id,))
    pharm_vector = np.zeros(len(medication_map))
    for event in cur.fetchall():
        pharm_vector[medication_map[event[6]]] += 1
    cur.close()
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