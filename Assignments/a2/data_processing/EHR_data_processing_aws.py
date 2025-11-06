import sqlite3
import numpy as np
import os
import re
import boto3

# ---------------------------
# AWS S3 CONFIG
# ---------------------------
BUCKET_NAME = "health-forge-ehr-diff-training-data-136268833180"
S3_PREFIX = "mimic_iv"
INPUT_DB_FILENAME = "MIMIC_IV_demo.sqlite"
LOCAL_INPUT_DB_PATH = f"/tmp/{INPUT_DB_FILENAME}"  # download S3 DB here
OUTPUT_DB_PATH = "/tmp/vector_store.sqlite"

s3_client = boto3.client("s3")


# ---------------------------
# DOWNLOAD INPUT DB FROM S3
# ---------------------------
def download_input_db():
    s3_client.download_file(BUCKET_NAME, f"{S3_PREFIX}/{INPUT_DB_FILENAME}", LOCAL_INPUT_DB_PATH)
    print(f"Downloaded {INPUT_DB_FILENAME} from S3 to {LOCAL_INPUT_DB_PATH}")


# ---------------------------
# SETUP VECTOR STORE
# ---------------------------
def set_up_vector_store():
    if os.path.exists(OUTPUT_DB_PATH):
        os.remove(OUTPUT_DB_PATH)

    conn = sqlite3.connect(OUTPUT_DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS vectors(subject_id INTEGER PRIMARY KEY, vec BLOB);")
    conn.commit()
    conn.close()


# ---------------------------
# ENUM AND FEATURE FUNCTIONS (from your code)
# ---------------------------
# Keep all the helper functions from your original code here:
# get_enums, get_prescriptions, get_poe, get_procedures_icd, get_services, get_diagnoses_icd, 
# get_omr_result_names, get_omr, get_all_hcpcs_codes, get_hcpcsevents, get_all_drugs, get_pharmacy, get_admissions

# ---------------------------
# MERGE DATABASE FUNCTION
# ---------------------------
def merge_database(db_input_path: str):
    set_up_vector_store()

    input_conn = sqlite3.connect(db_input_path)
    input_conn.row_factory = sqlite3.Row   # enable name-based access
    output_conn = sqlite3.connect(OUTPUT_DB_PATH)
    input_cur = input_conn.cursor()
    output_cur = output_conn.cursor()

    enums = get_enums(input_conn)

    # Loop through patients
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

    print(f"Finished creating vector store: {OUTPUT_DB_PATH}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    download_input_db()
    merge_database(LOCAL_INPUT_DB_PATH)
