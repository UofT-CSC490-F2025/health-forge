import sqlite3
import numpy as np
from pathlib import Path

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"

# ---------------------------------------------------------------------------
# TABLE FEATURE EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def get_enums(db_conn: sqlite3.Connection) -> dict:
    cur = db_conn.cursor()
    enums = {
        "prescriptions": {},
        "diagnoses_icd": {},
        "poe": {},
        "procedures_icd": {},
        "services": {}
    }

    # --- Prescriptions: GSN ---
    cur.execute("SELECT DISTINCT gsn FROM prescriptions;")
    gsn_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["prescriptions"]["gsn"] = {v: i for i, v in enumerate(gsn_codes)}

    # --- Diagnoses (ICD codes) ---
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["diagnoses_icd"]["icd_codes"] = {v: i for i, v in enumerate(icd_codes)}

    # --- POE (order types) ---
    cur.execute("SELECT DISTINCT order_type FROM poe;")
    order_types = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["poe"]["order_type"] = {v: i for i, v in enumerate(order_types)}

    # --- Procedures ICD ---
    cur.execute("SELECT DISTINCT icd_code, icd_version FROM procedures_icd;")
    proc_codes = cur.fetchall()
    proc_norm = [
        f"{version}_{(code.replace('.', ''))}"
        for code, version in proc_codes if code is not None
    ]
    enums["procedures_icd"]["icd_codes"] = {v: i for i, v in enumerate(proc_norm)}

    # --- Services (curr_service abbreviations) ---
    cur.execute("SELECT DISTINCT curr_service FROM services;")
    services = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["services"]["curr_service"] = {v: i for i, v in enumerate(services)}

    cur.close()
    return enums


# order_type was chosen as the feature to extract from poe table as it is the only structured column that holds meaning.
# aspects such as time of order, or transaction type are not important to an EHR. unfortunatetly it is not defined if order_subtype
# is a structured form of data entry so it is ignored for now.
#
# NOTE: poe_detail table is omitted entirely as the only column that holds information being field_name and field_value
# hold not structured meaning. Their typing is dynamic and meaning needs to be inferred. A more detailed view on the order
# made for the patient can be obtained from the prescription table.
def get_poe(db_conn: sqlite3.Connection, subject_id: int, poe_enum: dict) -> np.ndarray:
    cur = db_conn.cursor()
    query = f"SELECT order_type FROM poe WHERE subject_id={subject_id};"

    order_type_enum = poe_enum['order_type']
    poe_vector = np.zeros(len(order_type_enum), dtype=np.int32)

    for (order_type,) in cur.execute(query):
        if order_type in order_type_enum:
            poe_vector[order_type_enum[order_type]] += 1

    cur.close()
    return poe_vector

# prescriptions table holds lots of information regarding prescriptions made to patients. This function only creates a 
# frequency encoding of the GSN codes. Note the process of elimination: 
#
# - drug_type is too vague, only 3 types exist and they do not hold much meaning
# - drug is free text and cannot be used. we need control over vector size
# - NDC is a great coded identifier, however it is too vast, leading to a very sparse vector
# - product strength and dose related columns are either free text or too specific to their medication type to hold meaning
# 
# As a compromise, GSN codes are used as they are a non-free text identifier that groups pharmaceutically equivalent drugs.
# This captures the essence of what drugs were prescribed to a patient without being too specific. For this reason, GSN
# maps multiple NDC codes to a single identifier, which keeps the structured nature of the data while keeping the vector size manageable. 
def get_prescriptions(db_conn: sqlite3.Connection, subject_id: int, prescription_enums: dict) -> np.ndarray:
    cur = db_conn.cursor()
    query = f"SELECT gsn FROM prescriptions WHERE subject_id={subject_id};"

    gsn_enum = prescription_enums['gsn']
    gsn_vector = np.zeros(len(gsn_enum), dtype=np.int32)

    for (gsn,) in cur.execute(query):
        if gsn in gsn_enum:
            gsn_vector[gsn_enum[gsn]] += 1

    cur.close()
    return gsn_vector


# the only data point of value in procedures_icd is the icd_code which represents the procedure the patient underwent.
# various cleaning was first done to the icd_code to normalize it. the decimal point was removed as it does not hold meaning
# (eg. 0.123 and 0123 are the same code). Additionally, the icd_version was prepended to the code to differentiate between
# icd9 and icd10 codes. This was done as hospital systems may use either version, and the same code in different versions may represent
# different procedures. icd10 is also more granular, so it is complex to translate icd9 codes to icd10 codes.
def get_procedures_icd(db_conn, subject_id: int, proc_enums: dict) -> np.ndarray:
    cur = db_conn.cursor()
    cur.execute(
        "SELECT icd_code, icd_version FROM procedures_icd WHERE subject_id=?;",
        (subject_id,)
    )

    vector = np.zeros(len(proc_enums), dtype=np.int32)
    for code, version in cur.fetchall():
        if code:
            norm_code = f"{version}_{code.replace('.', '')}"
            if norm_code in proc_enums:
                vector[proc_enums[norm_code]] += 1

    cur.close()
    return vector


# services table holds a curr_service column and a prev_service column. we can simplely isolate curr_service and create
# a frequency encoding of this column to get the patients service history. the column values are a non-free-text
# abbreviation from a fixed pool provided in the documentation, so we know the shape must be manageable.
def get_services(db_conn: sqlite3.Connection, subject_id: int, services_enum: dict) -> np.ndarray:
    cur = db_conn.cursor()
    query = "SELECT curr_service FROM services WHERE subject_id=?;"
    vec = np.zeros(len(services_enum["curr_service"]), dtype=np.int32)

    for (svc,) in cur.execute(query, (subject_id,)):
        if svc in services_enum["curr_service"]:
            vec[services_enum["curr_service"][svc]] += 1

    cur.close()
    return vec

# transfers table is omitted as it holds meaning in the form of a timeline of patient transfers, even within a single
# hospital admission. since our frequency encodings do not capture temporal data, this table is not useful for our purposes.




# testing code, testing shape and frequency encoding
if __name__ == "__main__":
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Grab a sample of patient IDs
    cur.execute("SELECT subject_id FROM patients LIMIT 5;")
    sample_ids = [row["subject_id"] for row in cur.fetchall()]

    # Build enums
    enums = get_enums(conn)

    for subject_id in sample_ids:
        print(f"\n--- Testing subject_id={subject_id} ---")

        # Prescriptions
        rx_vec = get_prescriptions(conn, subject_id, enums["prescriptions"])
        rx_density = np.count_nonzero(rx_vec) / len(rx_vec)
        print("Prescriptions vector:", rx_vec.shape, "Nonzeros:", np.count_nonzero(rx_vec), f"Density: {rx_density:.2%}")

        # Procedures
        proc_vec = get_procedures_icd(conn, subject_id, enums["procedures_icd"]["icd_codes"])
        proc_density = np.count_nonzero(proc_vec) / len(proc_vec)
        print("Procedures vector:", proc_vec.shape, "Nonzeros:", np.count_nonzero(proc_vec), f"Density: {proc_density:.2%}")

        # POE
        poe_vec = get_poe(conn, subject_id, enums["poe"])
        poe_density = np.count_nonzero(poe_vec) / len(poe_vec)
        print("POE vector:", poe_vec.shape, "Nonzeros:", np.count_nonzero(poe_vec), f"Density: {poe_density:.2%}")
        print("POE vector raw:", poe_vec.tolist())
        active_poe = [otype for otype, idx in enums["poe"]["order_type"].items() if poe_vec[idx] > 0]
        print("POE active types:", active_poe)

        # Services
        svc_vec = get_services(conn, subject_id, enums["services"])
        svc_density = np.count_nonzero(svc_vec) / len(svc_vec)
        print("Services vector:", svc_vec.shape, "Nonzeros:", np.count_nonzero(svc_vec), f"Density: {svc_density:.2%}")
        print("Services vector raw:", svc_vec.tolist())
        active_services = [svc for svc, idx in enums["services"]["curr_service"].items() if svc_vec[idx] > 0]
        print("Services active types:", active_services)

    cur.close()
    conn.close()