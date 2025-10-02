import sqlite3
import numpy as np
import os
from pathlib import Path
import re

db_path = Path(__file__).parent.parent / "MIMIC_IV_demo.sqlite"
output_db_path = Path(__file__).parent.parent / "vector_store.sqlite"

def set_up_vector_store():

    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(output_db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS vectors(subject_id INTEGER PRIMARY KEY, vec BLOB);")
    conn.commit()




# ---------------------------------------------------------------------------
# TABLE FEATURE EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------
def merge_database(db_input_path: Path) -> None:
    input_conn = sqlite3.connect(db_input_path)
    input_conn.row_factory = sqlite3.Row   # enables name-based access
    output_conn = sqlite3.connect(output_db_path)
    input_cur = input_conn.cursor()
    output_cur = output_conn.cursor()


    enums = get_enums(input_conn)

    for patient in input_cur.execute("SELECT DISTINCT * FROM patients;"):
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
        hcpcs_vector = get_hcpcsevents(input_conn, subject_id,hcpcs_map)
        pharm_vector = get_pharmacy(input_conn, subject_id, medication_map)

        final = np.concatenate((subject_id_array, 
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
                                pharm_vector))
        
        vector_blob = final.astype("float32").tobytes()

        output_cur.execute("INSERT INTO vectors (subject_id, vec) VALUES (?, ?);", (subject_id, vector_blob))

    output_conn.commit()
    output_conn.close()
        




def get_enums(db_conn: sqlite3.Connection) -> dict:
    cur = db_conn.cursor()
    enums = {
        "prescriptions": {},
        "diagnoses_icd": {},
        "poe": {},
        "procedures_icd": {},
        "services": {},
        "admissions": {}
    }

    # --- Prescriptions: GSN ---
    cur.execute("SELECT DISTINCT gsn FROM prescriptions;")
    gsn_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["prescriptions"]["gsn"] = {v: i for i, v in enumerate(gsn_codes)}

    # # --- Diagnoses (ICD codes) ---
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

    # --- diagnoses_icd ---
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = cur.fetchall()
    icd_codes = [i[0] for i in icd_codes] # List[tuple[str]] -> List[str]
    icd_codes_enum = {icd_codes[i]: i for i in range(len(icd_codes))}
    enums["diagnoses_icd"]["icd_codes"] = icd_codes_enum


    # --- Admissions ---
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
    enums["admissions"]["admission_location"] =  admission_location_map
    enums["admissions"]["discharge_location"] = discharge_location_map

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
def get_diagnoses_icd(db_conn: sqlite3.Connection, subject_id: int, diagnoses_icd_enum: dict) -> np.ndarray:
    cur = db_conn.cursor()
    columns = ["icd_code"]
    columns_str = ','.join(columns)
    query = f"SELECT {columns_str} FROM diagnoses_icd WHERE subject_id={subject_id};"

    icd_codes_enum = diagnoses_icd_enum['icd_codes']
    icd_codes_vector = np.zeros(len(icd_codes_enum), dtype=np.int32)

    for diagnosis in cur.execute(query):
        code = diagnosis["icd_code"]

        icd_codes_vector[icd_codes_enum[code]] += 1
    
    cur.close()
    return icd_codes_vector


def get_omr_result_names(db_conn: sqlite3.Connection) -> dict:
    cur = db_conn.cursor()

    cur.execute("SELECT DISTINCT result_name FROM omr ORDER BY result_name;")

    all_result_names = cur.fetchall()
    all_result_names = [result[0] for result in all_result_names]
    result_names_index_mapping = {}

    cur_index = 0
    for result in all_result_names:
        
        if 'Blood Pressure' in result:
            result_names_index_mapping[result] = (cur_index, cur_index + 1)
            cur_index += 2
        else:
            result_names_index_mapping[result] = (cur_index,)
            cur_index += 1

    return result_names_index_mapping, cur_index + 1

def get_omr(db_conn: sqlite3.Connection, subject_id: int, result_mapping: dict, vector_length: int) -> np.ndarray:
    cur = db_conn.cursor()

    # Get the latest measurement in each measurement category
    query = f"""SELECT * 
        FROM (
            SELECT omr_t.*, 
                ROW_NUMBER() OVER (PARTITION BY result_name ORDER BY chartdate DESC, seq_num DESC) AS rn
            FROM omr omr_t
            WHERE subject_id = ?
        ) AS x
        WHERE rn = 1;
        """
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
            result_value = result[4]
            result_index = systolic_index = result_mapping[result[3]][0]

            result_value =  float(re.sub(r'[^0-9.]', '', result_value))
            omr_vector[result_index] = result_value

    return omr_vector


def get_all_hcpcs_codes(db_conn: sqlite3.Connection) -> dict:
    cur = db_conn.cursor()

    cur.execute("SELECT DISTINCT hcpcs_cd FROM hcpcsevents ORDER BY hcpcs_cd;")

    all_hcpcs_codes = cur.fetchall()
    all_hcpcs_codes = [result[0] for result in all_hcpcs_codes]
    hcpcs_index_mapping = {}

    for i in range(len(all_hcpcs_codes)):
        hcpcs_index_mapping[all_hcpcs_codes[i]] = i

    return hcpcs_index_mapping

def get_hcpcsevents(db_conn: sqlite3.Connection, subject_id: int, hcpcs_map: dict) -> np.ndarray:
    cur = db_conn.cursor()

    cur.execute("SELECT * FROM hcpcsevents WHERE subject_id = ?;", (subject_id,))
    patient_hcpcs_events = cur.fetchall()

    hcpcs_vector = np.zeros(len(hcpcs_map.keys()))
    for event in patient_hcpcs_events:
        hcpcs_vector[hcpcs_map[event[3]]] += 1
    return hcpcs_vector

def get_all_drugs(db_conn: sqlite3.Connection) -> dict:
    cur = db_conn.cursor()

    cur.execute("SELECT DISTINCT medication FROM pharmacy ORDER BY medication")

    all_medications = cur.fetchall()
    all_medications = [medication[0] for medication in all_medications]

    medication_map = {}

    for i in range(len(all_medications)):
        medication_map[all_medications[i]] = i
    return medication_map

def get_pharmacy(db_conn: sqlite3.Connection, subject_id: int, medication_map: dict) -> np.ndarray:
    cur = db_conn.cursor()

    cur.execute("SELECT * FROM pharmacy WHERE subject_id = ?;", (subject_id,))
    patient_pharm_events = cur.fetchall()

    pharm_vector = np.zeros(len(medication_map.keys()))
    for event in patient_pharm_events:
        pharm_vector[medication_map[event[6]]] += 1
    return pharm_vector


def get_admissions(db_conn: sqlite3.Connection, subject_id: int, admission_maps: dict):

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
    return total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector


if __name__ == "__main__":
    set_up_vector_store()
    merge_database(db_path)