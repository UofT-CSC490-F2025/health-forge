import sqlite3
import numpy as np
from pathlib import Path
import re

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

def merge_database(db_input_path: Path, output_table_name: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # enables name-based access
    cur = conn.cursor()


    result_mapping, vector_length = get_omr_result_names(conn)
    hcpcs_map = get_all_hcpcs_codes(conn)
    medication_map = get_all_drugs(conn)
    for patient in cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]
        gender = 0 if patient["gender"] == 'M' else 1
        age = patient["anchor_age"]


        get_omr(conn, subject_id, result_mapping, vector_length)
        get_hcpcsevents(conn, subject_id,hcpcs_map)
        get_pharmacy(conn, subject_id, medication_map)

    cur.close()
    conn.close()


def get_omr_result_names(db_conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()

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

if __name__ == "__main__":
    merge_database(db_path, "")
    




