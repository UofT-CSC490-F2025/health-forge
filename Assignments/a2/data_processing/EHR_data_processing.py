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
    for patient in cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]
        gender = 0 if patient["gender"] == 'M' else 1
        age = patient["anchor_age"]


        get_omr(conn, subject_id, result_mapping, vector_length)

    cur.close()
    conn.close()


def get_omr_result_names(db_conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT result_name FROM omr ORDER BY result_name;")

    all_result_names = cur.fetchall()
    all_result_names = [result[0] for result in all_result_names]
    print(all_result_names)
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
    cur = conn.cursor()

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


        

    print(omr_vector)
    return omr_vector

if __name__ == "__main__":
    merge_database(db_path, "")
    




