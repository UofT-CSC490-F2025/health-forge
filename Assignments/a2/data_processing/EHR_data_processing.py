import sqlite3
import numpy as np
from pathlib import Path

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"

def merge_database(db_input_path: Path, output_table_name: str) -> None:

    conn = sqlite3.connect(db_input_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    admission_maps = get_admission_enums(conn)

    patient_vectors = {}

    for patient in cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]

        total_admissions, admission_type_vector, admission_location_vector, discharge_location_vector = get_admissions(
            conn, subject_id, admission_maps
        )

        admission_vector = np.concatenate(
            ([total_admissions], admission_type_vector, admission_location_vector, discharge_location_vector)
        )

        patient_vectors[subject_id] = admission_vector

    cur.close()
    conn.close()


def get_admission_enums(db_conn: sqlite3.Connection) -> dict:

    cur = db_conn.cursor()

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

    cur.close()
    return {
        "admission_type": admission_type_map,
        "admission_location": admission_location_map,
        "discharge_location": discharge_location_map
    }


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
    merge_database(db_path, "")