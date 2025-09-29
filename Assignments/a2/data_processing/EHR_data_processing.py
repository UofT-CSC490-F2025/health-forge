import sqlite3
import numpy as np
from pathlib import Path

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()


def merge_database(db_input_path: Path, output_table_name: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # enables name-based access
    cur = conn.cursor()

    enums = get_enums(conn)

    for patient in cur.execute("SELECT * FROM patients;"):
        subject_id = patient["subject_id"]
        gender = 0 if patient["gender"] == 'M' else 1
        age = patient["anchor_age"]
        # anchor_year = patient["anchor_year"]
        # anchor_year_group = patient["anchor_year_group"]
        dod = 1 if patient["dod"] else 0

        drug_type_vector, formulary_drug_cd_vector = get_prescriptions(conn, subject_id, enums["prescriptions"])
        icd_codes_vector = get_diagnoses_icd(conn, subject_id, enums['diagnoses_icd'])

    cur.close()
    conn.close()


def get_enums(db_conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    enums = {
                "prescriptions": {},
                "diagnoses_icd": {},
            }

    # --- Prescription ---
    cur.execute("SELECT DISTINCT formulary_drug_cd FROM prescriptions;")
    formulary_drug_cds = cur.fetchall()
    formulary_drug_cds = [f[0] for f in formulary_drug_cds] # List[Tuple[String]] -> List[String]
    formulary_drug_cds_enum = {formulary_drug_cds[i]: i for i in range(len(formulary_drug_cds))}
    enums["prescriptions"]["formulary_drug_cd"] = formulary_drug_cds_enum
    enums["prescriptions"]["drug_type"] = {"MAIN": 0, "BASE": 1, "ADDITIVE": 2}

    # --- diagnoses_icd ---
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = cur.fetchall()
    icd_codes = [i[0] for i in icd_codes] # List[tuple[str]] -> List[str]
    icd_codes_enum = {icd_codes[i]: i for i in range(len(icd_codes))}
    enums["diagnoses_icd"]["icd_codes"] = icd_codes_enum


    cur.close()
    return enums


def get_prescriptions(db_conn: sqlite3.Connection, subject_id: int, prescription_enums: dict) -> tuple[np.ndarray, np.ndarray]:
    cur = db_conn.cursor()
    columns = ["drug_type", "formulary_drug_cd"]
    columns_str = ','.join(columns)
    query = f"SELECT {columns_str} FROM prescriptions WHERE subject_id={subject_id};"
    
    formulary_drug_cds_enum = prescription_enums['formulary_drug_cd']
    drug_type_enum = prescription_enums['drug_type']

    drug_type_vector = np.zeros(3, dtype=np.int32)
    formulary_drug_cd_vector = np.zeros(len(formulary_drug_cds_enum), dtype=np.int32)

    for prescription in cur.execute(query):
        drug_type = prescription['drug_type']
        formulary_drug_cd = prescription['formulary_drug_cd']

        drug_type_vector[drug_type_enum[drug_type]] += 1
        formulary_drug_cd_vector[formulary_drug_cds_enum[formulary_drug_cd]] += 1
    
    cur.close()
    return (drug_type_vector, formulary_drug_cd_vector)

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

if __name__ == "__main__":
    merge_database(db_path, "test")

