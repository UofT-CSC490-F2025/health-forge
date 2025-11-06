import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path


# === Paths ===
db_path = Path(__file__).parent.parent / "MIMIC_IV_demo.sqlite"
output_csv_path = Path(__file__).parent.parent / "vector_columns.csv"
output_npy_path = Path(__file__).parent.parent / "vector_columns.npy"


# === Helper Functions ===
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

    # Prescriptions
    cur.execute("SELECT DISTINCT gsn FROM prescriptions;")
    gsn_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["prescriptions"]["gsn"] = {v: i for i, v in enumerate(gsn_codes)}

    # Diagnoses ICD
    cur.execute("SELECT DISTINCT icd_code FROM diagnoses_icd;")
    icd_codes = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["diagnoses_icd"]["icd_codes"] = {v: i for i, v in enumerate(icd_codes)}

    # POE
    cur.execute("SELECT DISTINCT order_type FROM poe;")
    order_types = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["poe"]["order_type"] = {v: i for i, v in enumerate(order_types)}

    # Procedures ICD
    cur.execute("SELECT DISTINCT icd_code, icd_version FROM procedures_icd;")
    proc_codes = cur.fetchall()
    proc_norm = [
        f"{version}_{(code.replace('.', ''))}"
        for code, version in proc_codes if code is not None
    ]
    enums["procedures_icd"]["icd_codes"] = {v: i for i, v in enumerate(proc_norm)}

    # Services
    cur.execute("SELECT DISTINCT curr_service FROM services;")
    services = [row[0] for row in cur.fetchall() if row[0] is not None]
    enums["services"]["curr_service"] = {v: i for i, v in enumerate(services)}

    # Admissions
    cur.execute("SELECT DISTINCT admission_type FROM admissions;")
    admission_types = [row[0] for row in cur.fetchall() if row[0] is not None]
    admission_type_map = {atype: i for i, atype in enumerate(admission_types)}

    cur.execute("SELECT DISTINCT admission_location FROM admissions;")
    admission_locations = [row[0] for row in cur.fetchall() if row[0] is not None]
    admission_location_map = {loc: i for i, loc in enumerate(admission_locations)}

    cur.execute("SELECT DISTINCT discharge_location FROM admissions;")
    discharge_locations = [row[0] for row in cur.fetchall() if row[0] is not None]
    discharge_location_map = {loc: i for i, loc in enumerate(discharge_locations)}

    enums["admissions"]["admission_type"] = admission_type_map
    enums["admissions"]["admission_location"] = admission_location_map
    enums["admissions"]["discharge_location"] = discharge_location_map

    cur.close()
    return enums


def get_omr_result_names(db_conn: sqlite3.Connection):
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

    return result_names_index_mapping, cur_index


def get_all_hcpcs_codes(db_conn: sqlite3.Connection):
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT hcpcs_cd FROM hcpcsevents ORDER BY hcpcs_cd;")
    all_hcpcs_codes = [result[0] for result in cur.fetchall()]
    cur.close()
    return {code: i for i, code in enumerate(all_hcpcs_codes)}


def get_all_drugs(db_conn: sqlite3.Connection):
    cur = db_conn.cursor()
    cur.execute("SELECT DISTINCT medication FROM pharmacy ORDER BY medication")
    all_meds = [row[0] for row in cur.fetchall()]
    cur.close()
    return {med: i for i, med in enumerate(all_meds)}


# === Build Column Names ===
def build_column_names(db_conn):
    enums = get_enums(db_conn)
    omr_map, _ = get_omr_result_names(db_conn)
    hcpcs_map = get_all_hcpcs_codes(db_conn)
    med_map = get_all_drugs(db_conn)

    sections = {}

    # Basic patient info
    sections["patients"] = ["subject_id", "patients.gender", "patients.age", "patients.dod"]

    # Prescriptions
    sections["prescriptions"] = [f"prescriptions.gsn.{gsn}" for gsn in enums["prescriptions"]["gsn"]]

    # Diagnoses
    sections["diagnoses_icd"] = [f"diagnoses_icd.icd_code.{icd}" for icd in enums["diagnoses_icd"]["icd_codes"]]

    # Procedures
    sections["procedures_icd"] = [f"procedures_icd.icd_code.{code}" for code in enums["procedures_icd"]["icd_codes"]]

    # POE
    sections["poe"] = [f"poe.order_type.{ot}" for ot in enums["poe"]["order_type"]]

    # Services
    sections["services"] = [f"services.curr_service.{svc}" for svc in enums["services"]["curr_service"]]

    # Admissions
    adm_cols = ["admissions.total_admissions"]
    adm_cols += [f"admissions.admission_type.{t}" for t in enums["admissions"]["admission_type"]]
    adm_cols += [f"admissions.admission_location.{loc}" for loc in enums["admissions"]["admission_location"]]
    adm_cols += [f"admissions.discharge_location.{dloc}" for dloc in enums["admissions"]["discharge_location"]]
    sections["admissions"] = adm_cols

    # OMR
    omr_cols = []
    for name in omr_map:
        if 'Blood Pressure' in name:
            omr_cols.append(f"omr.{name}.systolic")
            omr_cols.append(f"omr.{name}.diastolic")
        else:
            omr_cols.append(f"omr.{name}")
    sections["omr"] = omr_cols

    # HCPCS
    sections["hcpcsevents"] = [f"hcpcsevents.hcpcs_cd.{code}" for code in hcpcs_map]

    # Pharmacy
    sections["pharmacy"] = [f"pharmacy.medication.{med}" for med in med_map]

    # Flatten all
    all_cols = []
    for sec in sections:
        all_cols += sections[sec]

    return all_cols, sections


# === Main ===
if __name__ == "__main__":
    conn = sqlite3.connect(db_path)
    col_names, sections = build_column_names(conn)
    conn.close()

    # Save to CSV (header only)
    pd.DataFrame(columns=col_names).to_csv(output_csv_path, index=False)

    # Save to NPY
    np.save(output_npy_path, np.array(col_names))

    # Print summary
    print(f"\n✅ Column headers saved to:")
    print(f"   CSV: {output_csv_path}")
    print(f"   NPY: {output_npy_path}")
    print(f"\n📊 Summary of column counts:")
    total = 0
    for k, v in sections.items():
        print(f"  {k:<18} {len(v):>6}")
        total += len(v)
    print(f"  {'TOTAL':<18} {total:>6}\n")
