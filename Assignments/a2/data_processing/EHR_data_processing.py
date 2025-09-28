import sqlite3
import pandas as pd
from pathlib import Path
from slugify import slugify 

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Tables:", tables)


#Get a list of all patient names


#For each patient

    # Add admission table to vector
    # Add diagnoses_icd to vector (holds patients and diagnosis)
    # Add drgcodes to vector  (holds reason for patient stay on each visit)
    # Add emar table to vector  (holds drugs administrated to each patient)
    # Add emar_details (holds additional details about each drug admin)
    # Add hspc_events (holds codes for every procdure and event done)
    # Add lab_events (holds results of lab measurements for a patient)
    # Add microbiologyevents 
    # Add patients table (simple, patient metadata)
    # Add pharmacy (detailed information about drugs administered)
    # Add poe (table of drug orders)
    # Add prescriptions (Holds infro about prescribed drugs to patients)
    # Add procedures_icd (Holds procedure history of each patient)
    # Add services (holds which service (type of admission kinda) that each patient had)
    # Add transfers (holds details about patient unit transfer  )




