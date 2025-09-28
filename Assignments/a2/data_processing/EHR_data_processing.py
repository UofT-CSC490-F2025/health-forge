import sqlite3
import numpy as np
from pathlib import Path

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()


#Get a list of all patient names
cur.execute("SELECT * FROM patients;")
all_patients = cur.fetchall()
print(all_patients)

#For each patient

for patient in all_patients:
    output_vector = [] #Will start with a normal dynamic array for now, we can use a NP array once we know the actual shape
    
    #add patient's id
    output_vector.append(patient[0])
    #add patient's gender
    output_vector.append(0 if patient[1] == 'M' else 1)
    #add patient's anchor_age
    output_vector.append(patient[2])
    #add patient's anchor_year
    output_vector.append(patient[3])

    #add patient's anchor_year_group
    output_vector.append(*[int(year) for year in patient[4].split('-')])


    # Add admission table to vector
    # Add diagnoses_icd to vector (holds patients and diagnosis)
    # Add drgcodes to vector  (holds reason for patient stay on each visit)
    # Add emar table to vector  (holds drugs administrated to each patient)
    # Add emar_details (holds additional details about each drug admin)
    # Add hspc_events (holds codes for every procdure and event done)
    # Add lab_events (holds results of lab measurements for a patient)
    # Add microbiologyevents
    # Add pharmacy (detailed information about drugs administered)
    # Add poe (table of drug orders)
    # Add prescriptions (Holds infro about prescribed drugs to patients)
    # Add procedures_icd (Holds procedure history of each patient)
    # Add services (holds which service (type of admission kinda) that each patient had)
    # Add transfers (holds details about patient unit transfer  )




