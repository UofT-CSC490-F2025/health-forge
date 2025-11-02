import sqlite3
import numpy as np

# ------------------------------
# Config: SQLite file and table
# ------------------------------
SQLITE_FILE = "ehr_data.sqlite"   # SQLite file in the same directory
TABLE_NAME = "ehr_table"          # Replace with your table name
OUTPUT_FILE = "ehr_data.npy"      # Output numpy file

# ------------------------------
# Connect to SQLite
# ------------------------------
conn = sqlite3.connect(SQLITE_FILE)
cursor = conn.cursor()

# ------------------------------
# Get column names
# ------------------------------
cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
columns = [info[1] for info in cursor.fetchall()]
print("Columns in table:", columns)

# ------------------------------
# Fetch all data
# ------------------------------
cursor.execute(f"SELECT * FROM {TABLE_NAME}")
rows = cursor.fetchall()
print(f"Number of rows fetched: {len(rows)}")

# ------------------------------
# Convert to NumPy array
# ------------------------------
data_array = np.array(rows)
print("Shape of NumPy array:", data_array.shape)

# ------------------------------
# Save to .npy file
# ------------------------------
np.save(OUTPUT_FILE, data_array)
print(f"Saved NumPy array to {OUTPUT_FILE}")

# ------------------------------
# Close connection
# ------------------------------
conn.close()
