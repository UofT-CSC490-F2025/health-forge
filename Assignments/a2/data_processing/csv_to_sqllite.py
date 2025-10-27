import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path(__file__).parent.parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)

csv_dir = Path(__file__).parent.parent / "mimic-iv-clinical-database-demo-2.2"
csv_files = list(csv_dir.rglob("*.csv.gz"))

print(f"Found {len(csv_files)} compressed CSVs")

for p in csv_files:
    table = p.name.replace(".csv.gz", "")   # e.g. "patients.csv.gz" → "patients"
    print(f"Loading {p.name} into table '{table}'")

    df = pd.read_csv(p)
    df.to_sql(table, conn, if_exists="replace", index=False)

print("Looking in:", csv_dir.resolve())
csv_files = list(csv_dir.glob("*.csv.gz"))
print("Found:", len(csv_files))
for p in csv_files[:5]:
    print("Example file:", p)

conn.close()
print(f"Wrote {len(csv_files)} tables into {db_path}")