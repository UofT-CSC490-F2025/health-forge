import sqlite3
import pandas as pd
from pathlib import Path
from slugify import slugify

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)

csv_dir = Path(__file__).parent.parent / "mimic-iv-clinical-database-demo-2.2"
csv_files = list(csv_dir.rglob("*.csv"))
for p in csv_files:
    table = slugify(p.stem, separator="_") or "table"
    df = pd.read_csv(p)
    df.to_sql(table, conn, if_exists="replace", index=False)

conn.close()
print(f"Wrote {len(csv_files)} tables into {db_path}")