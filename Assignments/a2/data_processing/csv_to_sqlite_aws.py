import sqlite3
import pandas as pd
import s3fs

# S3 bucket and folder
bucket_name = "health-forge-ehr-diff-training-data-136268833180"
prefix = "mimic_iv/"  # folder containing the gz files

# SQLite DB path (local in SageMaker)
db_path = "/tmp/MIMIC_IV.sqlite"
conn = sqlite3.connect(db_path)

# Create s3 filesystem object
fs = s3fs.S3FileSystem()

# List all CSV.GZ files under the folder
csv_files = fs.glob(f"s3://{bucket_name}/{prefix}*.csv.gz")
print(f"Found {len(csv_files)} compressed CSVs in S3")

# Load each CSV into SQLite
for s3_path in csv_files:
    table = s3_path.split("/")[-1].replace(".csv.gz", "")
    print(f"Loading {s3_path} into table '{table}'")

    df = pd.read_csv(s3_path, storage_options={'anon': False}, compression='gzip')
    df.to_sql(table, conn, if_exists="replace", index=False)

conn.close()
print(f"Wrote {len(csv_files)} tables into {db_path}")
