import sqlite3
import pandas as pd
import s3fs

bucket_name = "health-forge-ehr-diff-training-data-136268833180"
prefix = "mimic_iv/"

# SQLite DB path
db_path = "/tmp/MIMIC_IV.sqlite"
conn = sqlite3.connect(db_path)

# Create S3 filesystem
fs = s3fs.S3FileSystem()  # uses SageMaker IAM role credentials

# List all CSV.GZ files
csv_files = fs.glob(f"s3://{bucket_name}/{prefix}*.csv.gz")
print(f"Found {len(csv_files)} compressed CSVs in S3")

for s3_path in csv_files:
    table = s3_path.split("/")[-1].replace(".csv.gz", "")
    print(f"Loading {s3_path} into table '{table}'")

    # Use the filesystem object when opening the file
    with fs.open(s3_path, 'rb') as f:
        df = pd.read_csv(f, compression='gzip')
        df.to_sql(table, conn, if_exists="replace", index=False)

conn.close()
print(f"Wrote {len(csv_files)} tables into {db_path}")
