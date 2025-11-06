import sqlite3
import pandas as pd
import boto3

# S3 bucket and folder
bucket_name = "your-bucket-name"
prefix = "mimic-iv-clinical-database-demo-2.2/"  # S3 folder containing gz files

# Connect to SQLite (still local on SageMaker)
db_path = "/tmp/MIMIC_IV_demo.sqlite"  # you can write to /tmp in SageMaker
conn = sqlite3.connect(db_path)

# Use boto3 to list files in S3
s3 = boto3.client("s3")
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Filter for CSV.GZ files
csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(".csv.gz")]
print(f"Found {len(csv_files)} compressed CSVs in S3")

# Load each file into SQLite
for key in csv_files:
    table = key.split("/")[-1].replace(".csv.gz", "")
    print(f"Loading {key} into table '{table}'")

    s3_path = f"s3://{bucket_name}/{key}"
    df = pd.read_csv(s3_path, compression='gzip')  # read directly from S3
    df.to_sql(table, conn, if_exists="replace", index=False)

conn.close()
print(f"Wrote {len(csv_files)} tables into {db_path}")
