import sqlite3
import pandas as pd
import s3fs
import boto3


bucket_name = "health-forge-ehr-diff-training-data-136268833180"
prefix = "mimic_iv/"
db_path = "/tmp/MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
fs = s3fs.S3FileSystem()

csv_files = fs.glob(f"s3://{bucket_name}/{prefix}*.csv.gz")
print(f"Found {len(csv_files)} compressed CSVs in S3")

for s3_path in csv_files:
    table = s3_path.split("/")[-1].replace(".csv.gz", "")
    print(f"Loading {s3_path} into table '{table}' in chunks")

    with fs.open(s3_path, 'rb') as f:
        # Read in chunks of 100,000 rows
        chunksize = 100_000
        for i, chunk in enumerate(pd.read_csv(f, compression='gzip', chunksize=chunksize)):
            chunk.to_sql(table, conn, if_exists='append' if i > 0 else 'replace', index=False)
            print(f"  Written chunk {i+1} for table {table}")

conn.close()
print(f"Finished writing tables into {db_path}")


s3 = boto3.client("s3")
s3.upload_file(db_path, bucket_name, f"{prefix}MIMIC_IV_demo.sqlite")
print("SQLite DB uploaded to S3")
