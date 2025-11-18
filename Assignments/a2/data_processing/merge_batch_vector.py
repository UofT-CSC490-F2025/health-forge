import modal
import boto3
import numpy as np
import io

BUCKET_NAME = "healthforge-final-bucket"
BATCH_PREFIX = "vector_batches/"
MERGED_KEY = "merged_patient_vectors.npy"

app = modal.App("gpu-ehr-processing")  # your existing app

@app.function(
    timeout=60 * 60,
    image=modal.Image.debian_slim().pip_install(
        ["boto3", "psycopg2-binary", "pandas"]
    ),
    secrets=[modal.Secret.from_name("aws-secret"),
             modal.Secret.from_name("postgres-secret")]
)
def merge_s3_batches():
    s3 = boto3.client("s3")

    # List all batch files
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=BATCH_PREFIX)
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(".npy")]

    if not files:
        print("No batch files found.")
        return

    files.sort()  # ensure order

    arrays = []
    for key in files:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = obj['Body'].read()
        arr = np.load(io.BytesIO(data))
        arrays.append(arr)
        print(f"Loaded {key}, shape={arr.shape}")

    merged_array = np.vstack(arrays)
    print(f"Merged array shape: {merged_array.shape}")

    # Save merged array to S3 directly
    buf = io.BytesIO()
    np.save(buf, merged_array)
    buf.seek(0)
    s3.upload_fileobj(buf, BUCKET_NAME, MERGED_KEY)
    print(f"Merged array uploaded to s3://{BUCKET_NAME}/{MERGED_KEY}")
    return {"merged_key": MERGED_KEY, "shape": merged_array.shape}

# Local entrypoint
@app.local_entrypoint()
def main():
    with app.run():
        handle = merge_s3_batches.spawn()
        result = handle.get()
        print(result)


if __name__ == "__main__":
    main()