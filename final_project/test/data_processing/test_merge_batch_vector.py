# test_merge_s3_batches.py
import io
import numpy as np
import pytest

# ⬇️ CHANGE THIS to your actual module path
# e.g. import final_project.vector_tagging.merge_batches as mod
import final_project.data_processing.merge_batch_vector as mod


# ---------------------------
# Fakes
# ---------------------------

class FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class FakeS3:
    def __init__(self, arrays_by_key):
        """
        arrays_by_key: dict[key -> np.ndarray]
        """
        self.arrays_by_key = arrays_by_key
        self.uploaded = None  # (bytes, bucket, key)

    def list_objects_v2(self, Bucket, Prefix):
        contents = [{"Key": k} for k in self.arrays_by_key.keys() if k.startswith(Prefix)]
        return {"Contents": contents} if contents else {}

    def get_object(self, Bucket, Key):
        arr = self.arrays_by_key[Key]
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        return {"Body": FakeBody(buf.read())}

    def upload_fileobj(self, fileobj, Bucket, Key):
        # capture uploaded bytes for inspection
        data = fileobj.read()
        self.uploaded = (data, Bucket, Key)


# ---------------------------
# merge_s3_batches: happy path
# ---------------------------

def test_merge_s3_batches_merges_arrays(monkeypatch):
    # Prepare two fake .npy "files" in S3
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[5, 6]], dtype=np.float32)

    arrays_by_key = {
        f"{mod.BATCH_PREFIX}batch_000.npy": arr1,
        f"{mod.BATCH_PREFIX}batch_001.npy": arr2,
    }

    fake_s3 = FakeS3(arrays_by_key)

    def fake_boto3_client(service_name):
        assert service_name == "s3"
        return fake_s3

    # Override the boto3 module inside our target module
    monkeypatch.setattr(
        mod,
        "boto3",
        type("FakeBoto3Module", (), {"client": staticmethod(fake_boto3_client)}),
    )

    # Call the Modal function locally
    res = mod.merge_s3_batches.local()

    # Check metadata
    assert res["merged_key"] == mod.MERGED_KEY
    assert res["shape"] == (3, 2)  # 2+1 rows, 2 columns

    # Check what was uploaded
    assert fake_s3.uploaded is not None
    data, bucket, key = fake_s3.uploaded
    assert bucket == mod.BUCKET_NAME
    assert key == mod.MERGED_KEY
