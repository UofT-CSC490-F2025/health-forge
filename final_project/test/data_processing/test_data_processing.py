import numpy as np
import pytest

import final_project.data_processing.EHR_data_processing_modal_final as mod


# ---------------------------
# Pure helper tests
# ---------------------------

def test_normalize_code_strip():
    assert mod.normalize_code_strip(" e11.65 ") == "E1165"
    assert mod.normalize_code_strip("250.00") == "25000"
    assert mod.normalize_code_strip(None) is None


def test_collapse_to_icd10_3():
    assert mod.collapse_to_icd10_3("E1165") == "E11"
    assert mod.collapse_to_icd10_3("E11") == "E11"
    assert mod.collapse_to_icd10_3("A9") == "A9"
    assert mod.collapse_to_icd10_3(None) is None


def test_load_gem_from_s3_parses_mapping(monkeypatch):
    # Fake S3 body / client
    class FakeBody:
        def read(self):
            return b"A10 B20 0\nA10 B21 1\nB01 C33 0\n"

    class FakeS3:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "key"
            return {"Body": FakeBody()}

    def fake_boto3_client(service_name):
        assert service_name == "s3"
        return FakeS3()

    monkeypatch.setattr(mod.boto3, "client", fake_boto3_client)

    mapping = mod.load_gem_from_s3("bucket", "key")

    # Order of lists is not guaranteed, so compare as sets
    assert set(mapping["A10"]) == {"B20", "B21"}
    assert set(mapping["B01"]) == {"C33"}


# ---------------------------
# Helpers for batch_worker tests
# ---------------------------

class FakeCursor:
    """
    Very simple cursor that returns controlled data depending on the last query.
    We pattern-match on pieces of the SQL string, which is enough for this test.
    """
    def __init__(self):
        self.last_query = None
        self.last_params = None

    def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params

    def fetchall(self):
        q = self.last_query or ""
        params = self.last_params

        # patients for the current batch
        if "FROM patients" in q and "WHERE subject_id = ANY" in q:
            subject_ids = params[0]
            # subject_id, gender, anchor_age, dod
            return [(sid, "M", 40.0 + sid, None) for sid in subject_ids]

        # last admission per subject
        if "FROM (" in q and "FROM admissions" in q and "rn = 1" in q:
            subject_ids = params[0]
            # subject_id, marital_status, race
            return [(sid, "MARRIED", "WHITE") for sid in subject_ids]

        # total admissions per subject
        if "SELECT subject_id, COUNT(*) as total" in q and "FROM admissions" in q:
            subject_ids = params[0]
            # subject_id, total
            return [(sid, 2) for sid in subject_ids]

        # diagnoses for this batch
        if "SELECT subject_id, icd_code, icd_version" in q and "FROM diagnoses_icd" in q:
            subject_ids = params[0]
            # one ICD9 for first subject, one ICD10 for second subject
            return [
                (subject_ids[0], "250.00", 9),   # ICD-9
                (subject_ids[1], "E11.65", 10),  # ICD-10
            ]

        # distinct codes across entire DB
        if "SELECT DISTINCT icd_code, icd_version" in q and "FROM diagnoses_icd" in q:
            return [
                ("250.00", 9),
                ("E11.65", 10),
            ]

        # enums for marital_status
        if "SELECT DISTINCT marital_status FROM admissions" in q:
            return [("MARRIED",), ("SINGLE",)]

        # enums for race
        if "SELECT DISTINCT race FROM admissions" in q:
            return [("WHITE",), ("BLACK",)]

        # fallback
        return []

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


class FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        # Just record what was uploaded; don't actually talk to AWS
        self.uploads.append((filename, bucket, key))




def test_batch_worker_builds_vectors_and_uploads(monkeypatch, tmp_path):
    # Monkeypatch DB connection
    def fake_connect(*args, **kwargs):
        return FakeConnection()

    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    # Monkeypatch GEM loading so we don't hit S3
    def fake_load_gem_from_s3(bucket, key):
        return {"25000": ["E1165"]}

    monkeypatch.setattr(mod, "load_gem_from_s3", fake_load_gem_from_s3)

    # --- Monkeypatch torch.cuda so we don't need a real GPU ---
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod.torch.cuda, "get_device_name", lambda idx: "FAKE_GPU")

    # --- Monkeypatch boto3.client for the S3 upload that batch_worker does ---
    fake_s3 = FakeS3()

    def fake_boto3_client(service_name):
        assert service_name == "s3"
        return fake_s3

    monkeypatch.setattr(mod.boto3, "client", fake_boto3_client)

    def do_nothing(a,b):
        pass

    monkeypatch.setattr(mod.np, "save", do_nothing) #don't save anything to disk
    # --- Run batch_worker on a tiny batch of 2 subjects ---
    subject_ids = [1, 2]
    res = mod.batch_worker.local(subject_ids, batch_index=0)

    # Verify return metadata
    assert res["batch_index"] == 0
    assert res["num_subjects"] == 2
    assert res["s3_key"] == f"{mod.S3_BATCH_PREFIX}batch_0.npy"

    # Verify that the upload was "performed"
    assert len(fake_s3.uploads) == 1
    uploaded_filename, uploaded_bucket, uploaded_key = fake_s3.uploads[0]

    assert uploaded_bucket == mod.BUCKET_NAME
    assert uploaded_key == f"{mod.S3_BATCH_PREFIX}batch_0.npy"


def test_main_exits_cleanly_with_no_patients(monkeypatch, caplog):
    class EmptyCursor:
        def __init__(self):
            self.last_query = None

        def execute(self, query, params=None):
            self.last_query = query

        def fetchall(self):
            # When main asks for all subject_ids, return empty list
            if "SELECT DISTINCT subject_id FROM patients" in (self.last_query or ""):
                return []
            return []

        def close(self):
            pass

    class EmptyConnection:
        def cursor(self):
            return EmptyCursor()

        def close(self):
            pass

    def fake_connect(*args, **kwargs):
        return EmptyConnection()

    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    # Avoid actually spawning any Modal workers if something in main changes
    monkeypatch.setattr(mod.batch_worker, "spawn", lambda *a, **k: pytest.fail("spawn should not be called when no patients"))

    with caplog.at_level("INFO"):
        mod.main()

    # We should see the "No patients found." warning
    assert any("No patients found." in record.getMessage() for record in caplog.records)
