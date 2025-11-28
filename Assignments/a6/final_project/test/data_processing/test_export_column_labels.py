# test_column_export.py
import os
import io
import numpy as np
import pytest

import final_project.data_processing.export_column_labels as mod


# ---------------------------
# Fake DB / Cursor for psycopg2
# ---------------------------

class FakeCursor:
    def __init__(self):
        self.last_query = None

    def execute(self, query, params=None):
        self.last_query = query

    def fetchall(self):
        q = self.last_query or ""

        # marital statuses
        if "SELECT DISTINCT marital_status FROM admissions" in q:
            return [("Married",), ("Single",), (None,)]

        # races
        if "SELECT DISTINCT race FROM admissions" in q:
            return [("White",), ("Black",), (None,)]

        # ICD codes
        if "SELECT DISTINCT icd_code, icd_version FROM diagnoses_icd" in q:
            # (icd_code, icd_version)
            return [
                ("250.00", 9),   
                ("E11.65", 10),  
                ("A09", 10),    
            ]

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


# ---------------------------
# Fake S3 client
# ---------------------------

class FakeS3:
    def __init__(self):
        self.uploads = []
        self.objects = {}  # for get_object if needed

    def upload_file(self, local_path, bucket, key):
        # record uploads so we can assert later
        self.uploads.append((local_path, bucket, key))

    def get_object(self, Bucket, Key):
        # Used by load_dxccsr_mapping or load_gem_from_s3 if we don’t override them
        body = self.objects.get(Key, "")
        if isinstance(body, str):
            body = body.encode("utf-8")
        return {"Body": io.BytesIO(body)}


# ---------------------------
# Tests for small helpers
# ---------------------------

def test_normalize_code_strip():
    assert mod.normalize_code_strip(" e11.65 ") == "E1165"
    assert mod.normalize_code_strip("250.00") == "25000"
    assert mod.normalize_code_strip(None) is None


def test_collapse_to_icd10_3():
    assert mod.collapse_to_icd10_3("E1165") == "E11"
    assert mod.collapse_to_icd10_3("E11") == "E11"
    assert mod.collapse_to_icd10_3("A9") is None 
    assert mod.collapse_to_icd10_3(None) is None


def test_sanitize_description():
    assert mod.sanitize_description("Acute myocardial infarction") == "acute_myocardial_infarction"
    assert mod.sanitize_description(" Weird  stuff!! ") == "weird_stuff"
    assert mod.sanitize_description("___Already__clean__") == "already_clean"


# ---------------------------
# Main test: export_column_labels.local()
# ---------------------------

def test_export_column_labels_local(monkeypatch, tmp_path):
    """
    Run the Modal function locally with:
      - fake DB (psycopg2)
      - fake S3 (boto3.client)
      - fake GEM mapping
      - fake DXCCSR mapping
    and assert:
      - result dict has correct shape
      - CSV is created
      - S3 upload is called
    """

    # 1. Patch psycopg2.connect -> FakeConnection
    def fake_connect(*args, **kwargs):
        return FakeConnection()

    monkeypatch.setattr(mod.psycopg2, "connect", fake_connect)

    # 2. Patch boto3.client -> FakeS3
    fake_s3 = FakeS3()

    def fake_boto3_client(service_name):
        assert service_name == "s3"
        return fake_s3

    monkeypatch.setattr(mod.boto3, "client", fake_boto3_client)

    def fake_load_dxccsr_mapping():
        return {
            "E11": "Type 2 diabetes mellitus",
            "A09": "Infectious gastroenteritis",
        }

    monkeypatch.setattr(mod, "load_dxccsr_mapping", fake_load_dxccsr_mapping)

    def fake_load_gem_from_s3(bucket, key):
        return {"25000": ["E1165"]}

    monkeypatch.setattr(mod, "load_gem_from_s3", fake_load_gem_from_s3)

    # 5. Patch TEMP_PATH to point into pytest’s tmp_path, so we don’t write to real /tmp
    csv_path = tmp_path / "columns.csv"
    monkeypatch.setattr(mod, "TEMP_PATH", str(csv_path))

    result = mod.export_column_labels.local()

    # Basic sanity checks on return value
    assert isinstance(result, dict)
    assert result["s3_key"] == mod.FINAL_S3_KEY
    assert isinstance(result["num_columns"], int)
    assert result["num_columns"] > 0

    # CSV file should exist and have a single header row
    assert csv_path.exists()
    text = csv_path.read_text()
    header_cols = text.strip().split(",")
    assert len(header_cols) == result["num_columns"]

    # Check a few expected prefixes in the column names
    # Fixed ones:
    assert "gender" in header_cols
    assert "age" in header_cols
    assert "dod" in header_cols

    # From marital statuses (Married, Single, N/A)
    assert any(col.startswith("marital_") for col in header_cols)
    # From race values (White, Black, N/A)
    assert any(col.startswith("race_") for col in header_cols)

    # From ICD prefixes + DXCCSR mapping ("E11", "A09")
    assert any(col.startswith("E11_") for col in header_cols)
    assert any(col.startswith("A09_") for col in header_cols)

    # Ensure S3 upload was called once with our temp CSV
    assert len(fake_s3.uploads) == 1
    uploaded_local, uploaded_bucket, uploaded_key = fake_s3.uploads[0]
    assert uploaded_bucket == mod.BUCKET_NAME
    assert uploaded_key == mod.FINAL_S3_KEY
    # The path passed to upload should be our TEMP_PATH
    assert os.path.samefile(uploaded_local, csv_path)
