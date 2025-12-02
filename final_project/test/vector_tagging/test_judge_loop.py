import json
import numpy as np
import pytest

import final_project.vector_tagging.JudgeLLM.modal_label_judge as ej


# -------------------------------
# Helpers for mocking HF + S3
# -------------------------------

class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        pass


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass


class FakeS3Client:
    def __init__(self):
        self.uploads = []           # list of (Filename, Bucket, Key)
        self.download_calls = []    # list of (Bucket, Key)

    def upload_file(self, Filename, Bucket, Key):
        # Do not touch disk; just record the call
        self.uploads.append((Filename, Bucket, Key))

    def download_fileobj(self, Bucket, Key, Fileobj):
        # In these tests we don't rely on the file contents at all,
        # so we only record the call and don't actually write.
        self.download_calls.append((Bucket, Key))


def fake_pipeline_factory_fixed_json(*args, **kwargs):
    """
    Returns a callable that ignores its inputs and always returns the same
    well-formed JSON text in the 'generated_text' field.
    """

    def _pipe(prompt, max_new_tokens, return_full_text, temperature):
        obj = {
            "scores": {
                "clinical_correctness": 5,
                "completeness_of_key_patient_information": 4,
                "precision_of_patient_information": 4,
                "conciseness": 4,
                "formatting": 4,
            },
            "explanation": "test explanation",
        }
        return [{"generated_text": json.dumps(obj)}]

    return _pipe


# -------------------------------
# In-memory file stubs
# -------------------------------

class InMemoryTextFile:
    def __init__(self):
        self.contents = ""

    def write(self, s):
        self.contents += s
        return len(s)

    def read(self, *args, **kwargs):
        return self.contents

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class InMemoryBinaryFile:
    def __init__(self):
        self.contents = bytearray()

    def write(self, b):
        # b is expected to be bytes
        self.contents.extend(b)
        return len(b)

    def read(self, *args, **kwargs):
        return bytes(self.contents)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


# ---------------------------------------------------
# TEST 1: test_mode=True (no S3 download, only upload)
# ---------------------------------------------------

def test_run_judge_on_s3_test_mode(monkeypatch):
    """
    Smoke test for the 'test_mode=True' path:
    - no S3 downloads
    - one S3 upload
    - returns expected structure
    - no real filesystem I/O
    """

    # ---- Mock HF components ----
    monkeypatch.setattr(
        ej,
        "AutoTokenizer",
        type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyTokenizer())}),
    )
    monkeypatch.setattr(
        ej,
        "AutoModelForCausalLM",
        type("AM", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyModel())}),
    )
    monkeypatch.setattr(ej, "pipeline", fake_pipeline_factory_fixed_json)

    # ---- Mock S3 client (use only upload_file in test_mode) ----
    fake_s3 = FakeS3Client()
    monkeypatch.setattr(ej.boto3, "client", lambda *_args, **_kwargs: fake_s3)

    # ---- Intercept open() so no real file is created ----
    # In test_mode, the code only does: open("/tmp/judge_test_results.json", "w")
    in_mem_judge_file = InMemoryTextFile()

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "/tmp/judge_test_results.json":
            return in_mem_judge_file
        raise AssertionError(f"Unexpected open() in test_mode=True: {file}, mode={mode}")

    monkeypatch.setattr("builtins.open", fake_open)

    # Call the function under test in test_mode
    result = ej.run_judge_on_s3.local(test_mode=True)

    # Basic shape checks
    assert isinstance(result, dict)
    assert result.get("status") == "test_ok"
    assert result.get("count") == 10

    correct_avg = result.get("correct_avg")
    misleading_avg = result.get("misleading_avg")

    # Both should be list of 5 (one per score dimension)
    assert isinstance(correct_avg, list) and len(correct_avg) == 5
    assert isinstance(misleading_avg, list) and len(misleading_avg) == 5

    # S3: should have exactly one upload (judge_test_results.json)
    assert len(fake_s3.uploads) == 1
    _, bucket, key = fake_s3.uploads[0]
    assert bucket == ej.BUCKET
    assert key == "judge_test_results.json"

    # Ensure we actually wrote some JSON into the in-memory file
    assert in_mem_judge_file.contents.strip().startswith("{")


# ---------------------------------------------------
# TEST 2: test_mode=False (normal S3 path, small fake data)
# ---------------------------------------------------

def test_run_judge_on_s3_normal_mode(monkeypatch):
    """
    Test the 'test_mode=False' path with a tiny fake S3 dataset:
    - override LIMIT so it only iterates a couple of items
    - fake S3 downloads (no real files)
    - fake HF pipeline with valid JSON
    - intercept ALL file I/O so nothing touches disk
    """

    # ---- Mock HF components ----
    monkeypatch.setattr(
        ej,
        "AutoTokenizer",
        type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyTokenizer())}),
    )
    monkeypatch.setattr(
        ej,
        "AutoModelForCausalLM",
        type("AM", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyModel())}),
    )
    monkeypatch.setattr(ej, "pipeline", fake_pipeline_factory_fixed_json)

    # ---- Fake S3 client (no actual file writes) ----
    fake_s3 = FakeS3Client()
    monkeypatch.setattr(ej.boto3, "client", lambda *_args, **_kwargs: fake_s3)

    # ---- Prepare fake in-memory data for the "loaded" arrays/CSV ----
    fake_vectors = np.array(
        [
            [1.0, 70.0, 0.0, 5.0],
            [0.0, 60.0, 1.0, 3.0],
        ],
        dtype=float,
    )
    fake_labels = np.array(["label one", "label two"], dtype=object)
    fake_columns = ["gender", "age", "isDead", "feat3"]

    # ---- Intercept np.load so it doesn't read real files ----
    def fake_np_load(path, *args, **kwargs):
        if path == "/tmp/vectors.npy":
            return fake_vectors
        if path == "/tmp/labels.npy":
            return fake_labels
        raise AssertionError(f"Unexpected np.load path: {path}")

    monkeypatch.setattr(ej.np, "load", fake_np_load)

    # ---- Intercept pd.read_csv so it doesn't read real files ----
    import pandas as pd

    def fake_read_csv(path, *args, **kwargs):
        if path == "/tmp/defs.csv":
            # Just return a DF with the right columns
            return pd.DataFrame(columns=fake_columns)
        raise AssertionError(f"Unexpected pd.read_csv path: {path}")

    monkeypatch.setattr(ej.pd, "read_csv", fake_read_csv)

    # ---- Intercept np.save for ground_truth_labels.npy so it doesn't write ----
    saved_ground_truth = []

    real_np_save = ej.np.save

    def fake_np_save(file, arr, *args, **kwargs):
        if file == "/tmp/ground_truth_labels.npy":
            # Just capture the data in-memory instead of writing
            saved_ground_truth.append(np.array(arr, copy=True))
            return
        # np.save is also used in other parts of the tests (e.g., fake S3),
        # so delegate to real_np_save for other paths
        return real_np_save(file, arr, *args, **kwargs)

    monkeypatch.setattr(ej.np, "save", fake_np_save)

    # ---- Intercept open() for all /tmp files, return in-memory objects ----
    # We only care that the code can write to them; the actual contents are
    # provided by the np.load / read_csv mocks above.
    judge_file = InMemoryTextFile()

    def fake_open(file, mode="r", *args, **kwargs):
        # Files that are written in the implementation:
        if file == "/tmp/judge_results.json":
            return judge_file

        # These are used only as binary sinks for S3 downloads; we don't need their data.
        if file in ("/tmp/vectors.npy", "/tmp/defs.csv", "/tmp/labels.npy"):
            return InMemoryBinaryFile()

        raise AssertionError(f"Unexpected open() in normal_mode: {file}, mode={mode}")

    monkeypatch.setattr("builtins.open", fake_open)

    # ---- Limit the number of vectors processed ----
    monkeypatch.setattr(ej, "LIMIT", 2)

    # ---- Call function in normal mode (no real I/O) ----
    result = ej.run_judge_on_s3.local(test_mode=False)

    # Basic shape checks
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert result.get("count") == 2

    mean_scores = result.get("mean_scores")
    assert isinstance(mean_scores, list)
    assert len(mean_scores) == 5

    # ground_truth_saved should be equal to number of vectors, since we
    # always return high scores from fake pipeline (mean >= 4)
    assert result.get("ground_truth_saved") == 2
    assert len(saved_ground_truth) == 1  # one np.save call for ground_truth_labels.npy

    # ---- S3 uploads: judge_results.json + ground_truth_labels.npy ----
    uploaded_keys = [key for (_fname, _bucket, key) in fake_s3.uploads]
    assert "judge_results.json" in uploaded_keys
    assert "ground_truth_labels.npy" in uploaded_keys

    # ---- S3 downloads: original_vectors, defs.csv, labels.npy ----
    downloaded_keys = [key for (_bucket, key) in fake_s3.download_calls]
    assert "original_vectors.npy" in downloaded_keys
    assert ej.DEF_CSV_KEY in downloaded_keys
    assert "vector_tags.npy" in downloaded_keys

    # ---- Check that judge_results.json was written in-memory ----
    assert judge_file.contents.strip().startswith("{")
    assert "mean_scores" in judge_file.contents
