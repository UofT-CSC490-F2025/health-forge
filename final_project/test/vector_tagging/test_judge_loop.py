# test_ehr_label_judge.py

import json
import numpy as np
import torch
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
        self.uploads.append((Filename, Bucket, Key))

    def download_fileobj(self, Bucket, Key, Fileobj):
        """For normal-mode test we will override behavior via monkeypatch."""
        self.download_calls.append((Bucket, Key))
        # Default: write nothing (tests that rely on this should override)


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


# ---------------------------------------------------
# TEST 1: test_mode=True (no S3 download, only upload)
# ---------------------------------------------------

def test_run_judge_on_s3_test_mode(monkeypatch, tmp_path):
    """
    Smoke test for the 'test_mode=True' path:
    - no S3 downloads
    - one S3 upload
    - returns expected structure
    """

    # ---- Mock HF components ----
    monkeypatch.setattr(
        ej, "AutoTokenizer",
        type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyTokenizer())}),
    )
    monkeypatch.setattr(
        ej, "AutoModelForCausalLM",
        type("AM", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyModel())}),
    )
    monkeypatch.setattr(ej, "pipeline", fake_pipeline_factory_fixed_json)

    # ---- Mock S3 client (use only upload_file in test_mode) ----
    fake_s3 = FakeS3Client()
    monkeypatch.setattr(ej.boto3, "client", lambda *_args, **_kwargs: fake_s3)

    # Call the function under test in test_mode
    # If you're using Modal, and direct calling doesn't work,
    # you might need to do ej.run_judge_on_s3.local(test_mode=True)
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


# ---------------------------------------------------
# TEST 2: test_mode=False (normal S3 path, small fake data)
# ---------------------------------------------------

def test_run_judge_on_s3_normal_mode(monkeypatch, tmp_path):
    """
    Test the 'test_mode=False' path with a tiny fake S3 dataset:
    - we override LIMIT so it only iterates a couple of items
    - we fake S3 downloads with small arrays / CSV
    - we fake HF pipeline to always return valid JSON
    """

    # ---- Mock HF components ----
    monkeypatch.setattr(
        ej, "AutoTokenizer",
        type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyTokenizer())}),
    )
    monkeypatch.setattr(
        ej, "AutoModelForCausalLM",
        type("AM", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyModel())}),
    )
    monkeypatch.setattr(ej, "pipeline", fake_pipeline_factory_fixed_json)

    # ---- Prepare Fake S3 with custom download behavior ----
    fake_s3 = FakeS3Client()

    def fake_download_fileobj(Bucket, Key, Fileobj):
        fake_s3.download_calls.append((Bucket, Key))

        # We need to supply different contents depending on key:
        if Key == "original_vectors.npy":
            # small 2x4 array of patient vectors
            arr = np.array(
                [
                    [1, 70, 0, 5],
                    [0, 60, 1, 3],
                ],
                dtype=float,
            )
            # np.save writes binary content to Fileobj
            np.save(Fileobj, arr)

        elif Key == ej.DEF_CSV_KEY:
            # A CSV with 4 column names (matching vector length)
            csv_text = "gender,age,isDead,feat3\n0,0,0,0\n"
            Fileobj.write(csv_text.encode("utf-8"))

        elif Key == "vector_tags.npy":
            labels = np.array(["label one", "label two"], dtype=object)
            np.save(Fileobj, labels)

        else:
            # Any unexpected key: write nothing
            pass

    fake_s3.download_fileobj = fake_download_fileobj
    monkeypatch.setattr(ej.boto3, "client", lambda *_args, **_kwargs: fake_s3)

    # ---- Limit the number of vectors processed ----
    monkeypatch.setattr(ej, "LIMIT", 2)

    # ---- Call function in normal mode ----
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

    # ---- S3 uploads: judge_results.json + ground_truth_labels.npy ----
    uploaded_keys = [key for (_fname, _bucket, key) in fake_s3.uploads]
    assert "judge_results.json" in uploaded_keys
    assert "ground_truth_labels.npy" in uploaded_keys

    # ---- S3 downloads: original_vectors, defs.csv, labels.npy ----
    downloaded_keys = [key for (_bucket, key) in fake_s3.download_calls]
    assert "original_vectors.npy" in downloaded_keys
    assert ej.DEF_CSV_KEY in downloaded_keys
    assert "vector_tags.npy" in downloaded_keys
