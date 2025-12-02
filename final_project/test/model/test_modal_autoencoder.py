# test_modal_autoencoder.py

import sys
import types
import numpy as np
import torch
import pytest

import final_project.modal_train_autoencoder as tr
import final_project.autoencoder as real_ae  # your real AE module


class FakeS3Client:
    def __init__(self):
        self.download_calls = []   # list of (Bucket, Key)
        self.uploads = []          # list of (Filename, Bucket, Key)

    def download_fileobj(self, Bucket, Key, Fileobj):
        self.download_calls.append((Bucket, Key))
        # we don't actually need to write into Fileobj for this test

    def upload_file(self, Filename, Bucket, Key):
        self.uploads.append((Filename, Bucket, Key))


def test_train_worker_runs_and_uploads_model(monkeypatch):
    # ---- 1. Provide a fake top-level "autoencoder" module ----
    # train_worker does: from autoencoder import EHRLatentAutoencoder
    fake_auto_mod = types.ModuleType("autoencoder")
    fake_auto_mod.EHRLatentAutoencoder = real_ae.EHRLatentAutoencoder
    monkeypatch.setitem(sys.modules, "autoencoder", fake_auto_mod)

    # ---- 2. Mock S3 client ----
    fake_s3 = FakeS3Client()
    monkeypatch.setattr(
        tr.boto3,
        "client",
        lambda *_args, **_kwargs: fake_s3,
    )

    # ---- 3. Mock np.load for training dataset ----
    def fake_np_load(path, *args, **kwargs):
        # Training code expects a 2D array, normalizes cols 1 and 3,
        # and asserts values in [0,1]. So we keep it in [0,1].
        # Shape: 40 samples x 6 features (>=4 cols; uses cols 1 & 3)
        return np.random.rand(40, 6).astype(np.float32)

    monkeypatch.setattr(tr.np, "load", fake_np_load)

    # ---- 4. Mock torch.save to avoid writing to disk ----
    saved_paths = []

    def fake_torch_save(state_dict, path):
        saved_paths.append(path)
        # no real file IO

    monkeypatch.setattr(torch, "save", fake_torch_save)

    # ---- 5. Force CPU & safe CUDA behavior ----
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    # get_device_name won't be called if device_count() == 0

    # ---- 6. Pretend /root/ehr_autoencoder_best.pt exists ----
    def fake_exists(path):
        if path == "/root/ehr_autoencoder_best.pt":
            return True
        return False

    monkeypatch.setattr(tr.os.path, "exists", fake_exists)

    # ---- 7. Run the Modal function locally ----
    result = tr.train_worker.local()
    assert result is None  # train_worker doesn't return anything

    # ---- 8. Assert S3 behavior ----

    # At least one download for DATA_KEY
    assert len(fake_s3.download_calls) >= 1
    bucket_0, key_0 = fake_s3.download_calls[0]
    assert bucket_0 == tr.BUCKET
    assert key_0 == tr.DATA_KEY

    # torch.save should have been called on the "best" checkpoint
    assert any("ehr_autoencoder_best.pt" in p for p in saved_paths)

    # Upload of best model to S3 should have been attempted
    uploaded_keys = [key for (_fname, _bucket, key) in fake_s3.uploads]
    assert tr.MODEL_OUTPUT_KEY in uploaded_keys
