# test_vector_tagging.py
import io
import sys
import numpy as np
import pytest

import final_project.vector_tagging.LLM.tagging_loop as mod


# ---------------------------
# Fixtures & simple fakes
# ---------------------------

class FakeLLM:
    def __init__(self, vector_definitions):
        # just store for sanity checks
        self.vector_definitions = vector_definitions

    def tag_vectors(self, batch):
        # Return (original_vector, "desc_i") for each row
        return [(batch[i], f"desc_{i}") for i in range(len(batch))]

    def encode_text(self, texts):
        # Return a simple fixed embedding for each text
        return [np.array([1.0, 2.0, 3.0], dtype=np.float32) for _ in texts]


class FakeCall:
    """Simulates a Modal FunctionCall."""
    def __init__(self, result):
        self._result = result

    def get(self, timeout=None):
        # Ignore timeout; always return immediately
        return self._result


# ---------------------------
# Test batch_worker (local)
# ---------------------------

def test_batch_worker_local(monkeypatch):
    # Ensure global llm starts clean
    monkeypatch.setattr(mod, "llm", None, raising=False)

    # Patch LLM to a lightweight fake
    monkeypatch.setattr(mod, "BioMistralVectorTagger", FakeLLM)

    # Tiny fake batch
    vec_batch = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    definition = ["feat1", "feat2"]

    # Run the Modal function locally
    res = mod.batch_worker.local(vec_batch, 0, np.array(definition, dtype=object))

    assert res["index"] == 0
    assert isinstance(res["result"], list)
    assert len(res["result"]) == len(vec_batch)

    # Each entry: [original_vector, text_desc, embedding]
    orig, desc, emb = res["result"][0]
    assert np.allclose(orig, vec_batch[0])
    assert isinstance(desc, str)
    assert emb.shape == (3,)


# ---------------------------
# Test orchestrator tag_ehr_vectors with IO mocked
# ---------------------------

def test_tag_ehr_vectors_orchestrator(monkeypatch):
    # ---- 1. Fake boto3 module so that "import boto3" inside the function uses this ----
    class FakeS3:
        def __init__(self):
            self.uploads = []

        def download_file(self, Bucket, Key, Filename):
            # Do nothing; np.load and open will be mocked anyway
            pass

        def upload_file(self, Filename, Bucket, Key):
            self.uploads.append((Filename, Bucket, Key))

    fake_s3 = FakeS3()

    class FakeBoto3Module:
        def client(self, name):
            assert name == "s3"
            return fake_s3

    # Ensure any "import boto3" sees our fake module
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3Module())

    # ---- 2. Mock np.load to bypass real files ----
    fake_vectors = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    monkeypatch.setattr(mod.np, "load", lambda path: fake_vectors)

    # ---- 3. Mock open + csv.reader to produce a fake definition row ----
    fake_def_row = ["feat1", "feat2", "feat3"]

    def fake_open(path, *args, **kwargs):
        # The function expects open(..., newline='') and passes the handle to csv.reader.
        # We return a StringIO whose content will be consumed by csv.reader,
        # but we override csv.reader anyway, so this content is irrelevant.
        return io.StringIO("dummy")

    monkeypatch.setattr(mod, "open", fake_open, raising=False)

    # csv.reader(f) -> [fake_def_row]
    monkeypatch.setattr(mod.csv, "reader", lambda f: [fake_def_row])

    # ---- 4. Patch LLM to fake one ----
    monkeypatch.setattr(mod, "BioMistralVectorTagger", FakeLLM)

    # ---- 5. Patch batch_worker.spawn to simulate remote GPU calls ----
    def fake_spawn(batch, idx, definition):
        # Check that we got the expected definition row
        assert definition == fake_def_row

        # Each row in `batch` becomes [orig_vector, "desc_i", embedding]
        fake_output = [
            [batch[i], f"desc_{i}", np.array([9.0, 9.0], dtype=np.float32)]
            for i in range(len(batch))
        ]
        return FakeCall({"result": fake_output})

    monkeypatch.setattr(mod.batch_worker, "spawn", fake_spawn)

    # ---- 6. Patch np.save to avoid writing files ----
    monkeypatch.setattr(mod.np, "save", lambda *a, **k: None)

    # ---- 7. Run orchestrator locally ----
    # This calls the raw function synchronously in the current process.
    result = mod.tag_ehr_vectors.local()

    # Function doesn't return anything; just make sure it completes
    assert result is None
