import json
import numpy as np
from lcsh_benchmark.retrieval.backends import FakeBackend
from lcsh_benchmark.retrieval import embed_index


def _write_vocab(path):
    rows = [
        {"uri": "u/1", "label": "Sociology", "authority": "lcsh", "normalized": "sociology"},
        {"uri": "u/2", "label": "Music", "authority": "lcsh", "normalized": "music"},
        {"uri": "u/3", "label": "Fiction", "authority": "lcgft", "normalized": "fiction"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_build_then_load_roundtrip(tmp_path):
    vp = tmp_path / "vocab.jsonl"; _write_vocab(vp)
    idx_dir = tmp_path / "index"
    be = FakeBackend(dim=16)
    idx = embed_index.build(str(vp), be, str(idx_dir), batch_size=2)
    assert idx.vectors.shape == (3, 16)
    assert idx.labels == ["Sociology", "Music", "Fiction"]
    # second call loads from cache without re-embedding
    idx2 = embed_index.build(str(vp), be, str(idx_dir), batch_size=2)
    assert np.allclose(idx.vectors, idx2.vectors)
    assert idx2.from_cache is True


def test_build_resumes_after_failed_batch_without_recomputing(tmp_path):
    import pytest
    vp = tmp_path / "vocab.jsonl"; _write_vocab(vp)   # 3 rows
    idx_dir = tmp_path / "index"

    class FlakyBackend:
        name = "flaky-4"
        def __init__(self): self.encoded = []
        def encode(self, texts):
            self.encoded.append(list(texts))
            if texts and texts[0] == "Fiction" and not getattr(self, "allow", False):
                raise RuntimeError("boom")           # fail the 2nd batch first time
            return np.ones((len(texts), 4), dtype=np.float32)
        def cost_usd(self, texts): return 0.0

    be = FlakyBackend()
    # batch1=[Sociology,Music] writes+checkpoints; batch2=[Fiction] raises
    with pytest.raises(RuntimeError):
        embed_index.build(str(vp), be, str(idx_dir), batch_size=2)

    be.allow = True
    be.encoded.clear()
    idx = embed_index.build(str(vp), be, str(idx_dir), batch_size=2)
    assert idx.vectors.shape == (3, 4)
    # resume re-encoded ONLY the failed batch, not the completed rows
    assert be.encoded == [["Fiction"]]
