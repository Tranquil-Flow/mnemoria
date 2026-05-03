"""v0.4 candidate C7 — embedding storage quantization tests.

Verify that float16 storage:
  1. Round-trips correctly via the codec
  2. Produces ~2× smaller stored embedding blobs
  3. Doesn't measurably regress recall accuracy on a focused corpus
  4. Co-exists with float32 (header byte makes blobs self-describing)
"""

import os
import tempfile

import numpy as np

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.embedding_codec import (
    encode_embedding,
    decode_embedding,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
)


def _make_store(*, dtype: str):
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    cfg.embedding_storage_dtype = dtype
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


# ── codec unit tests ─────────────────────────────────────────────────────


def test_codec_float32_roundtrip_exact():
    arr = np.array([0.1, 0.2, -0.3, 0.99], dtype=np.float32)
    blob = encode_embedding(arr, "float32")
    assert blob[0] == DTYPE_FLOAT32
    back = decode_embedding(blob)
    assert np.array_equal(arr, back)


def test_codec_float16_roundtrip_lossy_but_close():
    arr = np.array([0.1, 0.2, -0.3, 0.99], dtype=np.float32)
    blob = encode_embedding(arr, "float16")
    assert blob[0] == DTYPE_FLOAT16
    back = decode_embedding(blob)
    err = np.max(np.abs(arr - back))
    assert err < 1e-3, f"float16 max error {err} too large"


def test_codec_float16_halves_payload_size():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    blob32 = encode_embedding(v, "float32")
    blob16 = encode_embedding(v, "float16")
    # +1 for header
    assert len(blob32) == 1 + 384 * 4
    assert len(blob16) == 1 + 384 * 2
    assert len(blob16) * 2 - 1 == len(blob32)


def test_codec_cosine_preserved_on_normalized_384d():
    """The actual ranking signal — cosine similarity on a normalized
    embedding — must round-trip near-perfectly through float16."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    blob = encode_embedding(v, "float16")
    back = decode_embedding(blob)
    cos = float(np.dot(v, back) / (np.linalg.norm(v) * np.linalg.norm(back)))
    assert cos > 0.999999, f"float16 cosine round-trip {cos} below threshold"


def test_codec_handles_none():
    assert encode_embedding(None) is None
    assert decode_embedding(None) is None
    assert decode_embedding(b"") is None


# ── integration: store + recall in float16 mode ──────────────────────────


def test_store_recall_in_float16_mode_returns_correct_top1():
    """End-to-end: with float16 storage, recall still picks the right
    fact from a small substantive corpus."""
    tmpdir, store = _make_store(dtype="float16")
    try:
        store.store("The production database port is 5432.", category="factual", importance=0.7)
        store.store("API rate limit is 1000 requests per minute.", category="factual", importance=0.7)
        store.store("WebSocket heartbeats fire every 30 seconds.", category="factual", importance=0.7)
        store.store("JWT keys rotate every 24 hours.", category="factual", importance=0.7)
        store.advance_time(0.0001)

        results = store.recall("What port does the production database use?", top_k=2)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content.lower()
        assert "5432" in top1 or "production database" in top1, (
            f"expected database-port answer at top-1, got {top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_float16_db_smaller_than_float32():
    """Verify the actual DB size (or stored blob size) shrinks with float16.
    Inspect the SQLite blob column for an ingested fact and confirm it's
    ~half the float32 size."""
    import sqlite3

    sizes = {}
    for dtype in ("float32", "float16"):
        tmpdir, store = _make_store(dtype=dtype)
        try:
            store.store(
                "The conference room is on the third floor of building B.",
                category="factual", importance=0.7,
            )
            store.advance_time(0.0001)
            conn = sqlite3.connect(store._config.db_path)
            row = conn.execute("SELECT length(embedding) FROM um_facts WHERE embedding IS NOT NULL LIMIT 1").fetchone()
            sizes[dtype] = row[0]
            conn.close()
        finally:
            tmpdir.cleanup()

    assert sizes["float16"] < sizes["float32"], (
        f"float16 blob ({sizes['float16']}) should be smaller than float32 ({sizes['float32']})"
    )
    # Allow for header byte; ratio should be close to 0.5
    ratio = sizes["float16"] / sizes["float32"]
    assert 0.45 < ratio < 0.55, (
        f"float16/float32 ratio {ratio:.3f} not near expected 0.5 "
        f"({sizes['float16']} / {sizes['float32']})"
    )


def test_float16_and_float32_blobs_coexist_via_header():
    """A blob written as float32 and a blob written as float16 both decode
    correctly via decode_embedding — the header byte distinguishes them."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(64).astype(np.float32)
    v /= np.linalg.norm(v)
    blob32 = encode_embedding(v, "float32")
    blob16 = encode_embedding(v, "float16")
    back32 = decode_embedding(blob32)
    back16 = decode_embedding(blob16)
    assert np.array_equal(v, back32)
    assert np.allclose(v, back16, atol=1e-3)
