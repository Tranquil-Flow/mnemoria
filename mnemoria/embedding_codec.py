"""Embedding (de)serialization with optional storage-dtype quantization.

v0.4 candidate C7. Stored embeddings carry a 1-byte header indicating the
on-disk dtype:

    byte 0 : DTYPE_FLOAT32 (0) | DTYPE_FLOAT16 (1)
    bytes 1+ : raw embedding bytes in that dtype

`decode_embedding` always returns a float32 numpy array regardless of the
storage dtype, so downstream consumers (cosine similarity, RRF, CE
rerank) need no changes.

`encode_embedding(arr, dtype)` accepts dtype strings ``"float32"`` or
``"float16"``. ``"float16"`` halves the on-disk size of the embedding
column (1536 B → 768 B at dim=384), which is by far the dominant per-fact
storage cost (~94% of the per-fact bytes).

Float16 has a max representable magnitude of ±65504 and ~3 decimal
digits of precision. Sentence-transformer outputs are L2-normalised
(per `mnemoria/embeddings.py` — `normalize_embeddings=True`), so values
stay in [-1, 1]. Float16 round-trip error on this range is ~5e-4 max,
which is well below the cosine-similarity granularity that affects
ranking. Calibrated by the C7 evaluation; see
`docs/research/V0.4_C7_EVAL.md`.

int8 (DTYPE_INT8 = 2) is reserved for a future enhancement once float16
is validated. int8 needs a per-fact float32 scale prefix and asymmetric
quantization to preserve cosine-similarity accuracy; defer.

**Backward compatibility caveat (prototype scope)**: the prototype
assumes all reads see headered blobs (i.e. fresh stores). A migration
helper ``decode_legacy_float32(blob)`` is provided for callers that
need to read pre-C7 blobs that lack the header byte; production
deployment of C7 will need a one-time migration pass that re-encodes
every existing row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_INT8 = 2  # reserved; not implemented in the v0.4 candidate

VALID_STORAGE_DTYPES = ("float32", "float16")


def encode_embedding(arr: Optional[np.ndarray], dtype: str = "float32") -> Optional[bytes]:
    """Serialize an embedding to bytes with a 1-byte dtype header.

    The input array can be any numpy dtype; it is cast to the requested
    storage dtype before serialization.

    Returns None when ``arr`` is None.
    """
    if arr is None:
        return None
    if dtype == "float16":
        return bytes([DTYPE_FLOAT16]) + np.ascontiguousarray(arr, dtype=np.float16).tobytes()
    if dtype == "float32":
        return bytes([DTYPE_FLOAT32]) + np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    raise ValueError(
        f"unsupported embedding storage dtype: {dtype!r}; expected one of {VALID_STORAGE_DTYPES}"
    )


def decode_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize a headered blob back to a float32 numpy array.

    Returns None if blob is None or empty. Always returns float32 so
    downstream consumers don't need to know the storage dtype.
    """
    if blob is None or len(blob) == 0:
        return None
    header = blob[0]
    payload = blob[1:]
    if header == DTYPE_FLOAT16:
        return np.frombuffer(payload, dtype=np.float16).astype(np.float32)
    if header == DTYPE_FLOAT32:
        return np.frombuffer(payload, dtype=np.float32).copy()
    raise ValueError(f"unknown embedding storage dtype tag: {header}")


def decode_legacy_float32(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode a pre-C7 blob that lacks the 1-byte header.

    Used by migration code or for backward-compatibility shims when reading
    a database that was populated by a pre-C7 mnemoria version. Treats the
    entire blob as raw float32 bytes.
    """
    if blob is None or len(blob) == 0:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()
