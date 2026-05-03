"""Embedding (de)serialization with storage-dtype quantization.

Stored embeddings carry a 1-byte header indicating the on-disk dtype:

    byte 0 : DTYPE_FLOAT32 (0) | DTYPE_FLOAT16 (1)
    bytes 1+ : raw embedding bytes in that dtype

`decode_embedding` always returns a float32 numpy array regardless of
storage dtype, so downstream consumers (cosine similarity, RRF, CE
rerank) need no changes. The header is forward-compatible — int8
(DTYPE_INT8 = 2) is reserved for a future asymmetric-quantization pass.

Default is float16 (halves the embedding column on disk; ~23% total DB
shrink). Float16 round-trip error on L2-normalised vectors is ~5e-4 max,
well below ranking-relevant cosine-similarity granularity (validated by
the v0.4 candidate eval suite — full ACCEPT-NEUTRAL on LoCoMo + 6-cat).
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


