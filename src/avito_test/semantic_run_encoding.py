"""Resume each saved range sequentially and freeze the complete E5 cache."""
# Resume from saved checkpoints without re-encoding completed ranges.
from __future__ import annotations

from .semantic_encode_segment import encode_segment
from .semantic_finalize_embeddings import SEGMENTS, finalize


if __name__ == "__main__":
    for start, end in SEGMENTS:
        encode_segment(start, end, threads=8)
    finalize()
