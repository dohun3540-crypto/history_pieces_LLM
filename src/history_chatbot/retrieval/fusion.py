"""dense와 sparse 순위를 reciprocal-rank 방식으로 융합한다."""

from __future__ import annotations

from collections.abc import Sequence

from history_chatbot.retrieval.base import RankedChunk


def reciprocal_rank_fusion(
    dense: Sequence[RankedChunk],
    sparse: Sequence[RankedChunk],
    *,
    rank_constant: int = 10,
) -> list[RankedChunk]:
    combined: dict[str, dict[str, object]] = {}
    for method, results in (("dense", dense), ("sparse", sparse)):
        for rank, result in enumerate(results, start=1):
            entry = combined.setdefault(
                result.chunk.chunk_id,
                {"chunk": result.chunk, "score": 0.0, "methods": [], "dense": 0.0, "sparse": 0.0},
            )
            entry["score"] = float(entry["score"]) + (
                1.0 / (rank_constant + rank)
            )
            cast_methods = entry["methods"]
            assert isinstance(cast_methods, list)
            cast_methods.append(method)
            entry[method] = result.score
    return sorted(
        (
            RankedChunk(
                chunk=entry["chunk"],  # type: ignore[arg-type]
                score=float(entry["score"]) / (2.0 / (rank_constant + 1)),
                methods=tuple(entry["methods"]),  # type: ignore[arg-type]
                dense_score=float(entry["dense"]),
                sparse_score=float(entry["sparse"]),
            )
            for entry in combined.values()
        ),
        key=lambda item: (-item.score, item.chunk.chunk_id),
    )
