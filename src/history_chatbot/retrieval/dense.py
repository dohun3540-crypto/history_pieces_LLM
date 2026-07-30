"""다운로드 없이 동작하는 개발용 dense 검색과 교체 가능한 인코더."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from history_chatbot.retrieval.base import DenseEncoder, RankedChunk, VectorStore
from history_chatbot.retrieval.query_normalizer import tokenize


class HashingDenseEncoder(DenseEncoder):
    """단어와 문자 n-gram을 해싱하는 개발·테스트용 로컬 인코더.

    신경망 의미 임베딩을 가장하지 않는다. 실제 의미 검색은 승인된
    SentenceTransformer 인코더로 이 인터페이스를 교체해야 한다.
    """

    model_id = "hashing-v1"
    revision = "builtin"

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension은 양수여야 합니다.")
        self.dimension = dimension

    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        del is_query
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        features: list[str] = list(tokenize(text))
        compact = "".join(features)
        features.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimension
            vector[position] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class DenseSearcher:
    def __init__(self, encoder: DenseEncoder, store: VectorStore) -> None:
        self.encoder = encoder
        self.store = store

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        if limit <= 0 or not self.store.chunks():
            return []
        vector = self.encoder.encode([query], is_query=True)[0]
        return self.store.search(vector, limit)
