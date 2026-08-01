"""다운로드 없이 동작하는 개발용 dense 검색과 교체 가능한 인코더."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

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


class SentenceModel(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> object: ...
    def get_sentence_embedding_dimension(self) -> int: ...


class SentenceTransformerEncoder(DenseEncoder):
    """로컬 캐시에 있는 sentence-transformers 모델만 명시적으로 사용한다."""

    query_prefix = "query: "
    passage_prefix = "passage: "
    normalize_embeddings = True

    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-small",
        *,
        revision: str = "main",
        model: SentenceModel | None = None,
        cache_folder: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("sentence-transformers가 설치되어 있지 않습니다.") from error
            try:
                model = SentenceTransformer(
                    model_id,
                    revision=revision,
                    cache_folder=cache_folder,
                    local_files_only=True,
                    device="cpu",
                )
            except Exception as error:
                raise RuntimeError(
                    "임베딩 모델이 로컬 캐시에 없습니다. 자동 다운로드하지 않습니다."
                ) from error
        self.model = model
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        self.dimension = int(
            dimension_getter() if dimension_getter else model.get_sentence_embedding_dimension()
        )

    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        prefix = self.query_prefix if is_query else self.passage_prefix
        prepared = [prefix + text for text in texts]
        if not prepared:
            return []
        encoded = self.model.encode(
            prepared,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in encoded]  # type: ignore[union-attr]


class DenseSearcher:
    def __init__(self, encoder: DenseEncoder, store: VectorStore) -> None:
        self.encoder = encoder
        self.store = store

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        if limit <= 0 or not self.store.chunks():
            return []
        vector = self.encoder.encode([query], is_query=True)[0]
        return self.store.search(vector, limit)
