import pytest

from history_chatbot.retrieval.dense import SentenceTransformerEncoder


class FakeSentenceModel:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[1, 0, 0] for _ in texts]


def test_query_passage_prefix_normalization_and_metadata_properties() -> None:
    model = FakeSentenceModel()
    encoder = SentenceTransformerEncoder(revision="revision-1", model=model)
    assert encoder.encode(["질문"], is_query=True) == [[1.0, 0.0, 0.0]]
    assert model.calls[-1][0] == ["query: 질문"]
    assert model.calls[-1][1]["normalize_embeddings"] is True
    encoder.encode(["본문"], is_query=False)
    assert model.calls[-1][0] == ["passage: 본문"]
    assert encoder.dimension == 3


def test_missing_dependency_or_cache_never_falls_back_to_hashing(monkeypatch) -> None:
    import builtins
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="설치"):
        SentenceTransformerEncoder()
