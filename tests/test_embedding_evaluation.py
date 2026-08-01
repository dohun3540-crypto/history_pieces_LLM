import json
from pathlib import Path

from history_chatbot.retrieval.evaluation import load_cases
from history_chatbot.retrieval.service import RetrievalConfig


def test_multilingual_evaluation_shape_and_ground_truth_ids() -> None:
    path = Path("tests/fixtures/retrieval/multilingual_e5_evaluation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = load_cases(path)
    assert len([x for x in cases if x.locale == "ko" and x.should_answer]) >= 10
    assert len([x for x in cases if x.locale == "zh-CN" and x.should_answer]) >= 10
    assert len(payload["paired_queries"]) >= 5
    assert len([x for x in cases if x.topic == "무관"]) >= 5
    assert len([x for x in cases if x.topic == "근거 부족"]) >= 5
    corpus = [json.loads(line) for line in Path("data/provisional_hackathon/processed/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    chunk_ids = {x["chunk_id"] for x in corpus}
    source_ids = {x["source_id"] for x in corpus}
    assert all(set(case.expected_chunk_ids) <= chunk_ids for case in cases)
    assert all(set(case.expected_source_ids) <= source_ids for case in cases)


def test_e5_candidate_config_is_explicit_and_does_not_replace_hashing_default() -> None:
    candidate = RetrievalConfig.load(Path("configs/retrieval.e5.candidate.yaml"))
    default = RetrievalConfig.load(Path("configs/retrieval.yaml"))
    assert candidate.embedding_model == "intfloat/multilingual-e5-small"
    assert candidate.embedding_revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert candidate.minimum_dense_score == 0.82
    assert candidate.rrf_k == 10
    assert default.embedding_model == "hashing-v1"
