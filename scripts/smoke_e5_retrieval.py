"""실제 E5 인덱스의 기록새 검색/대화 smoke 결과를 JSON으로 기록한다."""

from __future__ import annotations

import json
import os
from pathlib import Path

from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.dense import SentenceTransformerEncoder
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig
from history_chatbot.runtime import RuntimeMode


ROOT = Path(__file__).resolve().parents[1]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def main() -> None:
    cache = Path(os.environ.get("HF_HOME", ROOT / ".runtime/model_cache/huggingface"))
    encoder = SentenceTransformerEncoder(MODEL, revision=REVISION, cache_folder=str(cache / "hub"))
    retrieval = HybridRetrievalService(
        RetrievalConfig(
            embedding_model=MODEL, embedding_revision=REVISION,
            local_storage_path=ROOT / ".runtime/indexes/hackathon/e5",
            provisional_chunks_path=ROOT / "data/provisional_hackathon/processed/chunks.jsonl",
            runtime_mode="hackathon", minimum_dense_score=.82,
        ),
        encoder=encoder,
    )
    errors = retrieval.validate_index()
    if errors:
        raise RuntimeError("; ".join(errors))
    chat = ConversationalRagOrchestrator(
        retrieval, MockLLM("확인 가능한 자료가 부족합니다."),
        SessionStore(RuntimeMode.HACKATHON), mode=RuntimeMode.HACKATHON,
    )

    def ask(case_id: str, query: str, **kwargs):
        response = chat.ask(query, **kwargs)
        return {
            "case_id": case_id, "query": query, "status": response.status,
            "situation": response.primary_situation_id, "grounded": response.grounded,
            "citation_count": len(response.sources),
            "retrieved_chunk_ids": list(response.retrieved_chunk_ids),
        }

    cases = [
        ask("ko_history", "목포는 언제 개항했나요?", locale="ko"),
        ask("zh_history", "朴爱顺的出生地和独立运动类别是什么？", locale="zh-CN"),
        ask("paired_ko", "김옥실은 어떤 독립운동에 참여했나요?", locale="ko"),
        ask("paired_zh", "金玉实参加了什么独立运动？", locale="zh-CN"),
        ask("unrelated", "양자컴퓨터의 큐비트 오류 정정 방법은?", locale="ko"),
        ask("insufficient", "목포 번화로 일본식 가옥을 설계한 건축가는 누구인가요?", locale="ko"),
        ask("piece_chat", "호남선 철도는 목포에서 언제 개통했나요?", locale="ko", conversation_mode="piece_chat", screen_type="piece_chat"),
        ask("free_chat", "박애순은 어떤 독립운동을 했나요?", locale="ko", conversation_mode="free_chat"),
    ]
    output = {
        "model": MODEL, "revision": REVISION,
        "metadata_validation_errors": errors,
        "hashing_fallback": "disabled; hashing-v1 requires explicit embedding_model",
        "citation_returned": any(item["citation_count"] for item in cases),
        "production_index_created": False,
        "cases": cases,
        "known_failure": (
            "근거 부족 질문은 관련 장소 청크를 검색해 retrieval 단계만으로 거절하지 못할 수 있다."
        ),
    }
    path = ROOT / "reports/e5_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
