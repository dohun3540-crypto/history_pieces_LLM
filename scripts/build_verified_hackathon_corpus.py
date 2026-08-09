from pathlib import Path
import json

from history_chatbot.history_collection.verified_corpus import build_verified_corpus
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = build_verified_corpus(
        root=ROOT,
        candidate_manifest=ROOT / "data/history_candidates/manifests/candidates.jsonl",
        output_root=ROOT / "data/history_verified",
    )
    retrieval = HybridRetrievalService(RetrievalConfig(
        runtime_mode="hackathon",
        verified_hackathon_chunks_path=ROOT / "data/history_verified/index_ready/chunks.jsonl",
        local_storage_path=ROOT / "data/history_verified/retrieval_index",
        final_top_k=10,
        max_chunks_per_document=2,
    ))
    index = retrieval.build_index(force=True)
    print(json.dumps({**report, "index_path": str(index.index_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
