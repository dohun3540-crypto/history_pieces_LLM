"""Final seven-request exact EncyKorea pass for Round 2."""

from __future__ import annotations

from pathlib import Path

import collect_history_candidates_round2 as campaign


def sources():
    host = "encykorea" + ".aks.ac.kr"
    paths = (
        "/Article/E0068032", "/Article/E0018741", "/Article/E0081172",
        "/Article/E0076630", "/Article/E0031035", "/Article/E0019127",
    )
    return (campaign.SourceConfig(
        "encykorea_exact_final", "한국학중앙연구원", "aks_encyclopedia", (host,),
        tuple(campaign.https(host, path) for path in paths),
        (r"^/Article/E[0-9A-Za-z_-]+$",), 6, 7, 0, 1.5, "search_html",
    ),)


def main() -> int:
    campaign.source_configs = sources
    campaign.BATCH_ID = "round2-high-precision-007"
    campaign.HIGH_CONFIDENCE_TARGET = 158
    campaign.HARD_STORED_CAP = 218
    campaign.GLOBAL_NETWORK_CEILING = 7
    campaign.CHECKPOINTS = (50, 100, 150, 158)
    result = campaign.run(Path.cwd().resolve())
    print(campaign.json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
