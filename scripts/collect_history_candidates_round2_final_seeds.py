"""Final bounded exact-seed pass within the Round 2 global network ceiling."""

from __future__ import annotations

from pathlib import Path

import collect_history_candidates_round2 as campaign


def config(source_id: str, institution: str, publisher: str, host: str, paths: tuple[str, ...], target: int):
    return campaign.SourceConfig(
        source_id, institution, publisher, (host,),
        tuple(campaign.https(host, path) for path in paths),
        (r"^/Article/E[0-9A-Za-z_-]+$",) if "ency" in source_id else (r"/(?:toc|dir|index)/GC[0-9]+$",),
        target, len(paths) + 1, 0, 1.5, "search_html",
    )


def final_configs():
    return (
        config("encykorea_exact_round2", "한국학중앙연구원", "aks_encyclopedia", "encykorea.aks.ac.kr", (
            "/Article/E0068032", "/Article/E0018741", "/Article/E0081172", "/Article/E0076630",
            "/Article/E0040078", "/Article/E0079741", "/Article/E0031035", "/Article/E0019127",
        ), 8),
        config("grandculture_daegu_round2", "한국학중앙연구원", "grandculture", "daegu.grandculture.net", (
            "/daegu/junggu/toc/GC40003992",
        ), 1),
        config("grandculture_gochang_round2", "한국학중앙연구원", "grandculture", "gochang.grandculture.net", (
            "/gochang/toc/GC02800907",
        ), 1),
        config("grandculture_yeongju_round2", "한국학중앙연구원", "grandculture", "yeongju.grandculture.net", (
            "/yeongju/dir/GC07401316",
        ), 1),
        config("grandculture_wanju_round2", "한국학중앙연구원", "grandculture", "wanju.grandculture.net", (
            "/wanju/index/GC07001377",
        ), 1),
    )


def main() -> int:
    campaign.source_configs = final_configs
    campaign.BATCH_ID = "round2-high-precision-006"
    campaign.HIGH_CONFIDENCE_TARGET = 158
    campaign.HARD_STORED_CAP = 218
    campaign.GLOBAL_NETWORK_CEILING = 17
    campaign.CHECKPOINTS = (50, 100, 150, 158)
    result = campaign.run(Path.cwd().resolve())
    print(campaign.json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
