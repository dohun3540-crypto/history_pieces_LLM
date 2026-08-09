"""Bounded KCI public-research pass for direct Mokpo history abstracts."""

from __future__ import annotations

from pathlib import Path

import collect_history_candidates_round2 as campaign


def main() -> int:
    original = campaign.source_configs
    campaign.source_configs = lambda: tuple(
        item for item in original() if item.source_id == "kci_mokpo_history"
    )
    campaign.BATCH_ID = "round2-high-precision-004"
    campaign.HIGH_CONFIDENCE_TARGET = 158
    campaign.HARD_STORED_CAP = 218
    campaign.GLOBAL_NETWORK_CEILING = 145
    campaign.CHECKPOINTS = (50, 100, 150, 158)
    result = campaign.run(Path.cwd().resolve())
    print(campaign.json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
