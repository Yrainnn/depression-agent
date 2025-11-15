"""Strategy description templates used to enrich LLM prompts."""

from __future__ import annotations

from typing import Dict, Mapping

StrategyDescription = Mapping[str, str]

STRATEGY_DESCRIPTIONS: Dict[str, StrategyDescription] = {
    "S5": {
        "name": "元认知",
        "description": "探索用户对自身情绪的认知和反思",
        "tone": "探索性、反思性、非评判性",
    },
}
