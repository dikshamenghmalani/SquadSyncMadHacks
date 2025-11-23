from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .constants import GIF_DIR, LOG_DIR


class Scenario:
    def __init__(
        self,
        *,
        name: str,
        gif_name: str,
        log_name: str,
        group_filter: Optional[List[str]] = None,
        custom_steps: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.name = name
        self.gif_path = GIF_DIR / gif_name
        self.log_path = LOG_DIR / log_name
        self.group_filter = group_filter
        self.custom_steps = custom_steps or {}


SCENARIOS: List[Scenario] = [
    Scenario(
        name="Blue-only visualization",
        gif_name="blue_only.gif",
        log_name="alerts_blue.log",
        group_filter=["Blue"],
        custom_steps={
            "Blue-H2": {"pace": 1.4},
            "Blue-H3": {"pace": 0.85},
        },
    ),
    Scenario(
        name="Slow blue merges into orange",
        gif_name="blue_slow.gif",
        log_name="alerts_blue_slow.log",
        custom_steps={
            "Blue-H4": {"pace": 0.4, "progress": 14.0},
        },
    ),
    Scenario(
        name="Fast blue merges forward",
        gif_name="blue_fast.gif",
        log_name="alerts_blue_fast.log",
    ),
]

