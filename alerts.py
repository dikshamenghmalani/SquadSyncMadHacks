"""Alert management utilities for the hiker mesh simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AlertEvent:
    step: int
    audience: str
    category: str
    actor: str
    message: str


class AlertManager:
    """Collects alert events and persists them to disk."""

    def __init__(self, output_path: Path | str = Path("alerts.log")) -> None:
        self.output_path = Path(output_path)
        self.events: List[AlertEvent] = []

    def record(
        self,
        *,
        step: int,
        audience: str,
        category: str,
        actor: str,
        message: str,
    ) -> None:
        self.events.append(
            AlertEvent(
                step=step,
                audience=audience,
                category=category,
                actor=actor,
                message=message,
            )
        )

    def as_strings(self) -> List[str]:
        """Return human-readable strings for console output."""
        formatted = []
        for event in self.events:
            formatted.append(
                f"[t={event.step:02d}][{event.category}] "
                f"({event.audience}) {event.message}"
            )
        return formatted

    def flush(self) -> None:
        """Persist the alerts to disk."""
        if not self.events:
            self.output_path.write_text("No alerts recorded.\n", encoding="utf-8")
            return

        lines = [
            f"{event.step:02d}\t{event.category}\t{event.audience}\t"
            f"{event.actor}\t{event.message}"
            for event in self.events
        ]
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

