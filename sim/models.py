from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .constants import (
    GROUP_LAYOUT,
    GROUP_SPEEDS,
    HIKERS_PER_GROUP,
    PERMANENT_RELAY_IDS,
    SIMULATION_STEPS,
    SPECIAL_RELAY_ID,
)
from .trail import to_world_position, trail_y


@dataclass
class Hiker:
    identifier: str
    group: str
    position: np.ndarray
    pace: float
    progress: float
    lane_offset: float
    last_x: float


def initialize_hikers(
    rng: np.random.Generator,
    group_subset: List[str] | None = None,
    custom_steps: Dict[str, Dict[str, float]] | None = None,
) -> List[Hiker]:
    selected_groups = (
        GROUP_LAYOUT
        if group_subset is None
        else {name: GROUP_LAYOUT[name] for name in group_subset if name in GROUP_LAYOUT}
    )
    if not selected_groups:
        raise ValueError("No valid groups available for initialization.")

    hikers: List[Hiker] = []
    for group_name, center in selected_groups.items():
        for member_index in range(HIKERS_PER_GROUP):
            progress = float(center[0] + rng.normal(loc=0.0, scale=0.4))
            progress = max(progress, 0.0)
            base_lane = float(center[1] - trail_y(center[0]))
            lane_offset = float(
                np.clip(
                    base_lane + rng.normal(loc=0.0, scale=0.15),
                    -0.9,
                    0.9,
                )
            )
            pace = float(np.clip(rng.normal(1.0, 0.18), 0.65, 1.4))
            identifier = f"{group_name}-H{member_index + 1}"

            if custom_steps and identifier in custom_steps:
                overrides = custom_steps[identifier]
                progress = overrides.get("progress", progress)
                pace = overrides.get("pace", pace)
                lane_offset = overrides.get("lane_offset", lane_offset)

            position = to_world_position(progress, lane_offset)

            hikers.append(
                Hiker(
                    identifier=identifier,
                    group=group_name,
                    position=position,
                    pace=pace,
                    progress=progress,
                    lane_offset=lane_offset,
                    last_x=float(progress),
                )
            )

    return hikers


def step_hikers(
    hikers: List[Hiker],
    step_index: int,
    rng: np.random.Generator,
) -> None:
    for idx, hiker in enumerate(hikers):
        base_speed = GROUP_SPEEDS.get(hiker.group, 0.11) * hiker.pace
        noise = max(rng.normal(loc=0.0, scale=0.01), -0.02)
        delta = max(base_speed + noise, 0.0)

        if hiker.identifier == SPECIAL_RELAY_ID:
            if 10 <= step_index < 25:
                delta += 0.2
            elif step_index >= 25:
                delta += 0.1

        hiker.progress = max(hiker.progress + delta, hiker.last_x)
        centerline_y = trail_y(hiker.progress)

        pack_tightness = 0.35 if hiker.group == "Green" else 1.0
        lateral_wave = 0.08 * np.sin(0.22 * step_index + idx * 0.5)
        lateral_position = pack_tightness * hiker.lane_offset + lateral_wave

        hiker.position = np.array(
            [hiker.progress, centerline_y + lateral_position],
            dtype=float,
        )
        hiker.last_x = hiker.progress

