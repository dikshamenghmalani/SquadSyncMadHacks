from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import TRAIL_BOUNDS, TRAIL_PROFILE


def trail_y(distance: float) -> float:
    if distance <= TRAIL_PROFILE[0][0]:
        return TRAIL_PROFILE[0][1]
    for (x0, y0), (x1, y1) in zip(TRAIL_PROFILE, TRAIL_PROFILE[1:]):
        if distance <= x1:
            ratio = (distance - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return TRAIL_PROFILE[-1][1]


def to_world_position(progress: float, lane_offset: float) -> np.ndarray:
    return np.array([progress, trail_y(progress) + lane_offset], dtype=float)


def to_geo_coords(progress: float, lane_offset: float) -> Tuple[float, float]:
    lon_min, lon_max = TRAIL_BOUNDS[0]
    lat_min, lat_max = TRAIL_BOUNDS[1]
    lon = lon_min + (progress / 26.0) * (lon_max - lon_min)
    normalized_lat = (trail_y(progress) + lane_offset + 6.0) / 12.0
    lat = lat_min + normalized_lat * (lat_max - lat_min)
    return lon, lat

