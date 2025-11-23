from __future__ import annotations

from pathlib import Path

import numpy as np

GROUP_LAYOUT = {
    "Blue": np.array([3.5, -1.6]),
    "Orange": np.array([11.0, 1.4]),
    "Green": np.array([18.5, -1.0]),
}
GROUP_NAMES = list(GROUP_LAYOUT.keys())

HIKERS_PER_GROUP = 4
COMM_RADIUS = 4.2
INTERGROUP_RADIUS = 1
SIMULATION_STEPS = 60
RNG_SEED = 21

SPECIAL_RELAY_ID = "Blue-H2"
PERMANENT_RELAY_IDS = {SPECIAL_RELAY_ID}

GROUP_SPEEDS = {
    "Blue": 0.12,
    "Orange": 0.1,
    "Green": 0.14,
}
GROUP_SEPARATION_THRESHOLD = 2.5
SEPARATION_OVERRIDES = {
    SPECIAL_RELAY_ID: 0.03,  # make Blue-H2 break away almost immediately
}

MAP_IMAGE_PATH = Path("assets/yosemite_map.png")
MAP_ATTRIBUTION = "Basemap: user-provided Yosemite map"

TRAIL_PROFILE = [
    (0.0, -0.3),
    (6.0, 0.1),
    (13.0, -0.2),
    (19.0, 0.35),
    (26.0, 0.0),
]

TRAIL_BOUNDS = ((-119.58, -119.54), (37.725, 37.745))

OUTPUT_DIR = Path("outputs")
GIF_DIR = OUTPUT_DIR / "gifs"
LOG_DIR = OUTPUT_DIR / "logs"

for directory in (GIF_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

