"""CLI entrypoint for running the organized hiker mesh scenarios."""
from __future__ import annotations

from sim.alerts import AlertManager
from sim.constants import TRAIL_BOUNDS
from sim.mesh import run_simulation
from sim.scenarios import SCENARIOS
from sim.visuals import build_palette, render_animation


def main() -> None:
    lon_bounds, lat_bounds = TRAIL_BOUNDS

    for scenario in SCENARIOS:
        print(f"Running: {scenario.name}")
        alert_manager = AlertManager(scenario.log_path)
        snapshots, alerts = run_simulation(
            alert_manager,
            groups_filter=scenario.group_filter,
            custom_steps=scenario.custom_steps,
        )
        palette = build_palette(
            sorted(
                {node["group"] for frame in snapshots for node in frame["nodes"].values()}
            )
        )
        render_animation(
            snapshots,
            palette,
            (lon_bounds, lat_bounds),
            scenario.gif_path,
        )
        alert_manager.flush()
        if alerts:
            for alert in alerts:
                print("  •", alert)
        else:
            print("  No alerts recorded.")
        print(f"  GIF saved to {scenario.gif_path.resolve()}")
        print(f"  Alerts written to {scenario.log_path.resolve()}\n")


if __name__ == "__main__":
    main()

