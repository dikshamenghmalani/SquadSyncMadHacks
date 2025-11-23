"""Simulate multi-group hikers that maintain an ad hoc mesh network."""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib

# Use a non-interactive backend so the script works headlessly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

from alerts import AlertManager


# --- Simulation constants -------------------------------------------------
GROUP_LAYOUT = {
    "Blue": np.array([3.5, -1.6]),
    "Yellow": np.array([11.0, 1.4]),
    "Red": np.array([18.5, -1.0]),
}
GROUP_NAMES = list(GROUP_LAYOUT.keys())
HIKERS_PER_GROUP = 4
COMM_RADIUS = 4.2
SIMULATION_STEPS = 60
RNG_SEED = 21
TRAIL_BOUNDS = ((0.0, 26.0), (-6.0, 6.0))  # (x_min, x_max), (y_min, y_max)
OUTPUT_GIF = Path("hiker_mesh.gif")
SPECIAL_RELAY_ID = "Blue-H2"
PERMANENT_RELAY_IDS = {SPECIAL_RELAY_ID}
GROUP_SPEEDS = {
    "Blue": 0.12,
    "Yellow": 0.1,
    "Red": 0.14,
}
TRAIL_PROFILE = [
    (0.0, -0.3),
    (6.0, 0.1),
    (13.0, -0.2),
    (19.0, 0.35),
    (26.0, 0.0),
]


def trail_y(distance: float) -> float:
    """Return the trail centerline's y-coordinate for a given x distance."""
    if distance <= TRAIL_PROFILE[0][0]:
        return TRAIL_PROFILE[0][1]
    for (x0, y0), (x1, y1) in zip(TRAIL_PROFILE, TRAIL_PROFILE[1:]):
        if distance <= x1:
            ratio = (distance - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return TRAIL_PROFILE[-1][1]


@dataclass
class Hiker:
    """Single hiker with state used for smooth motion."""

    identifier: str
    group: str
    position: np.ndarray
    home: np.ndarray
    pace: float
    progress: float
    lane_offset: float
    last_x: float


def initialize_hikers(
    hikers_per_group: int, rng: np.random.Generator
) -> List[Hiker]:
    """Place each group near a predefined trail centroid with small jitter."""
    hikers: List[Hiker] = []
    for group_name, center in GROUP_LAYOUT.items():
        for member_index in range(hikers_per_group):
            progress = float(center[0] + rng.normal(loc=0.0, scale=0.4))
            progress = max(progress, 0.0)
            centerline_y = trail_y(progress)
            base_lane = float(center[1] - trail_y(center[0]))
            lane_offset = float(
                np.clip(
                    base_lane + rng.normal(loc=0.0, scale=0.15),
                    -0.9,
                    0.9,
                )
            )
            base_position = np.array(
                [progress, centerline_y + lane_offset],
                dtype=float,
            )
            identifier = f"{group_name}-H{member_index + 1}"
            hikers.append(
                Hiker(
                    identifier=identifier,
                    group=group_name,
                    position=base_position.copy(),
                    home=base_position.copy(),
                    pace=float(np.clip(rng.normal(1.0, 0.18), 0.65, 1.4)),
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
    """Advance hikers strictly forward along the shared trail centerline."""
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

        pack_tightness = 0.35 if hiker.group == "Yellow" else 1.0
        lateral_wave = 0.08 * np.sin(0.22 * step_index + idx * 0.5)
        lateral_position = pack_tightness * hiker.lane_offset + lateral_wave

        hiker.position = np.array(
            [hiker.progress, centerline_y + lateral_position],
            dtype=float,
        )
        hiker.last_x = hiker.progress


def build_mesh(
    hikers: List[Hiker], communication_radius: float
) -> nx.Graph:
    """Create an undirected graph connecting hikers within range."""
    graph = nx.Graph()
    for hiker in hikers:
        graph.add_node(
            hiker.identifier,
            group=hiker.group,
            position=tuple(hiker.position),
        )

    for left, right in itertools.combinations(hikers, 2):
        distance = np.linalg.norm(left.position - right.position)
        if distance <= communication_radius:
            graph.add_edge(left.identifier, right.identifier, weight=distance)

    return graph


def evaluate_statuses(
    hikers: List[Hiker], mesh: nx.Graph
) -> Tuple[
    Dict[str, Dict[str, float | bool | List[str] | Tuple[float, float]]],
    Dict[str, List[str]],
]:
    """Flag hikers that detached from their group or became isolated."""
    group_sizes = Counter(hiker.group for hiker in hikers)
    group_members = {
        group: sorted(h.identifier for h in hikers if h.group == group)
        for group in group_sizes
    }

    components = list(nx.connected_components(mesh))
    component_sizes: Dict[str, int] = {}
    node_to_component: Dict[str, set[str]] = {}
    component_group_map: Dict[str, Set[str]] = {}
    for component in components:
        size = len(component)
        groups_in_component = {mesh.nodes[node]["group"] for node in component}
        for node in component:
            component_sizes[node] = size
            node_to_component[node] = component
            component_group_map[node] = groups_in_component

    statuses: Dict[
        str, Dict[str, float | bool | List[str] | Tuple[float, float]]
    ] = {}
    for hiker in hikers:
        neighbors = list(mesh.neighbors(hiker.identifier))
        same_group_neighbors = [
            neighbor
            for neighbor in neighbors
            if mesh.nodes[neighbor]["group"] == hiker.group
        ]

        component_members = node_to_component.get(
            hiker.identifier, {hiker.identifier}
        )
        reachable_group_members = sorted(
            component_members.intersection(set(group_members[hiker.group]))
            - {hiker.identifier}
        )
        group_has_company = group_sizes[hiker.group] > 1

        relay_groups = sorted(
            component_group_map.get(hiker.identifier, set()) - {hiker.group}
        )

        statuses[hiker.identifier] = {
            "group": hiker.group,
            "position": tuple(hiker.position),
            "has_direct_group_link": bool(same_group_neighbors),
            "left_group": group_has_company and not same_group_neighbors,
            "reachable_group_members": reachable_group_members,
            "disconnected": group_has_company and not reachable_group_members,
            "isolated": len(neighbors) == 0,
            "component_size": component_sizes.get(hiker.identifier, 1),
            "relay_groups": relay_groups,
        }

    for hiker in hikers:
        if hiker.identifier in PERMANENT_RELAY_IDS:
            if statuses[hiker.identifier]["left_group"]:
                statuses[hiker.identifier]["disconnected"] = True

    return statuses, group_members


def monitor_alerts(
    step_index: int,
    statuses: Dict[str, Dict[str, float | bool | List[str] | Tuple[float, float]]],
    tracker: Dict[str, Dict[str, bool | None]],
    mesh: nx.Graph,
    group_members: Dict[str, List[str]],
    alert_manager: AlertManager,
) -> List[str]:
    """Emit alerts when hikers detach from their group or become isolated."""
    alerts: List[str] = []

    def shortest_path_to_member(
        source: str, targets: List[str]
    ) -> List[str] | None:
        candidate_paths: List[List[str]] = []
        for member in targets:
            try:
                candidate_paths.append(nx.shortest_path(mesh, source, member))
            except nx.NetworkXNoPath:
                continue
        if not candidate_paths:
            return None
        return min(candidate_paths, key=len)

    for hiker_id, status in statuses.items():
        previous = tracker[hiker_id]

        current_left_group = status["left_group"]
        current_disconnected = status["disconnected"]
        current_isolated = status["isolated"]
        relay_groups = status["relay_groups"]
        current_relay_support = current_disconnected and bool(relay_groups)

        # Initialize the tracker on the first observation.
        if previous["left_group"] is None:
            previous["left_group"] = current_left_group
        if previous["disconnected"] is None:
            previous["disconnected"] = current_disconnected
        if previous["isolated"] is None:
            previous["isolated"] = current_isolated
        if previous["relay_support"] is None:
            previous["relay_support"] = current_relay_support

        if (
            previous["left_group"] is False
            and current_left_group
            and not current_disconnected
        ):
            reachable_members = status["reachable_group_members"]
            path = shortest_path_to_member(hiker_id, reachable_members)

            if path:
                message = (
                    f"{hiker_id} left {status['group']} bubble but is relayed via "
                    f"{' -> '.join(path)}"
                )
            else:
                message = (
                    f"{hiker_id} left {status['group']} bubble with no relay route."
                )
            alerts.append(f"[t={step_index:02d}] {message}")
            alert_manager.record(
                step=step_index,
                audience="network",
                category="left_group",
                actor=hiker_id,
                message=message,
            )

        if previous["disconnected"] is False and current_disconnected:
            x, y = status["position"]
            message = (
                f"{hiker_id} disconnected from {status['group']} at ({x:.2f}, {y:.2f})."
            )
            alerts.append(f"[t={step_index:02d}] {message}")
            alert_manager.record(
                step=step_index,
                audience=status["group"],
                category="disconnected",
                actor=hiker_id,
                message=message,
            )

            for teammate in group_members.get(status["group"], []):
                if teammate == hiker_id:
                    continue
                teammate_msg = (
                    f"{teammate} notified that teammate {hiker_id} disconnected "
                    f"at ({x:.2f}, {y:.2f})."
                )
                alerts.append(f"[t={step_index:02d}] {teammate_msg}")
                alert_manager.record(
                    step=step_index,
                    audience=teammate,
                    category="group_notification",
                    actor=hiker_id,
                    message=teammate_msg,
                )

        if previous["isolated"] is False and current_isolated:
            message = f"{hiker_id} is fully isolated with no communication path."
            alerts.append(f"[t={step_index:02d}] {message}")
            alert_manager.record(
                step=step_index,
                audience="network",
                category="isolated",
                actor=hiker_id,
                message=message,
            )

        if previous["relay_support"] is False and current_relay_support:
            x, y = status["position"]
            host_str = ", ".join(relay_groups)
            message = (
                f"{hiker_id} is disconnected from {status['group']} but "
                f"now relayed by {host_str} near ({x:.2f}, {y:.2f})."
            )
            alerts.append(f"[t={step_index:02d}] {message}")
            alert_manager.record(
                step=step_index,
                audience="network",
                category="relay_support",
                actor=hiker_id,
                message=message,
            )

            for teammate in group_members.get(status["group"], []):
                if teammate == hiker_id:
                    continue
                teammate_msg = (
                    f"{teammate} updated: {hiker_id} now with {host_str} "
                    f"near ({x:.2f}, {y:.2f})."
                )
                alerts.append(f"[t={step_index:02d}] {teammate_msg}")
                alert_manager.record(
                    step=step_index,
                    audience=teammate,
                    category="group_update",
                    actor=hiker_id,
                    message=teammate_msg,
                )

            for host_group in relay_groups:
                for host_member in group_members.get(host_group, []):
                    host_msg = (
                        f"{host_member} relaying {hiker_id} for {status['group']} "
                        f"near ({x:.2f}, {y:.2f})."
                    )
                    alerts.append(f"[t={step_index:02d}] {host_msg}")
                    alert_manager.record(
                        step=step_index,
                        audience=host_member,
                        category="host_update",
                        actor=hiker_id,
                        message=host_msg,
                    )

        if (
            previous["disconnected"] is True
            and not current_disconnected
            and hiker_id not in PERMANENT_RELAY_IDS
        ):
            x, y = status["position"]
            reachable_members = status["reachable_group_members"]
            path = shortest_path_to_member(hiker_id, reachable_members) or [hiker_id]
            message = (
                f"{hiker_id} reconnected with {status['group']} via "
                f"{' -> '.join(path)} near ({x:.2f}, {y:.2f})."
            )
            alerts.append(f"[t={step_index:02d}] {message}")
            alert_manager.record(
                step=step_index,
                audience=status["group"],
                category="reconnected",
                actor=hiker_id,
                message=message,
            )

            for teammate in group_members.get(status["group"], []):
                if teammate == hiker_id:
                    continue
                teammate_msg = (
                    f"{teammate} updated: {hiker_id} now reachable near ({x:.2f}, {y:.2f})."
                )
                alerts.append(f"[t={step_index:02d}] {teammate_msg}")
                alert_manager.record(
                    step=step_index,
                    audience=teammate,
                    category="group_update",
                    actor=hiker_id,
                    message=teammate_msg,
                )

        previous["left_group"] = current_left_group
        previous["disconnected"] = current_disconnected
        previous["isolated"] = current_isolated
        previous["relay_support"] = current_relay_support

    return alerts


def capture_snapshot(
    step_index: int,
    hikers: List[Hiker],
    statuses: Dict[str, Dict[str, float | bool | List[str] | Tuple[float, float]]],
    mesh: nx.Graph,
) -> Dict:
    """Store the data needed to render a frame of the animation."""
    nodes_payload = {}
    for hiker in hikers:
        nodes_payload[hiker.identifier] = {
            "pos": tuple(hiker.position),
            "group": hiker.group,
            "left_group": statuses[hiker.identifier]["left_group"],
            "isolated": statuses[hiker.identifier]["isolated"],
            "disconnected": statuses[hiker.identifier]["disconnected"],
        }

    snapshot = {
        "step": step_index,
        "nodes": nodes_payload,
        "edges": sorted(mesh.edges()),
    }
    return snapshot


def run_simulation(alert_manager: AlertManager) -> Tuple[List[Dict], List[str]]:
    """Run the full simulation and record per-step snapshots plus alerts."""
    rng = np.random.default_rng(RNG_SEED)
    hikers = initialize_hikers(HIKERS_PER_GROUP, rng)
    tracker = {
        hiker.identifier: {
            "left_group": None,
            "disconnected": None,
            "isolated": None,
            "relay_support": None,
        }
        for hiker in hikers
    }

    snapshots: List[Dict] = []
    alert_log: List[str] = []

    for step_index in range(SIMULATION_STEPS):
        step_hikers(hikers, step_index, rng)
        mesh = build_mesh(hikers, COMM_RADIUS)
        statuses, group_members = evaluate_statuses(hikers, mesh)
        alert_log.extend(
            monitor_alerts(
                step_index,
                statuses,
                tracker,
                mesh,
                group_members,
                alert_manager,
            )
        )
        snapshots.append(capture_snapshot(step_index, hikers, statuses, mesh))

    return snapshots, alert_log


def build_palette(groups: List[str]) -> Dict[str, str]:
    """Assign a unique color to each group."""
    cmap = plt.get_cmap("tab10")
    palette = {}
    for idx, group in enumerate(sorted(groups)):
        palette[group] = cmap(idx % cmap.N)
    return palette


def render_animation(
    snapshots: List[Dict],
    palette: Dict[str, str],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    output_path: Path,
) -> None:
    """Render a NetworkX animation that highlights risky hikers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    def draw_frame(frame_index: int) -> None:
        ax.clear()
        snapshot = snapshots[frame_index]
        nodes = snapshot["nodes"]
        mesh = nx.Graph()
        mesh.add_nodes_from(nodes.keys())
        mesh.add_edges_from(snapshot["edges"])

        positions = {node_id: nodes[node_id]["pos"] for node_id in mesh.nodes}
        node_order = list(mesh.nodes)
        node_colors = [palette[nodes[node]["group"]] for node in node_order]
        node_sizes = []
        border_colors = []

        for node in node_order:
            if nodes[node]["isolated"]:
                node_sizes.append(460)
                border_colors.append("#b30000")
            elif nodes[node]["left_group"]:
                node_sizes.append(380)
                border_colors.append("#ff8c00")
            else:
                node_sizes.append(320)
                border_colors.append("#1f2933")

        nx.draw_networkx_edges(
            mesh,
            positions,
            width=1.6,
            alpha=0.75,
            edge_color="#94a3b8",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            mesh,
            positions,
            node_color=node_colors,
            edgecolors=border_colors,
            linewidths=2.0,
            node_size=node_sizes,
            ax=ax,
        )
        nx.draw_networkx_labels(
            mesh,
            positions,
            font_size=7,
            font_weight="bold",
            ax=ax,
        )

        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_xlabel("Trail longitude (km)")
        ax.set_ylabel("Trail latitude (km)")
        ax.set_title(
            "Mesh-connected hikers\n"
            "Orange outline = left group, Red outline = isolated"
        )
        ax.text(
            0.02,
            0.96,
            f"Time step {snapshot['step']:02d}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        legend_entries = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=group,
                markerfacecolor=palette[group],
                markeredgecolor="#1f2933",
                markersize=10,
            )
            for group in palette
        ]
        legend_entries.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Left group bubble",
                    markerfacecolor="#d9d9d9",
                    markeredgecolor="#ff8c00",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Isolated",
                    markerfacecolor="#d9d9d9",
                    markeredgecolor="#b30000",
                    markersize=10,
                ),
            ]
        )
        ax.legend(
            handles=legend_entries,
            loc="upper right",
            framealpha=0.9,
            fontsize=8,
        )

    animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(snapshots),
        interval=350,
        blit=False,
        repeat=False,
    ).save(output_path, writer="pillow", fps=4)

    plt.close(fig)


def main() -> None:
    alert_manager = AlertManager()
    snapshots, alert_log = run_simulation(alert_manager)
    groups = sorted({node["group"] for frame in snapshots for node in frame["nodes"].values()})
    palette = build_palette(groups)

    render_animation(snapshots, palette, TRAIL_BOUNDS, OUTPUT_GIF)

    if alert_log:
        print("Alerts:")
        for alert in alert_log:
            print("  •", alert)
    else:
        print("No alerts triggered.")

    print(f"\nAnimation saved to {OUTPUT_GIF.resolve()}")
    alert_manager.flush()
    print(f"Alert log saved to {alert_manager.output_path.resolve()}")


if __name__ == "__main__":
    main()

