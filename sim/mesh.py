from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .alerts import AlertManager
from .constants import (
    COMM_RADIUS,
    INTERGROUP_RADIUS,
    GROUP_SEPARATION_THRESHOLD,
    PERMANENT_RELAY_IDS,
    RNG_SEED,
    SIMULATION_STEPS,
    SEPARATION_OVERRIDES,
)
from .models import Hiker, initialize_hikers, step_hikers
from .trail import to_geo_coords


def _threshold_for(node_id: str) -> float:
    return SEPARATION_OVERRIDES.get(node_id, GROUP_SEPARATION_THRESHOLD)


def build_mesh(
    hikers: List[Hiker],
    communication_radius: float = COMM_RADIUS,
    intergroup_radius: float = INTERGROUP_RADIUS,
) -> nx.Graph:
    graph = nx.Graph()
    threshold_lookup = {hiker.identifier: _threshold_for(hiker.identifier) for hiker in hikers}

    for hiker in hikers:
        graph.add_node(
            hiker.identifier,
            group=hiker.group,
            position=tuple(hiker.position),
        )

    for left, right in itertools.combinations(hikers, 2):
        distance = np.linalg.norm(left.position - right.position)
        if left.group == right.group:
            limit = min(threshold_lookup[left.identifier], threshold_lookup[right.identifier])
            if distance <= limit:
                graph.add_edge(left.identifier, right.identifier, weight=distance)
            continue

        if distance <= intergroup_radius:
            graph.add_edge(left.identifier, right.identifier, weight=distance)

    return graph


def evaluate_statuses(
    hikers: List[Hiker], mesh: nx.Graph
) -> Tuple[
    Dict[str, Dict[str, float | bool | List[str] | Tuple[float, float]]],
    Dict[str, List[str]],
]:
    group_sizes = Counter(hiker.group for hiker in hikers)
    group_members = {
        group: sorted(h.identifier for h in hikers if h.group == group)
        for group in group_sizes
    }

    components = list(nx.connected_components(mesh))
    component_sizes: Dict[str, int] = {}
    node_to_component: Dict[str, set[str]] = {}
    component_group_map: Dict[str, set[str]] = {}
    for component in components:
        size = len(component)
        groups_in_component = {mesh.nodes[node]["group"] for node in component}
        for node in component:
            component_sizes[node] = size
            node_to_component[node] = component
            component_group_map[node] = groups_in_component

    position_lookup = {hiker.identifier: hiker.position for hiker in hikers}
    statuses: Dict[str, Dict[str, float | bool | List[str] | Tuple[float, float]]] = {}
    for hiker in hikers:
        neighbors = list(mesh.neighbors(hiker.identifier))
        same_group_neighbors = [
            neighbor
            for neighbor in neighbors
            if mesh.nodes[neighbor]["group"] == hiker.group
        ]

        min_same_group_distance = None
        for other in hikers:
            if other.group != hiker.group or other.identifier == hiker.identifier:
                continue
            distance = np.linalg.norm(position_lookup[hiker.identifier] - position_lookup[other.identifier])
            if min_same_group_distance is None or distance < min_same_group_distance:
                min_same_group_distance = distance

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

        threshold = _threshold_for(hiker.identifier)

        left_group = group_has_company and (
            min_same_group_distance is None
            or min_same_group_distance > threshold
        )

        statuses[hiker.identifier] = {
            "group": hiker.group,
            "position": tuple(hiker.position),
            "has_direct_group_link": not left_group,
            "left_group": left_group,
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
    nodes_payload = {}
    for hiker in hikers:
        nodes_payload[hiker.identifier] = {
            "pos": tuple(hiker.position),
            "display_pos": to_geo_coords(hiker.progress, hiker.lane_offset),
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


def run_simulation(
    alert_manager: AlertManager,
    groups_filter: List[str] | None = None,
    custom_steps: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[List[Dict], List[str]]:
    rng = np.random.default_rng(RNG_SEED)
    hikers = initialize_hikers(rng, groups_filter, custom_steps)
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

