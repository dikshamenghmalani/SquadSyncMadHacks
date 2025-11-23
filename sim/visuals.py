from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import networkx as nx

from .constants import MAP_ATTRIBUTION, MAP_IMAGE_PATH


def build_palette(groups: List[str]) -> Dict[str, str]:
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
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    basemap_img = plt.imread(MAP_IMAGE_PATH) if MAP_IMAGE_PATH.exists() else None

    def draw_frame(frame_index: int) -> None:
        ax.clear()
        snapshot = snapshots[frame_index]
        nodes = snapshot["nodes"]
        mesh = nx.Graph()
        mesh.add_nodes_from(nodes.keys())
        mesh.add_edges_from(snapshot["edges"])

        positions = {node_id: nodes[node_id]["display_pos"] for node_id in mesh.nodes}
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

        lon_bounds, lat_bounds = bounds
        if basemap_img is not None:
            ax.imshow(
                basemap_img,
                extent=[lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
                origin="upper",
                aspect="equal",
                zorder=0,
            )

        nx.draw_networkx_edges(
            mesh,
            positions,
            width=1.6,
            alpha=0.85,
            edge_color="#0f1010",
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

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.margins(0)
        if basemap_img is not None:
            ax.set_axis_off()
        else:
            ax.set_xlabel("Trail longitude (km)")
            ax.set_ylabel("Trail latitude (km)")
        ax.set_title(
            "Mesh-connected hikers\n"
            "Orange outline = left group, Red outline = isolated",
            color="white" if basemap_img is not None else "black",
        )
        ax.text(
            0.02,
            0.96,
            f"Time step {snapshot['step']:02d}",
            transform=ax.transAxes,
            fontsize=10,
            color="white" if basemap_img is not None else "black",
            bbox=dict(facecolor="black" if basemap_img is not None else "white", alpha=0.6, edgecolor="none"),
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
            facecolor="#f9fafb",
        )
        if basemap_img is not None:
            ax.text(
                0.01,
                0.01,
                MAP_ATTRIBUTION,
                transform=ax.transAxes,
                fontsize=6,
                color="#f9fafb",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
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

