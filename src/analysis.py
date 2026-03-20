"""
Módulo de análisis y visualización de resultados.

Genera:
1. Mapa de cobertura RSRP y SINR (heatmaps).
2. Mapa de emplazamientos con usuarios y celdas asignadas.
3. Gráfica comparativa de throughput (macro-only vs macro+small cells).
4. Histograma de SINR y distribución de carga por celda.
5. Análisis de handover: fracción de usuarios en zona de borde.

Todos los resultados se guardan en el directorio ``results/``.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para entornos de CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.network import BaseStation, UserEquipment, Network
from src.propagation import cost231_hata_path_loss_array


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Mapa de cobertura RSRP
# ---------------------------------------------------------------------------

def _best_server_rsrp(grid_x: np.ndarray, grid_y: np.ndarray,
                       base_stations: List[BaseStation]) -> np.ndarray:
    """Calcula la RSRP del mejor servidor en cada punto de la rejilla."""
    rsrp_grid = np.full(grid_x.shape, -200.0)
    for bs in base_stations:
        d = np.sqrt((grid_x - bs.x) ** 2 + (grid_y - bs.y) ** 2)
        pl = cost231_hata_path_loss_array(
            bs.freq_mhz, bs.height_m, 1.5, d, bs.city_size
        )
        rsrp = bs.tx_power_dbm - pl
        rsrp_grid = np.maximum(rsrp_grid, rsrp)
    return rsrp_grid


def plot_rsrp_map(base_stations: List[BaseStation],
                  area_m: Tuple[float, float] = (1000, 1000),
                  resolution: int = 200,
                  title: str = "Mapa RSRP",
                  filename: str = "rsrp_map.png") -> str:
    """Genera y guarda el mapa de RSRP del mejor servidor.

    Args:
        base_stations: Lista de BSs a incluir.
        area_m:        Tamaño del área (ancho, alto) en metros.
        resolution:    Número de puntos por eje de la rejilla.
        title:         Título de la figura.
        filename:      Nombre del fichero de salida (en ``results/``).

    Returns:
        Ruta completa del fichero guardado.
    """
    _ensure_results_dir()
    xs = np.linspace(0, area_m[0], resolution)
    ys = np.linspace(0, area_m[1], resolution)
    gx, gy = np.meshgrid(xs, ys)
    rsrp = _best_server_rsrp(gx, gy, base_stations)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        rsrp,
        extent=[0, area_m[0], 0, area_m[1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=-130,
        vmax=-60,
    )
    plt.colorbar(im, ax=ax, label="RSRP (dBm)")

    # Marcar BSs
    for bs in base_stations:
        marker = "^" if bs.cell_type == "macro" else "s"
        color = "blue" if bs.cell_type == "macro" else "red"
        ax.plot(bs.x, bs.y, marker=marker, color=color, markersize=12,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(bs.name, (bs.x, bs.y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="white",
                    fontweight="bold")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    macro_patch = mpatches.Patch(color="blue", label="Macrocelda (▲)")
    sc_patch = mpatches.Patch(color="red", label="Small Cell (■)")
    ax.legend(handles=[macro_patch, sc_patch], loc="upper right")

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Mapa de emplazamientos con usuarios
# ---------------------------------------------------------------------------

_CELL_COLORS = {
    "macro": "blue",
    "small": "red",
}
_BS_MARKERS = {
    "macro": "^",
    "small": "s",
}
_UE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_deployment_map(base_stations: List[BaseStation],
                         users: List[UserEquipment],
                         area_m: Tuple[float, float] = (1000, 1000),
                         title: str = "Mapa de emplazamientos",
                         filename: str = "deployment_map.png") -> str:
    """Genera el mapa con BSs, usuarios y sus asignaciones de celda.

    Args:
        base_stations: Lista de BSs.
        users:         Lista de UEs (con ``serving_bs`` asignado).
        area_m:        Tamaño del área en metros.
        title:         Título.
        filename:      Fichero de salida.

    Returns:
        Ruta del fichero guardado.
    """
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 7))

    bs_index = {bs: i for i, bs in enumerate(base_stations)}

    # Dibujar líneas de asociación UE → BS
    for ue in users:
        if ue.serving_bs is None:
            continue
        idx = bs_index.get(ue.serving_bs, 0)
        color = _UE_COLORS[idx % len(_UE_COLORS)]
        ax.plot([ue.x, ue.serving_bs.x], [ue.y, ue.serving_bs.y],
                color=color, alpha=0.15, linewidth=0.7, zorder=1)

    # Dibujar UEs
    for ue in users:
        idx = bs_index.get(ue.serving_bs, 0) if ue.serving_bs else 0
        color = _UE_COLORS[idx % len(_UE_COLORS)]
        ax.scatter(ue.x, ue.y, s=20, color=color, alpha=0.7, zorder=3)

    # Dibujar BSs
    legend_handles = []
    for bs in base_stations:
        idx = bs_index[bs]
        color = _UE_COLORS[idx % len(_UE_COLORS)]
        m = _BS_MARKERS.get(bs.cell_type, "o")
        ax.plot(bs.x, bs.y, marker=m, color=color, markersize=14,
                markeredgecolor="black", markeredgewidth=1.5, zorder=6,
                linestyle="None")
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{bs.name} ({bs.cell_type})")
        )

    ax.set_xlim(0, area_m[0])
    ax.set_ylim(0, area_m[1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Gráfica comparativa de throughput
# ---------------------------------------------------------------------------

def plot_throughput_comparison(scenarios: List[dict],
                                filename: str = "throughput_comparison.png") -> str:
    """Gráfica comparativa de CDF de throughput para varios escenarios.

    Args:
        scenarios: Lista de dicts con claves:
                   - 'label' (str): nombre del escenario.
                   - 'users' (list[UserEquipment]): UEs con throughput calculado.
        filename:  Fichero de salida.

    Returns:
        Ruta del fichero guardado.
    """
    _ensure_results_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, sc in enumerate(scenarios):
        tputs_mbps = [ue.throughput_bps / 1e6 for ue in sc["users"]]
        color = colors[i % len(colors)]

        # CDF
        sorted_t = np.sort(tputs_mbps)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax1.plot(sorted_t, cdf, label=sc["label"], color=color, linewidth=2)

        # Box plot data
        ax2.boxplot(tputs_mbps, positions=[i + 1], widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color="black", linewidth=2))

    ax1.set_xlabel("Throughput (Mbps)")
    ax1.set_ylabel("CDF")
    ax1.set_title("CDF de throughput por usuario")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(range(1, len(scenarios) + 1))
    ax2.set_xticklabels([sc["label"] for sc in scenarios], rotation=15)
    ax2.set_ylabel("Throughput (Mbps)")
    ax2.set_title("Distribución de throughput (box plot)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Comparativa: Macro solo vs Macro + Small Cells", fontsize=13)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Histograma de SINR y carga por celda
# ---------------------------------------------------------------------------

def plot_sinr_histogram(scenarios: List[dict],
                         filename: str = "sinr_histogram.png") -> str:
    """Histograma de SINR para cada escenario.

    Args:
        scenarios: Lista de dicts con 'label' y 'users'.
        filename:  Fichero de salida.

    Returns:
        Ruta del fichero guardado.
    """
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, sc in enumerate(scenarios):
        sinrs = [ue.sinr_db for ue in sc["users"]]
        ax.hist(sinrs, bins=30, alpha=0.6, label=sc["label"],
                color=colors[i % len(colors)], edgecolor="white")

    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Número de usuarios")
    ax.set_title("Distribución de SINR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_load_distribution(network: Network,
                            title: str = "Distribución de carga por celda",
                            filename: str = "load_distribution.png") -> str:
    """Gráfico de barras de carga (número de usuarios) por BS.

    Args:
        network:  Red tras ejecutar ``run()``.
        title:    Título de la figura.
        filename: Fichero de salida.

    Returns:
        Ruta del fichero guardado.
    """
    _ensure_results_dir()
    counts = network.users_per_bs()
    names = list(counts.keys())
    values = [counts[n] for n in names]
    colors = ["#1f77b4" if "Macro" in n else "#ff7f0e" for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Celda")
    ax.set_ylabel("Usuarios asignados")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Análisis de handover
# ---------------------------------------------------------------------------

def handover_analysis(network_macro: Network,
                       network_hetnet: Network,
                       rsrp_threshold_dbm: float = -100.0) -> dict:
    """Analiza el impacto del handover al añadir small cells.

    Calcula:
    - Número y porcentaje de usuarios en zona de borde (RSRP < umbral) en
      cada escenario.
    - Usuarios que cambiarían de celda al pasar al escenario HetNet (handover).
    - Reducción de carga en la macro.

    Args:
        network_macro:   Red macro-only tras ``run()``.
        network_hetnet:  Red HetNet (macro + small cells) tras ``run()``.
        rsrp_threshold_dbm: Umbral de RSRP para considerar usuario de borde.

    Returns:
        Diccionario con métricas de handover.
    """
    edge_macro = len(network_macro.edge_users(rsrp_threshold_dbm))
    edge_hetnet = len(network_hetnet.edge_users(rsrp_threshold_dbm))
    n_users = len(network_macro.users)

    # Usuarios que cambian de BS (HO al activar small cells)
    handover_count = 0
    for ue_m, ue_h in zip(network_macro.users, network_hetnet.users):
        if ue_m.serving_bs is not None and ue_h.serving_bs is not None:
            if ue_m.serving_bs.name != ue_h.serving_bs.name:
                handover_count += 1

    # Reducción de carga en la macro
    macro_load_before = sum(
        1 for ue in network_macro.users
        if ue.serving_bs is not None and ue.serving_bs.cell_type == "macro"
    )
    macro_load_after = sum(
        1 for ue in network_hetnet.users
        if ue.serving_bs is not None and ue.serving_bs.cell_type == "macro"
    )

    return {
        "n_users": n_users,
        "edge_users_macro_only": edge_macro,
        "edge_users_macro_only_pct": 100 * edge_macro / max(n_users, 1),
        "edge_users_hetnet": edge_hetnet,
        "edge_users_hetnet_pct": 100 * edge_hetnet / max(n_users, 1),
        "handover_triggered": handover_count,
        "handover_pct": 100 * handover_count / max(n_users, 1),
        "macro_load_before": macro_load_before,
        "macro_load_after": macro_load_after,
        "macro_offload_pct": 100 * (macro_load_before - macro_load_after)
                             / max(macro_load_before, 1),
    }
