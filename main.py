"""
Simulación principal: Macrocelda + Small Cells en zona de alta demanda.

Escenario: Calle comercial / festival / estadio (1 km × 1 km).
  - Macrocelda LTE: 1800 MHz, 45 dBm, 25 m, 20 MHz
  - Small Cell 1 y 2: 2600 MHz, 27 dBm, 5 m, 20 MHz

Tareas:
  1. Escenario solo-macro y HetNet (macro + 2 small cells).
  2. Mapa de cobertura RSRP y emplazamientos.
  3. Cálculo de SINR y throughput (Round Robin + Proportional Fair).
  4. Análisis de handover y usuarios de borde.
  5. Gráfica comparativa de throughput y distribución de SINR.

Uso::

    python main.py                    # 50 usuarios, seed 42
    python main.py --users 100        # 100 usuarios
    python main.py --users 80 --cre 6 # con sesgo CRE de 6 dB
    python main.py --help
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

# Añadir la raíz del proyecto al path cuando se ejecuta directamente
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.network import MacroCell, SmallCell, UserEquipment, Network
from src.scheduler import ProportionalFairScheduler
from src.analysis import (
    plot_rsrp_map,
    plot_deployment_map,
    plot_throughput_comparison,
    plot_sinr_histogram,
    plot_load_distribution,
    handover_analysis,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Configuración del escenario
# ---------------------------------------------------------------------------

AREA_M = (1000, 1000)  # 1 km × 1 km

# Posición de la macrocelda: centro del área
MACRO_X, MACRO_Y = 500, 500

# Posiciones de las small cells:
#   SC1 → zona noreste congestionada (festival/calle comercial)
#   SC2 → zona sureste con alta densidad de usuarios
SC1_X, SC1_Y = 700, 650
SC2_X, SC2_Y = 350, 300


def build_base_stations(include_small_cells: bool = True):
    """Crea las estaciones base del escenario."""
    macro = MacroCell(
        x=MACRO_X, y=MACRO_Y,
        tx_power_dbm=45.0,
        height_m=25.0,
        freq_mhz=1800.0,
        bandwidth_mhz=20.0,
        name="Macro-1800",
    )
    if not include_small_cells:
        return [macro]

    sc1 = SmallCell(
        x=SC1_X, y=SC1_Y,
        tx_power_dbm=27.0,
        height_m=5.0,
        freq_mhz=2600.0,
        bandwidth_mhz=20.0,
        name="SC1-2600",
    )
    sc2 = SmallCell(
        x=SC2_X, y=SC2_Y,
        tx_power_dbm=27.0,
        height_m=5.0,
        freq_mhz=2600.0,
        bandwidth_mhz=20.0,
        name="SC2-2600",
    )
    return [macro, sc1, sc2]


def generate_users(n_users: int, seed: int = 42,
                   hotspot: bool = True) -> list:
    """Genera usuarios distribuidos en el área de simulación.

    Con ``hotspot=True`` concentra el 60 % de los usuarios en dos zonas
    de alta demanda (cerca de las small cells) para simular el festival o
    la calle comercial.

    Args:
        n_users: Número total de usuarios.
        seed:    Semilla del generador aleatorio.
        hotspot: Si True, añade concentración de usuarios en hotspots.

    Returns:
        Lista de objetos :class:`~src.network.UserEquipment`.
    """
    rng = np.random.default_rng(seed)
    users = []
    uid = 0

    if hotspot:
        # 40 % uniformes, 30 % en hotspot 1, 30 % en hotspot 2
        n_uniform = int(n_users * 0.40)
        n_hot1 = int(n_users * 0.30)
        n_hot2 = n_users - n_uniform - n_hot1

        # Uniformes
        xs = rng.uniform(0, AREA_M[0], n_uniform)
        ys = rng.uniform(0, AREA_M[1], n_uniform)
        for x, y in zip(xs, ys):
            users.append(UserEquipment(x=float(x), y=float(y), uid=uid))
            uid += 1

        # Hotspot 1 (zona noreste, alrededor de SC1)
        xs = rng.normal(SC1_X, 80, n_hot1)
        ys = rng.normal(SC1_Y, 80, n_hot1)
        xs = np.clip(xs, 0, AREA_M[0])
        ys = np.clip(ys, 0, AREA_M[1])
        for x, y in zip(xs, ys):
            users.append(UserEquipment(x=float(x), y=float(y), uid=uid))
            uid += 1

        # Hotspot 2 (zona sureste, alrededor de SC2)
        xs = rng.normal(SC2_X, 80, n_hot2)
        ys = rng.normal(SC2_Y, 80, n_hot2)
        xs = np.clip(xs, 0, AREA_M[0])
        ys = np.clip(ys, 0, AREA_M[1])
        for x, y in zip(xs, ys):
            users.append(UserEquipment(x=float(x), y=float(y), uid=uid))
            uid += 1
    else:
        xs = rng.uniform(0, AREA_M[0], n_users)
        ys = rng.uniform(0, AREA_M[1], n_users)
        for x, y in zip(xs, ys):
            users.append(UserEquipment(x=float(x), y=float(y), uid=uid))
            uid += 1

    return users


def _copy_users(users):
    """Devuelve una copia profunda de la lista de usuarios."""
    return [copy.deepcopy(ue) for ue in users]


# ---------------------------------------------------------------------------
# Ejecución de los dos escenarios
# ---------------------------------------------------------------------------

def run_scenario_rr(base_stations, users, cre_offset_db=0.0):
    """Ejecuta el escenario con scheduler Round Robin (igual que Network.run)."""
    net = Network(base_stations, users, cre_offset_db=cre_offset_db)
    net.run()
    return net


def run_scenario_pf(base_stations, users, cre_offset_db=0.0,
                    pf_steps: int = 200):
    """Ejecuta el escenario con scheduler Proportional Fair."""
    net = Network(base_stations, users, cre_offset_db=cre_offset_db)
    net.assign_cells()
    sched = ProportionalFairScheduler(net, alpha=0.9)
    sched.run(n_steps=pf_steps)
    return net


# ---------------------------------------------------------------------------
# Informe de texto
# ---------------------------------------------------------------------------

def print_report(label: str, net: Network, ho_stats: dict | None = None) -> None:
    """Imprime un resumen de métricas del escenario."""
    print(f"\n{'=' * 60}")
    print(f"  Escenario: {label}")
    print(f"{'=' * 60}")
    print(f"  Usuarios totales       : {len(net.users)}")
    print(f"  Throughput medio       : {net.mean_throughput_bps() / 1e6:.2f} Mbps")
    print(f"  SINR media             : {net.mean_sinr_db():.1f} dB")
    edge = net.edge_users()
    print(f"  Usuarios de borde      : {len(edge)} "
          f"({100 * len(edge) / max(len(net.users), 1):.1f} %)")
    print(f"  Distribución de carga:")
    for bs_name, count in net.users_per_bs().items():
        print(f"    {bs_name:20s}: {count} usuarios")

    if ho_stats:
        print(f"\n  --- Análisis de Handover ---")
        print(f"  Handovers al activar HetNet  : {ho_stats['handover_triggered']} "
              f"({ho_stats['handover_pct']:.1f} %)")
        print(f"  Descarga de tráfico macro    : {ho_stats['macro_offload_pct']:.1f} %")
        print(f"  Usuarios borde macro-only    : {ho_stats['edge_users_macro_only']} "
              f"({ho_stats['edge_users_macro_only_pct']:.1f} %)")
        print(f"  Usuarios borde HetNet        : {ho_stats['edge_users_hetnet']} "
              f"({ho_stats['edge_users_hetnet_pct']:.1f} %)")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulación macro + small cells en zona de alta demanda"
    )
    parser.add_argument("--users", type=int, default=50,
                        help="Número de usuarios activos (30-100, default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--cre", type=float, default=0.0,
                        help="Offset CRE para small cells en dB (default: 0)")
    parser.add_argument("--pf-steps", type=int, default=200,
                        help="Iteraciones del scheduler PF (default: 200)")
    parser.add_argument("--no-hotspot", action="store_true",
                        help="Distribución uniforme de usuarios (sin hotspots)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Simulación: {args.users} usuarios, seed={args.seed}, "
          f"CRE={args.cre} dB")

    # -----------------------------------------------------------------------
    # 1. Generar usuarios
    # -----------------------------------------------------------------------
    users_base = generate_users(
        args.users, seed=args.seed, hotspot=not args.no_hotspot
    )

    # -----------------------------------------------------------------------
    # 2. Escenario A: Solo macrocelda (Round Robin)
    # -----------------------------------------------------------------------
    bs_macro = build_base_stations(include_small_cells=False)
    users_macro = _copy_users(users_base)
    net_macro = run_scenario_rr(bs_macro, users_macro)

    # -----------------------------------------------------------------------
    # 3. Escenario B: Macro + 2 Small Cells (Round Robin)
    # -----------------------------------------------------------------------
    bs_hetnet = build_base_stations(include_small_cells=True)
    users_hetnet = _copy_users(users_base)
    net_hetnet = run_scenario_rr(
        bs_hetnet, users_hetnet, cre_offset_db=args.cre
    )

    # -----------------------------------------------------------------------
    # 4. Escenario C: Macro + 2 Small Cells (Proportional Fair)
    # -----------------------------------------------------------------------
    users_hetnet_pf = _copy_users(users_base)
    net_hetnet_pf = run_scenario_pf(
        build_base_stations(include_small_cells=True),
        users_hetnet_pf,
        cre_offset_db=args.cre,
        pf_steps=args.pf_steps,
    )

    # -----------------------------------------------------------------------
    # 5. Análisis de handover
    # -----------------------------------------------------------------------
    ho_stats = handover_analysis(net_macro, net_hetnet)

    # -----------------------------------------------------------------------
    # 6. Informes de texto
    # -----------------------------------------------------------------------
    print_report("Solo Macro (RR)", net_macro)
    print_report("Macro + Small Cells (RR)", net_hetnet, ho_stats)
    print_report("Macro + Small Cells (PF)", net_hetnet_pf)

    # -----------------------------------------------------------------------
    # 7. Visualizaciones
    # -----------------------------------------------------------------------

    # 7a. Mapas RSRP
    p1 = plot_rsrp_map(
        bs_macro, area_m=AREA_M,
        title="RSRP – Solo Macrocelda",
        filename="rsrp_macro_only.png",
    )
    p2 = plot_rsrp_map(
        bs_hetnet, area_m=AREA_M,
        title="RSRP – Macro + Small Cells",
        filename="rsrp_hetnet.png",
    )

    # 7b. Mapas de emplazamientos
    p3 = plot_deployment_map(
        bs_macro, net_macro.users, area_m=AREA_M,
        title="Emplazamientos – Solo Macrocelda",
        filename="deployment_macro_only.png",
    )
    p4 = plot_deployment_map(
        bs_hetnet, net_hetnet.users, area_m=AREA_M,
        title="Emplazamientos – Macro + Small Cells",
        filename="deployment_hetnet.png",
    )

    # 7c. Comparativa de throughput
    scenarios = [
        {"label": "Solo Macro (RR)", "users": net_macro.users},
        {"label": "HetNet RR", "users": net_hetnet.users},
        {"label": "HetNet PF", "users": net_hetnet_pf.users},
    ]
    p5 = plot_throughput_comparison(
        scenarios, filename="throughput_comparison.png"
    )

    # 7d. Histograma SINR
    p6 = plot_sinr_histogram(scenarios, filename="sinr_histogram.png")

    # 7e. Distribución de carga HetNet
    p7 = plot_load_distribution(
        net_hetnet,
        title="Distribución de carga – Macro + Small Cells (RR)",
        filename="load_distribution.png",
    )

    print(f"\nResultados guardados en: {RESULTS_DIR}/")
    for p in [p1, p2, p3, p4, p5, p6, p7]:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
