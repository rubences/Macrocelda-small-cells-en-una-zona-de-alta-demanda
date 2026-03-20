"""
Tests unitarios para el módulo de red (network.py).
Cubre BaseStation, MacroCell, SmallCell, UserEquipment y Network.
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.network import (
    BaseStation,
    MacroCell,
    SmallCell,
    UserEquipment,
    Network,
    UE_HEIGHT_M,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_macro():
    return MacroCell(x=500, y=500, name="Macro")


def make_sc1():
    return SmallCell(x=700, y=650, name="SC1")


def make_sc2():
    return SmallCell(x=300, y=350, name="SC2")


def make_users(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return [
        UserEquipment(
            x=float(rng.uniform(0, 1000)),
            y=float(rng.uniform(0, 1000)),
            uid=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BaseStation
# ---------------------------------------------------------------------------

class TestBaseStation:
    def test_bandwidth_hz(self):
        bs = make_macro()
        assert bs.bandwidth_hz == 20e6

    def test_distance_to(self):
        bs = MacroCell(x=0, y=0)
        assert bs.distance_to(3, 4) == pytest.approx(5.0)

    def test_distance_to_self_is_zero(self):
        bs = make_macro()
        assert bs.distance_to(bs.x, bs.y) == pytest.approx(0.0)

    def test_rsrp_at_decreases_with_distance(self):
        bs = make_macro()
        rsrp_near = bs.rsrp_at(510, 500)   # 10 m
        rsrp_far = bs.rsrp_at(800, 500)    # 300 m
        assert rsrp_near > rsrp_far

    def test_noise_dbm_positive_value(self):
        bs = make_macro()
        noise = bs.noise_dbm()
        # Para 20 MHz de BW debería ser alrededor de -101 dBm
        assert -120 < noise < -80


class TestMacroCell:
    def test_cell_type(self):
        assert make_macro().cell_type == "macro"

    def test_default_freq(self):
        assert make_macro().freq_mhz == 1800.0

    def test_default_power(self):
        assert make_macro().tx_power_dbm == 45.0

    def test_default_height(self):
        assert make_macro().height_m == 25.0


class TestSmallCell:
    def test_cell_type(self):
        assert make_sc1().cell_type == "small"

    def test_default_freq(self):
        assert make_sc1().freq_mhz == 2600.0

    def test_default_power(self):
        assert make_sc1().tx_power_dbm == 27.0

    def test_default_height(self):
        assert make_sc1().height_m == 5.0


# ---------------------------------------------------------------------------
# Network – asignación de celdas
# ---------------------------------------------------------------------------

class TestNetworkCellAssignment:
    def test_all_users_assigned(self):
        bs = [make_macro(), make_sc1()]
        users = make_users(20)
        net = Network(bs, users)
        net.assign_cells()
        assert all(ue.serving_bs is not None for ue in users)

    def test_user_near_sc_assigned_to_sc(self):
        """Un usuario muy cercano a la small cell debe ser asignado a ella."""
        macro = make_macro()
        sc = SmallCell(x=100, y=100, name="SC")
        ue = UserEquipment(x=102, y=102, uid=0)
        net = Network([macro, sc], [ue])
        net.assign_cells()
        assert ue.serving_bs.name == "SC"

    def test_user_near_macro_assigned_to_macro(self):
        """Un usuario muy cercano a la macro debe ser asignado a ella."""
        macro = make_macro()
        sc = SmallCell(x=100, y=100, name="SC")
        ue = UserEquipment(x=502, y=502, uid=0)
        net = Network([macro, sc], [ue])
        net.assign_cells()
        assert ue.serving_bs.name == "Macro"

    def test_cre_offset_shifts_assignment(self):
        """El offset CRE debe atraer usuarios hacia la small cell."""
        macro = MacroCell(x=500, y=500, name="Macro")
        sc = SmallCell(x=600, y=500, name="SC")
        # Usuario más cercano a la macro → sin CRE va a la macro
        ue_no_cre = UserEquipment(x=540, y=500, uid=0)
        ue_cre = UserEquipment(x=540, y=500, uid=1)

        net_no_cre = Network([macro, sc], [ue_no_cre], cre_offset_db=0)
        net_no_cre.assign_cells()
        # Sin CRE el usuario va a la macro (más cercana y más potente)
        assert ue_no_cre.serving_bs.name == "Macro"

        # Con CRE muy agresivo (50 dB) la small cell atrae al usuario
        net_cre = Network([macro, sc], [ue_cre], cre_offset_db=50)
        net_cre.assign_cells()
        assert ue_cre.serving_bs.name == "SC"


# ---------------------------------------------------------------------------
# Network – SINR y throughput
# ---------------------------------------------------------------------------

class TestNetworkSinrThroughput:
    def test_sinr_computed(self):
        bs = [make_macro()]
        users = make_users(5)
        net = Network(bs, users)
        net.assign_cells()
        net.compute_sinr()
        # Todos los usuarios deben tener SINR calculado (no el valor inicial -50)
        for ue in users:
            assert ue.sinr_db > -50

    def test_throughput_positive(self):
        bs = [make_macro()]
        users = make_users(5)
        net = Network(bs, users)
        net.run()
        assert all(ue.throughput_bps > 0 for ue in users)

    def test_more_users_lower_individual_throughput(self):
        """Más usuarios en la misma BS → menor throughput individual."""
        macro = make_macro()
        users_few = make_users(5, seed=1)
        users_many = make_users(50, seed=1)

        net_few = Network([macro], users_few)
        net_few.run()

        net_many = Network([macro], users_many)
        net_many.run()

        mean_few = net_few.mean_throughput_bps()
        mean_many = net_many.mean_throughput_bps()
        assert mean_few > mean_many

    def test_hetnet_higher_throughput_than_macro_only(self):
        """Añadir small cells mejora el throughput: usuarios cerca de las SCs
        se benefician de la descarga de carga de la macro."""
        rng = np.random.default_rng(42)
        # Crear usuarios concentrados cerca de las small cell locations
        # para asegurar que algunos serán asignados a las SCs
        users_positions = []
        # 20 usuarios cerca de SC1 (700,650)
        for _ in range(20):
            x = float(np.clip(rng.normal(700, 30), 0, 1000))
            y = float(np.clip(rng.normal(650, 30), 0, 1000))
            users_positions.append((x, y))
        # 20 usuarios cerca de SC2 (300,350)
        for _ in range(20):
            x = float(np.clip(rng.normal(300, 30), 0, 1000))
            y = float(np.clip(rng.normal(350, 30), 0, 1000))
            users_positions.append((x, y))
        # 10 usuarios uniformes
        for _ in range(10):
            x = float(rng.uniform(0, 1000))
            y = float(rng.uniform(0, 1000))
            users_positions.append((x, y))

        def make_ue_list():
            return [UserEquipment(x=x, y=y, uid=i)
                    for i, (x, y) in enumerate(users_positions)]

        net_macro = Network([make_macro()], make_ue_list())
        net_macro.run()

        sc1 = SmallCell(x=700, y=650, name="SC1",
                        tx_power_dbm=30, freq_mhz=1800)
        sc2 = SmallCell(x=300, y=350, name="SC2",
                        tx_power_dbm=30, freq_mhz=1800)
        net_hetnet = Network([make_macro(), sc1, sc2], make_ue_list())
        net_hetnet.run()

        # HetNet debe mejorar la SINR de los usuarios de borde (macro menos cargada)
        # y dar acceso a los cercanos a SCs con mejor relación señal/ruido
        assert net_hetnet.mean_throughput_bps() > net_macro.mean_throughput_bps()


# ---------------------------------------------------------------------------
# Network – utilidades
# ---------------------------------------------------------------------------

class TestNetworkUtilities:
    def test_users_per_bs(self):
        bs = [make_macro(), make_sc1()]
        users = make_users(30)
        net = Network(bs, users)
        net.assign_cells()
        counts = net.users_per_bs()
        assert sum(counts.values()) == 30

    def test_edge_users_all_when_threshold_very_high(self):
        """Con umbral muy alto todos los usuarios son de borde."""
        bs = [make_macro()]
        users = make_users(10)
        net = Network(bs, users)
        net.assign_cells()
        edge = net.edge_users(rsrp_threshold_dbm=0.0)
        assert len(edge) == 10

    def test_edge_users_none_when_threshold_very_low(self):
        """Con umbral muy bajo ningún usuario es de borde."""
        bs = [make_macro()]
        users = make_users(10)
        net = Network(bs, users)
        net.assign_cells()
        edge = net.edge_users(rsrp_threshold_dbm=-200.0)
        assert len(edge) == 0
