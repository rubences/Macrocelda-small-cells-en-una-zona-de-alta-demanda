"""
Nodos de red y modelo de usuario para el escenario macro + small cells.

Define las clases:
- BaseStation  – estación base genérica (macro o small cell).
- MacroCell    – macrocelda LTE (800/1800 MHz, 43-46 dBm, 20-25 m).
- SmallCell    – small cell LTE (1800/2600 MHz, 24-30 dBm, 4-6 m).
- UserEquipment – terminal de usuario con posición y asignación de celda.
- Network       – contenedor de BSs y UEs; calcula RSRP, SINR y asignación.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.propagation import cost231_hata_path_loss, rsrp_dbm


# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------
NOISE_FIGURE_DB = 7.0          # Factor de ruido del receptor (dB)
THERMAL_NOISE_DBM_HZ = -174.0  # Densidad espectral de ruido térmico a 290 K (dBm/Hz)
UE_HEIGHT_M = 1.5              # Altura del terminal de usuario (m)


def _thermal_noise_dbm(bandwidth_hz: float) -> float:
    """Potencia de ruido térmico en dBm para un ancho de banda dado."""
    return THERMAL_NOISE_DBM_HZ + 10 * math.log10(bandwidth_hz) + NOISE_FIGURE_DB


# ---------------------------------------------------------------------------
# Estaciones base
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class BaseStation:
    """Estación base LTE genérica.

    Attributes:
        x, y:          Posición en metros dentro del área de simulación.
        tx_power_dbm:  Potencia de transmisión en dBm.
        height_m:      Altura de la antena en metros.
        freq_mhz:      Frecuencia portadora en MHz.
        bandwidth_mhz: Ancho de banda del canal en MHz.
        cell_type:     'macro' o 'small'.
        name:          Identificador legible.
        city_size:     Parámetro del modelo COST-231 ('large' o 'medium').
    """
    x: float
    y: float
    tx_power_dbm: float
    height_m: float
    freq_mhz: float
    bandwidth_mhz: float
    cell_type: str
    name: str
    city_size: str = "large"

    @property
    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    def distance_to(self, ux: float, uy: float) -> float:
        """Distancia euclidiana en metros a un punto (ux, uy)."""
        return math.sqrt((self.x - ux) ** 2 + (self.y - uy) ** 2)

    def path_loss_db(self, ux: float, uy: float) -> float:
        """Pérdida de trayectoria COST-231 Hata hacia (ux, uy)."""
        d = self.distance_to(ux, uy)
        return cost231_hata_path_loss(
            self.freq_mhz, self.height_m, UE_HEIGHT_M, d, self.city_size
        )

    def rsrp_at(self, ux: float, uy: float) -> float:
        """RSRP en dBm en el punto (ux, uy)."""
        return rsrp_dbm(self.tx_power_dbm, self.path_loss_db(ux, uy))

    def noise_dbm(self) -> float:
        """Potencia de ruido térmico en dBm para el ancho de banda del canal."""
        return _thermal_noise_dbm(self.bandwidth_hz)


def MacroCell(x: float, y: float,
              tx_power_dbm: float = 45.0,
              height_m: float = 25.0,
              freq_mhz: float = 1800.0,
              bandwidth_mhz: float = 20.0,
              name: str = "Macro") -> BaseStation:
    """Crea una macrocelda LTE con parámetros típicos (43-46 dBm, 20-25 m)."""
    return BaseStation(
        x=x, y=y,
        tx_power_dbm=tx_power_dbm,
        height_m=height_m,
        freq_mhz=freq_mhz,
        bandwidth_mhz=bandwidth_mhz,
        cell_type="macro",
        name=name,
        city_size="large",
    )


def SmallCell(x: float, y: float,
              tx_power_dbm: float = 27.0,
              height_m: float = 5.0,
              freq_mhz: float = 2600.0,
              bandwidth_mhz: float = 20.0,
              name: str = "SC") -> BaseStation:
    """Crea una small cell LTE con parámetros típicos (24-30 dBm, 4-6 m)."""
    return BaseStation(
        x=x, y=y,
        tx_power_dbm=tx_power_dbm,
        height_m=height_m,
        freq_mhz=freq_mhz,
        bandwidth_mhz=bandwidth_mhz,
        cell_type="small",
        name=name,
        city_size="medium",
    )


# ---------------------------------------------------------------------------
# Equipo de usuario
# ---------------------------------------------------------------------------

@dataclass
class UserEquipment:
    """Terminal de usuario con posición y métricas de rendimiento.

    Attributes:
        x, y:        Posición en metros.
        uid:         Identificador único.
        serving_bs:  BS asignada (tras la selección de celda).
        rsrp_dbm:    RSRP de la celda servidora (dBm).
        sinr_db:     SINR experimentada (dB).
        throughput_bps: Throughput estimado (bps).
    """
    x: float
    y: float
    uid: int
    serving_bs: Optional[BaseStation] = field(default=None, repr=False)
    rsrp_dbm: float = -200.0
    sinr_db: float = -50.0
    throughput_bps: float = 0.0


# ---------------------------------------------------------------------------
# Red heterogénea
# ---------------------------------------------------------------------------

class Network:
    """Modelo de red LTE heterogénea.

    Gestiona la lista de BSs y UEs, realiza la selección de celda con soporte
    de Cell Range Expansion (CRE) y calcula el SINR y throughput de cada usuario.

    Args:
        base_stations: Lista de objetos :class:`BaseStation`.
        users:         Lista de objetos :class:`UserEquipment`.
        cre_offset_db: Sesgo de offset para small cells en CRE (dB).
                       0 → selección tradicional por RSRP máxima.
    """

    def __init__(self, base_stations: List[BaseStation],
                 users: List[UserEquipment],
                 cre_offset_db: float = 0.0):
        self.base_stations = base_stations
        self.users = users
        self.cre_offset_db = cre_offset_db

    # ------------------------------------------------------------------
    # Selección de celda
    # ------------------------------------------------------------------

    def _effective_rsrp(self, bs: BaseStation, ux: float, uy: float) -> float:
        """RSRP efectiva con sesgo CRE aplicado a small cells."""
        rsrp = bs.rsrp_at(ux, uy)
        if bs.cell_type == "small" and self.cre_offset_db != 0.0:
            rsrp += self.cre_offset_db
        return rsrp

    def assign_cells(self) -> None:
        """Asigna cada UE a la BS con mayor RSRP (+ offset CRE si aplica).

        Cell = argmax_i(RSRP_i + offset_bias_i)
        """
        for ue in self.users:
            best_bs = max(
                self.base_stations,
                key=lambda bs: self._effective_rsrp(bs, ue.x, ue.y)
            )
            ue.serving_bs = best_bs
            ue.rsrp_dbm = best_bs.rsrp_at(ue.x, ue.y)

    # ------------------------------------------------------------------
    # SINR y capacidad
    # ------------------------------------------------------------------

    def compute_sinr(self) -> None:
        """Calcula el SINR de cada UE considerando interferencia inter-celda.

        Modelo simplificado: todos los recursos se comparten en el mismo
        subcanal (worst-case interference).  Si la macro y las small cells
        operan en bandas distintas, la interferencia cruzada es 0.

        SINR = S / (N + I)
        donde:
          S = potencia recibida de la BS servidora (mW)
          N = ruido térmico de la BS servidora (mW)
          I = suma de interferencias de las BSs co-canal (mW)
        """
        for ue in self.users:
            if ue.serving_bs is None:
                continue

            s_bs = ue.serving_bs
            pl_s = s_bs.path_loss_db(ue.x, ue.y)
            signal_dbm = s_bs.tx_power_dbm - pl_s
            signal_mw = 10 ** (signal_dbm / 10)

            noise_mw = 10 ** (s_bs.noise_dbm() / 10)

            # Interferencia de las demás BSs co-canal (misma frecuencia)
            interference_mw = 0.0
            for bs in self.base_stations:
                if bs is s_bs:
                    continue
                if abs(bs.freq_mhz - s_bs.freq_mhz) > 100:
                    # Bandas distintas: sin interferencia relevante
                    continue
                pl_i = bs.path_loss_db(ue.x, ue.y)
                interferer_mw = 10 ** ((bs.tx_power_dbm - pl_i) / 10)
                interference_mw += interferer_mw

            sinr = signal_mw / (noise_mw + interference_mw)
            ue.sinr_db = 10 * math.log10(max(sinr, 1e-10))

    # ------------------------------------------------------------------
    # Throughput (Shannon / spectral efficiency)
    # ------------------------------------------------------------------

    def compute_throughput(self) -> None:
        """Estima el throughput de cada UE dividiendo el ancho de banda entre
        los usuarios asignados a la misma BS.

        T_x = (BW_servidora / N_usuarios_servidora) · log2(1 + SINR_x)

        Este modelo de scheduler aplica una asignación equitativa (Round Robin)
        como punto de partida.  Para Proportional Fair ver :mod:`src.scheduler`.
        """
        # Contar usuarios por BS
        user_counts: dict = {bs: 0 for bs in self.base_stations}
        for ue in self.users:
            if ue.serving_bs is not None:
                user_counts[ue.serving_bs] += 1

        for ue in self.users:
            if ue.serving_bs is None:
                continue
            n_users = max(user_counts[ue.serving_bs], 1)
            bw_per_user = ue.serving_bs.bandwidth_hz / n_users
            sinr_linear = 10 ** (ue.sinr_db / 10)
            ue.throughput_bps = bw_per_user * math.log2(1 + sinr_linear)

    # ------------------------------------------------------------------
    # Ejecución completa
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta la cadena completa: asignación → SINR → throughput."""
        self.assign_cells()
        self.compute_sinr()
        self.compute_throughput()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def users_per_bs(self) -> dict:
        """Devuelve un diccionario {BS.name: número de usuarios asignados}."""
        counts: dict = {bs.name: 0 for bs in self.base_stations}
        for ue in self.users:
            if ue.serving_bs is not None:
                counts[ue.serving_bs.name] += 1
        return counts

    def mean_throughput_bps(self) -> float:
        """Throughput medio por usuario en bps."""
        tputs = [ue.throughput_bps for ue in self.users]
        return float(np.mean(tputs)) if tputs else 0.0

    def mean_sinr_db(self) -> float:
        """SINR media en dB."""
        sinrs = [ue.sinr_db for ue in self.users]
        return float(np.mean(sinrs)) if sinrs else 0.0

    def edge_users(self, rsrp_threshold_dbm: float = -100.0) -> List[UserEquipment]:
        """Devuelve los UEs con RSRP por debajo del umbral (usuarios de borde)."""
        return [ue for ue in self.users if ue.rsrp_dbm < rsrp_threshold_dbm]
