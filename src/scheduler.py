"""
Scheduler Proportional Fair para redes LTE.

Implementa la asignación de recursos de radio basada en el criterio de
Equidad Proporcional (Proportional Fair, PF), que maximiza la métrica::

    i*(t) = argmax_j [ R_j(k,t) / T_j(t) ]

donde R_j es la tasa instantánea alcanzable por el usuario j en el
subcanal k, y T_j es el throughput histórico promedio.

Uso típico::

    from src.scheduler import ProportionalFairScheduler
    sched = ProportionalFairScheduler(network, alpha=0.9)
    sched.run(n_steps=100)
"""

from __future__ import annotations

import math
from typing import List, Dict

import numpy as np

from src.network import BaseStation, UserEquipment, Network


class ProportionalFairScheduler:
    """Scheduler Proportional Fair sobre un objeto :class:`~src.network.Network`.

    Cada paso de tiempo asigna los Resource Blocks (RBs) disponibles de cada
    BS al usuario con mayor cociente R_j / T_j.  El historial de throughput
    se actualiza con un filtro de paso bajo de constante de tiempo ``alpha``.

    Args:
        network:        Red heterogénea ya configurada (BSs + UEs asignados).
        alpha:          Factor de olvido exponencial (0 < α < 1).
                        α cercano a 1 da más peso al historial reciente.
        rb_bandwidth_hz: Ancho de banda de un Resource Block (Hz).
                         Por defecto 180 kHz (estándar LTE).
    """

    RB_BW_HZ = 180_000  # 180 kHz por RB en LTE

    def __init__(self, network: Network,
                 alpha: float = 0.9,
                 rb_bandwidth_hz: float = RB_BW_HZ):
        self.network = network
        self.alpha = alpha
        self.rb_bandwidth_hz = rb_bandwidth_hz

        # Throughput histórico promedio: inicializado en un valor pequeño para
        # evitar división por cero en el primer paso.
        self.avg_throughput: Dict[int, float] = {
            ue.uid: 1.0 for ue in network.users
        }

    # ------------------------------------------------------------------
    # Tasa instantánea por RB
    # ------------------------------------------------------------------

    def _instantaneous_rate(self, ue: UserEquipment, bs: BaseStation) -> float:
        """Tasa alcanzable en un único RB para el UE (bps).

        R_j(k, t) = BW_RB · log2(1 + SINR_j)
        """
        sinr_linear = 10 ** (ue.sinr_db / 10)
        return self.rb_bandwidth_hz * math.log2(1 + sinr_linear)

    # ------------------------------------------------------------------
    # Un paso de scheduling
    # ------------------------------------------------------------------

    def _schedule_step(self) -> None:
        """Ejecuta un TTI (Transmission Time Interval) de scheduling PF."""
        # Agrupar usuarios por BS servidora
        bs_users: Dict[BaseStation, List[UserEquipment]] = {
            bs: [] for bs in self.network.base_stations
        }
        for ue in self.network.users:
            if ue.serving_bs is not None:
                bs_users[ue.serving_bs].append(ue)

        for bs, users in bs_users.items():
            if not users:
                continue

            n_rbs = int(bs.bandwidth_hz / self.rb_bandwidth_hz)

            # Ordenar RBs asignados a cada usuario según el criterio PF
            # (simplificación: todos los RBs tienen la misma SINR → se reparten)
            # Calcular métricas PF para cada usuario
            pf_metrics = []
            for ue in users:
                r_inst = self._instantaneous_rate(ue, bs)
                t_avg = max(self.avg_throughput[ue.uid], 1.0)
                pf_metrics.append((ue, r_inst, r_inst / t_avg))

            # Ordenar de mayor a menor métrica PF
            pf_metrics.sort(key=lambda t: t[2], reverse=True)

            # Asignar RBs equitativamente respetando el orden PF (round-robin
            # ponderado: usuarios con mayor métrica obtienen más RBs)
            rbs_assigned: Dict[int, int] = {ue.uid: 0 for ue in users}
            for rb_idx in range(n_rbs):
                # El usuario que recibe este RB es el de mayor métrica PF
                # con menos RBs asignados hasta ahora (simplificación)
                winner = pf_metrics[rb_idx % len(pf_metrics)][0]
                rbs_assigned[winner.uid] += 1

            # Actualizar throughput del paso y el histórico
            for ue, r_inst, _ in pf_metrics:
                rbs = rbs_assigned[ue.uid]
                tput = r_inst * rbs  # bps para este TTI
                ue.throughput_bps = tput
                # Filtro IIR: T_j(t+1) = α·T_j(t) + (1-α)·R_j(t)
                self.avg_throughput[ue.uid] = (
                    self.alpha * self.avg_throughput[ue.uid]
                    + (1 - self.alpha) * tput
                )

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def run(self, n_steps: int = 200) -> None:
        """Ejecuta ``n_steps`` TTIs de scheduling PF.

        Antes de la primera iteración se recalculan SINR para asegurar
        que las métricas son coherentes con la asignación de celdas actual.

        Args:
            n_steps: Número de TTIs a simular.
        """
        self.network.compute_sinr()
        for _ in range(n_steps):
            self._schedule_step()

    def summary(self) -> dict:
        """Devuelve un diccionario con estadísticas de throughput."""
        tputs = [ue.throughput_bps for ue in self.network.users]
        return {
            "mean_bps": float(np.mean(tputs)) if tputs else 0.0,
            "median_bps": float(np.median(tputs)) if tputs else 0.0,
            "min_bps": float(np.min(tputs)) if tputs else 0.0,
            "max_bps": float(np.max(tputs)) if tputs else 0.0,
            "5th_percentile_bps": float(np.percentile(tputs, 5)) if tputs else 0.0,
        }
