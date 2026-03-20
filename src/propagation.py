"""
Modelos de propagación de radio para redes LTE heterogéneas.

Implementa el modelo empírico COST-231 Hata para predicción de pérdida de
trayectoria en entornos urbanos, utilizado tanto para macroceldas como para
small cells en el escenario de zona de alta demanda.

Referencias:
- COST Action 231, "Digital Mobile Radio: COST 231 Final Report", 1999.
- Okumura-Hata extended to 2 GHz band.
"""

import math
import numpy as np


def antenna_correction_factor(freq_mhz: float, h_rx: float,
                               city_size: str = "large") -> float:
    """Calcula el factor de corrección de altura de antena móvil a(h_rx).

    Args:
        freq_mhz: Frecuencia de portadora en MHz.
        h_rx:     Altura de la antena receptora (UE) en metros.
        city_size: Tamaño de ciudad: 'large' para ciudades grandes (≥400 MHz),
                   'medium' para ciudades pequeñas/medianas o suburbanas.

    Returns:
        Factor de corrección a(h_rx) en dB.
    """
    if city_size == "large" and freq_mhz >= 400:
        return 3.2 * (math.log10(11.75 * h_rx)) ** 2 - 4.97
    # Ciudades medianas/pequeñas y áreas suburbanas
    return ((1.1 * math.log10(freq_mhz) - 0.7) * h_rx
            - (1.56 * math.log10(freq_mhz) - 0.8))


def cost231_hata_path_loss(freq_mhz: float, h_tx: float, h_rx: float,
                            distance_m: float, city_size: str = "large") -> float:
    """Pérdida de trayectoria según el modelo COST-231 Hata (dB).

    Válido para:
        - Frecuencias: 1500–2000 MHz
        - Distancias: 1–20 km (se aceptan distancias menores con menor precisión)
        - Altura de la estación base: 30–200 m
        - Altura del terminal: 1–10 m

    Fórmula::

        PL_dB = 46.3 + 33.9·log10(f) - 13.82·log10(h_tx) - a(h_rx)
                + (44.9 - 6.55·log10(h_tx))·log10(d) + C_dB

    donde C_dB = 0 dB para ciudades medianas y 3 dB para metropolitanas.

    Args:
        freq_mhz:   Frecuencia portadora en MHz.
        h_tx:       Altura de la antena transmisora (BS) en metros.
        h_rx:       Altura de la antena receptora (UE) en metros. Por defecto 1.5 m.
        distance_m: Distancia BS-UE en **metros** (se convierte internamente a km).
        city_size:  'large' (ciudad grande, C=3 dB) o 'medium' (C=0 dB).

    Returns:
        Pérdida de trayectoria en dB.  Valor mínimo devuelto: 0 dB.
    """
    if distance_m <= 0:
        return 0.0

    d_km = max(distance_m / 1000.0, 0.001)  # evitar log(0)
    a_hrx = antenna_correction_factor(freq_mhz, h_rx, city_size)
    c_db = 3.0 if city_size == "large" else 0.0

    pl = (46.3
          + 33.9 * math.log10(freq_mhz)
          - 13.82 * math.log10(h_tx)
          - a_hrx
          + (44.9 - 6.55 * math.log10(h_tx)) * math.log10(d_km)
          + c_db)
    return max(pl, 0.0)


def cost231_hata_path_loss_array(freq_mhz: float, h_tx: float, h_rx: float,
                                  distances_m: np.ndarray,
                                  city_size: str = "large") -> np.ndarray:
    """Versión vectorizada de :func:`cost231_hata_path_loss`.

    Args:
        freq_mhz:    Frecuencia portadora en MHz.
        h_tx:        Altura transmisora en metros.
        h_rx:        Altura receptora en metros.
        distances_m: Array de distancias en metros (forma arbitraria).
        city_size:   'large' o 'medium'.

    Returns:
        Array de pérdidas de trayectoria en dB con la misma forma que
        ``distances_m``.
    """
    distances_m = np.asarray(distances_m, dtype=float)
    d_km = np.maximum(distances_m / 1000.0, 0.001)
    a_hrx = antenna_correction_factor(freq_mhz, h_rx, city_size)
    c_db = 3.0 if city_size == "large" else 0.0

    pl = (46.3
          + 33.9 * np.log10(freq_mhz)
          - 13.82 * np.log10(h_tx)
          - a_hrx
          + (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d_km)
          + c_db)
    return np.maximum(pl, 0.0)


def rsrp_dbm(tx_power_dbm: float, path_loss_db: float) -> float:
    """Calcula la RSRP (Reference Signal Received Power) en dBm.

    Modelo simplificado: RSRP = P_tx - PL  (sin ganancias de antena adicionales).

    Args:
        tx_power_dbm: Potencia de transmisión de la BS en dBm.
        path_loss_db: Pérdida de trayectoria en dB.

    Returns:
        RSRP en dBm.
    """
    return tx_power_dbm - path_loss_db
