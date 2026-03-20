"""
Tests unitarios para el modelo de propagación COST-231 Hata.
"""

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propagation import (
    antenna_correction_factor,
    cost231_hata_path_loss,
    cost231_hata_path_loss_array,
    rsrp_dbm,
)


class TestAntennaCorrectionFactor:
    def test_large_city_above_400mhz(self):
        """Para ciudades grandes y f >= 400 MHz usa la fórmula cuadrática."""
        a = antenna_correction_factor(1800, 1.5, "large")
        expected = 3.2 * (math.log10(11.75 * 1.5)) ** 2 - 4.97
        assert abs(a - expected) < 1e-9

    def test_medium_city(self):
        """Para ciudades medianas usa la fórmula lineal."""
        f = 1800
        h = 1.5
        a = antenna_correction_factor(f, h, "medium")
        expected = (1.1 * math.log10(f) - 0.7) * h - (1.56 * math.log10(f) - 0.8)
        assert abs(a - expected) < 1e-9

    def test_large_city_below_400mhz_falls_back_to_medium(self):
        """Frecuencias < 400 MHz usan la fórmula de ciudad mediana incluso con 'large'."""
        a_large = antenna_correction_factor(300, 1.5, "large")
        a_medium = antenna_correction_factor(300, 1.5, "medium")
        assert abs(a_large - a_medium) < 1e-9


class TestCost231HataPathLoss:
    def test_positive_path_loss(self):
        """La pérdida de trayectoria debe ser positiva para distancias razonables."""
        pl = cost231_hata_path_loss(1800, 25, 1.5, 500)
        assert pl > 0

    def test_increases_with_distance(self):
        """La pérdida aumenta al aumentar la distancia."""
        pl_near = cost231_hata_path_loss(1800, 25, 1.5, 100)
        pl_far = cost231_hata_path_loss(1800, 25, 1.5, 1000)
        assert pl_far > pl_near

    def test_zero_distance_returns_zero(self):
        """Distancia 0 → pérdida 0."""
        pl = cost231_hata_path_loss(1800, 25, 1.5, 0)
        assert pl == 0.0

    def test_large_city_higher_path_loss_than_medium(self):
        """Ciudades grandes tienen más pérdida de trayectoria que ciudades medianas."""
        pl_large = cost231_hata_path_loss(1800, 25, 1.5, 500, "large")
        pl_medium = cost231_hata_path_loss(1800, 25, 1.5, 500, "medium")
        # La ciudad grande aplica C=3 dB; la diferencia total incluye también
        # la variación del factor de corrección de antena a(h_rx).
        assert pl_large > pl_medium

    def test_higher_bs_less_path_loss(self):
        """Una antena más alta produce menos pérdida de trayectoria."""
        pl_low = cost231_hata_path_loss(1800, 10, 1.5, 500)
        pl_high = cost231_hata_path_loss(1800, 40, 1.5, 500)
        assert pl_high < pl_low

    def test_typical_urban_range(self):
        """Para 500 m en entorno urbano la pérdida debe estar en un rango razonable."""
        pl = cost231_hata_path_loss(1800, 25, 1.5, 500, "large")
        assert 90 < pl < 150, f"Pérdida fuera de rango esperado: {pl:.1f} dB"


class TestCost231HataPathLossArray:
    def test_array_matches_scalar(self):
        """La versión array debe coincidir con la escalar punto a punto."""
        distances = np.array([100.0, 300.0, 500.0, 1000.0])
        pl_array = cost231_hata_path_loss_array(1800, 25, 1.5, distances, "large")
        for d, pl_a in zip(distances, pl_array):
            pl_s = cost231_hata_path_loss(1800, 25, 1.5, d, "large")
            assert abs(pl_a - pl_s) < 1e-6

    def test_2d_array_shape_preserved(self):
        """El resultado tiene la misma forma que el array de entrada."""
        distances = np.ones((10, 10)) * 500
        pl = cost231_hata_path_loss_array(1800, 25, 1.5, distances)
        assert pl.shape == (10, 10)

    def test_minimum_is_zero(self):
        """La pérdida mínima devuelta es 0 (nunca negativa)."""
        distances = np.array([0.001, 0.01, 0.1])
        pl = cost231_hata_path_loss_array(1800, 25, 1.5, distances)
        assert np.all(pl >= 0)


class TestRsrpDbm:
    def test_rsrp_decreases_with_path_loss(self):
        """A mayor pérdida, menor RSRP."""
        rsrp1 = rsrp_dbm(45, 80)
        rsrp2 = rsrp_dbm(45, 120)
        assert rsrp1 > rsrp2

    def test_rsrp_typical_value(self):
        """RSRP = P_tx - PL."""
        assert rsrp_dbm(45, 100) == pytest.approx(-55.0)
