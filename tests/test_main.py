"""
Tests de integración para el script principal (main.py).
Verifica que la simulación completa se ejecuta sin errores y produce
ficheros de salida.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import (
    build_base_stations,
    generate_users,
    run_scenario_rr,
    run_scenario_pf,
    _copy_users,
    main,
    AREA_M,
)
from src.analysis import RESULTS_DIR, handover_analysis


class TestBuildBaseStations:
    def test_macro_only(self):
        bss = build_base_stations(include_small_cells=False)
        assert len(bss) == 1
        assert bss[0].cell_type == "macro"

    def test_hetnet(self):
        bss = build_base_stations(include_small_cells=True)
        assert len(bss) == 3
        types = [bs.cell_type for bs in bss]
        assert types.count("macro") == 1
        assert types.count("small") == 2


class TestGenerateUsers:
    def test_correct_count(self):
        users = generate_users(50)
        assert len(users) == 50

    def test_unique_uids(self):
        users = generate_users(30)
        uids = [ue.uid for ue in users]
        assert len(set(uids)) == 30

    def test_positions_within_area(self):
        users = generate_users(100)
        for ue in users:
            assert 0 <= ue.x <= AREA_M[0]
            assert 0 <= ue.y <= AREA_M[1]

    def test_uniform_no_hotspot(self):
        users = generate_users(50, hotspot=False)
        assert len(users) == 50


class TestRunScenarios:
    def test_rr_all_users_have_throughput(self):
        bss = build_base_stations(False)
        users = generate_users(20)
        net = run_scenario_rr(bss, users)
        assert all(ue.throughput_bps > 0 for ue in users)

    def test_pf_all_users_have_throughput(self):
        bss = build_base_stations(True)
        users = generate_users(20)
        net = run_scenario_pf(bss, users, pf_steps=20)
        assert all(ue.throughput_bps >= 0 for ue in users)


class TestHandoverAnalysis:
    def test_handover_stats_keys(self):
        bss_macro = build_base_stations(False)
        bss_hetnet = build_base_stations(True)
        users = generate_users(30)

        net_macro = run_scenario_rr(bss_macro, _copy_users(users))
        net_hetnet = run_scenario_rr(bss_hetnet, _copy_users(users))

        stats = handover_analysis(net_macro, net_hetnet)
        for key in ["n_users", "handover_triggered", "macro_offload_pct",
                    "edge_users_macro_only", "edge_users_hetnet"]:
            assert key in stats

    def test_macro_offload_nonnegative(self):
        bss_macro = build_base_stations(False)
        bss_hetnet = build_base_stations(True)
        users = generate_users(50)

        net_macro = run_scenario_rr(bss_macro, _copy_users(users))
        net_hetnet = run_scenario_rr(bss_hetnet, _copy_users(users))

        stats = handover_analysis(net_macro, net_hetnet)
        assert stats["macro_offload_pct"] >= 0


class TestMainScript:
    def test_main_runs_without_error(self, tmp_path, monkeypatch):
        """La ejecución completa de main() no debe lanzar excepciones."""
        monkeypatch.chdir(tmp_path)
        # Usar pocos usuarios y pasos para que el test sea rápido
        main(["--users", "20", "--pf-steps", "10", "--seed", "0"])

    def test_output_files_created(self, tmp_path, monkeypatch):
        """main() debe crear los ficheros de imagen en results/."""
        import src.analysis as analysis_module
        original_dir = analysis_module.RESULTS_DIR
        results_path = str(tmp_path / "results")
        monkeypatch.setattr(analysis_module, "RESULTS_DIR", results_path)

        main(["--users", "20", "--pf-steps", "10", "--seed", "1"])

        expected_files = [
            "rsrp_macro_only.png",
            "rsrp_hetnet.png",
            "deployment_macro_only.png",
            "deployment_hetnet.png",
            "throughput_comparison.png",
            "sinr_histogram.png",
            "load_distribution.png",
        ]
        for fname in expected_files:
            fpath = os.path.join(results_path, fname)
            assert os.path.exists(fpath), f"Fichero no encontrado: {fname}"

        monkeypatch.setattr(analysis_module, "RESULTS_DIR", original_dir)
