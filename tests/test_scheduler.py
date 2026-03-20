"""
Tests unitarios para el scheduler Proportional Fair.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.network import MacroCell, SmallCell, UserEquipment, Network
from src.scheduler import ProportionalFairScheduler


def make_network(n_users=10, include_sc=False, seed=0):
    rng = np.random.default_rng(seed)
    macro = MacroCell(x=500, y=500)
    bss = [macro]
    if include_sc:
        bss.append(SmallCell(x=700, y=650, name="SC1"))

    users = [
        UserEquipment(
            x=float(rng.uniform(0, 1000)),
            y=float(rng.uniform(0, 1000)),
            uid=i,
        )
        for i in range(n_users)
    ]
    net = Network(bss, users)
    net.assign_cells()
    return net


class TestProportionalFairScheduler:
    def test_throughput_positive_after_run(self):
        net = make_network(10)
        sched = ProportionalFairScheduler(net)
        sched.run(n_steps=10)
        assert all(ue.throughput_bps > 0 for ue in net.users)

    def test_summary_keys(self):
        net = make_network(10)
        sched = ProportionalFairScheduler(net)
        sched.run(n_steps=5)
        s = sched.summary()
        for key in ["mean_bps", "median_bps", "min_bps", "max_bps",
                    "5th_percentile_bps"]:
            assert key in s

    def test_summary_mean_positive(self):
        net = make_network(20)
        sched = ProportionalFairScheduler(net)
        sched.run(n_steps=50)
        assert sched.summary()["mean_bps"] > 0

    def test_avg_throughput_initialized(self):
        net = make_network(5)
        sched = ProportionalFairScheduler(net)
        assert all(v > 0 for v in sched.avg_throughput.values())

    def test_pf_with_hetnet(self):
        """El scheduler PF funciona también en una red heterogénea."""
        net = make_network(20, include_sc=True)
        sched = ProportionalFairScheduler(net)
        sched.run(n_steps=50)
        assert all(ue.throughput_bps >= 0 for ue in net.users)
