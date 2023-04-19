import os
from unittest import *

import pytest

from network_schedule import NetworkSchedule
from session_metadata import SessionMetadata


class TestNetworkSchedule(TestCase):

    def test_network_schedule_correct(self):
        ns = NetworkSchedule(dataset_id=0, n_sessions=2, sessions=[0, 0, 1], start_times=[0, int(5e5), int(1e6)])
        self.assertEqual(ns.get_session_start_times(0), [0, 500_000])
        self.assertEqual(ns.get_session_start_times(1), [1_000_000])

    def test_network_schedule_incomplete(self):
        ns = NetworkSchedule(dataset_id=0, n_sessions=2, sessions=[0, 0, 1], start_times=[0, int(5e5), int(1e6)])
        with pytest.raises(ValueError):
            ns.get_session_start_times(2)

    def test_calculate_length(self):
        total_alice = sum([b.duration for b in SessionMetadata("../configs/bqc_alice.yml").blocks])
        total_bob = sum([b.duration for b in SessionMetadata("../configs/bqc_bob.yml").blocks])
        ns = NetworkSchedule(dataset_id=0, n_sessions=6, save=False)
        self.assertEqual(ns.length, max(total_alice, total_bob) * 12)

    def test_calculate_probabilities(self):
        ns = NetworkSchedule(dataset_id=6, n_sessions=6, save=False)
        p = ns._calculate_probabilities()
        min_sep = {
            "bqc": 20000,
            "pingpong": 222000,
            "qkd": 15000,
        }
        sessions = {
            "bqc": 2,
            "pingpong": 2,
            "qkd": 5,
        }
        for name in ["bqc", "pingpong", "qkd"]:
            rate = ns.length / (ns.QC_LENGTH + min_sep[name])
            self.assertEqual(p[name], 1 / rate * sessions[name] * 2)

    def test_generate_random_ns(self):
        ns = NetworkSchedule(dataset_id=6, n_sessions=18, save=False)
        for name in ["bqc", "pingpong", "qkd"]:
            self.assertTrue(len(ns.get_session_start_times(name)) > 0)

    def test_save_network_schedule(self):
        filename = "test_network_schedule"
        for i in range(6):
            NetworkSchedule(dataset_id=0, n_sessions=6, filename=filename)
            self.assertTrue(f"{filename}_id-{i}.csv" in os.listdir("../network_schedules"))

        for i in range(6):
            os.remove(f"../network_schedules/{filename}_id-{i}.csv")
