import pytest

from network_schedule import NetworkSchedule
from unittest import *


class TestNetworkSchedule(TestCase):

    def test_network_schedule_correct(self):
        ns = NetworkSchedule([1, 3], [2, 4], [0, 1])
        self.assertEqual(ns.get_session_start_time(0), 1)
        self.assertEqual(ns.get_session_duration(0), 2)
        self.assertEqual(ns.get_session_start_time(1), 3)
        self.assertEqual(ns.get_session_duration(1), 4)

    def test_network_schedule_empty(self):
        ns = NetworkSchedule()
        self.assertIsNone(ns.get_session_duration(1))
        self.assertIsNone(ns.get_session_start_time(1))

    def test_network_schedule_incomplete(self):
        ns = NetworkSchedule([1, 3], [2, 4], [0, 1])
        with pytest.raises(ValueError):
            ns.get_session_duration(2)
        with pytest.raises(ValueError):
            ns.get_session_start_time(2)
