from activity_metadata import ActivityMetadata, ActiveSet
from session_metadata import SessionMetadata
from unittest import *


class TestActivityMetadata(TestCase):

    def setUp(self):
        self.sm = SessionMetadata("../session_configs/qkd.yaml")
        pass

    def test_activity_metadata_correct(self):
        am = ActivityMetadata(self.sm)
        print(am.resource_reqs)
        pass

    def test_calculate_resource_reqs(self):
        # TODO
        pass

    def test_calculate_duration(self):
        # TODO
        pass

    def test_calculate_time_lags(self):
        # TODO
        pass


class TestActiveSet(TestCase):

    def setUp(self):
        # TODO
        pass

    def test_create_active_set(self):
        # TODO
        pass

    def test_merging_active_sets(self):
        # TODO
        pass
