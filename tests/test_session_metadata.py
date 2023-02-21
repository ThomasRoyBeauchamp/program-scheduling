from session_metadata import BlockMetadata, SessionMetadata
from unittest import *


class TestBlockMetadata(TestCase):

    def setUp(self):
        self.config = {"comm_q": 1, "storage_q": 1, "instructions": [1, 2, 0, 0], "CS_id": None}

    def test_block_metadata_correct(self):
        bm = BlockMetadata(self.config)
        self.assertEqual(bm.comm_q, self.config["comm_q"])
        self.assertEqual(bm.storage_q, self.config["storage_q"])
        self.assertIs(bm.instructions, self.config["instructions"])
        self.assertIsNone(bm.CS_id)
        pass

    def test_block_metadata_wrong_keys(self):
        # TODO
        pass

    def test_block_metadata_wrong_instructions(self):
        # TODO
        pass


class TestSessionMetadata(TestCase):

    def setUp(self):
        # TODO
        pass

    def tearDown(self):
        # TODO
        pass

    def test_session_metadata_correct(self):
        # TODO
        pass

    def test_session_metadata_wrong_config(self):
        # TODO
        pass
