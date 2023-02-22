import yaml
import pytest
import os

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
        self.config.pop("instructions")
        with pytest.raises(ValueError):
            BlockMetadata(self.config)

    def test_block_metadata_wrong_instructions(self):
        self.config["instructions"] = [1, 1, 0]
        with pytest.raises(ValueError):
            BlockMetadata(self.config)


class TestSessionMetadata(TestCase):

    def setUp(self):
        self.yaml_file = "test_config.yaml"
        with open(self.yaml_file, 'r') as file_handle:
            self.config = yaml.load(file_handle, yaml.SafeLoader) or {}

    def test_session_metadata_correct(self):
        sm = SessionMetadata(self.yaml_file)
        self.assertEqual(sm.session_id, self.config.get("session_id"))
        self.assertEqual(sm.app_deadline, self.config.get("app_deadline"))
        self.assertEqual(sm.T1, self.config.get("T1"))
        self.assertEqual(sm.T2, self.config.get("T2"))
        self.assertEqual(sm.gate_duration, self.config.get("gate_duration"))
        self.assertEqual(sm.gate_fidelity, self.config.get("gate_fidelity"))
        self.assertEqual(sm.cc_duration, self.config.get("cc_duration"))
        self.assertEqual(len(sm.blocks), len(self.config.get("blocks")))

    def test_session_metadata_custom_id(self):
        sm = SessionMetadata(self.yaml_file, session_id=42)
        self.assertEqual(sm.session_id, 42)

    def test_session_metadata_wrong_config(self):
        self.config.pop("app_deadline")
        yaml.dump(self.config, open("temp.yaml", "w"))
        with pytest.raises(ValueError):
            SessionMetadata("temp.yaml")
        os.remove("temp.yaml")
