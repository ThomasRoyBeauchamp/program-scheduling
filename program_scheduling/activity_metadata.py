import logging
import sys

from program_scheduling.program_scheduling.session_metadata import SessionMetadata, BlockMetadata
from math import gcd, floor
from functools import reduce

logger = logging.getLogger(__name__)


class ActivityMetadata:

    def __init__(self, session_metadata: SessionMetadata, network_schedule=None):
        self.session_id = session_metadata.session_id
        self.app_deadline = session_metadata.app_deadline

        self.n_blocks = len(session_metadata.blocks)
        self.block_names = [b.name for b in session_metadata.blocks]
        self.cs_ids = [b.CS for b in session_metadata.blocks]
        self.types = [b.type for b in session_metadata.blocks]

        self.successors = [[i + 1] for i in range(self.n_blocks - 1)]
        self.successors.append([])

        self.resource_reqs = [self._calculate_resource_reqs(b) for b in session_metadata.blocks]

        self.durations = [b.duration for b in session_metadata.blocks]
        self._update_qc_blocks_based_on_network_schedule(network_schedule)
        self.d_max = self._calculate_time_lags(session_metadata)
        self.qc_indices = self._calculate_qc_indices(session_metadata)

    @staticmethod
    def _calculate_resource_reqs(block_metadata: BlockMetadata):
        # resource requirements should be defined as [CPU, QPU]
        classical = block_metadata.type[0] == "C"
        quantum = block_metadata.type[0] == "Q"
        # sanity check that block is either fully classical or fully quantum
        assert (classical and not quantum) or (not classical and quantum)
        return [int(classical), int(quantum)]

    @staticmethod
    def _calculate_qc_indices(session_metadata: SessionMetadata):
        last_index = 0
        qc_indices = []
        for b in session_metadata.blocks:
            if b.type != "QC":
                qc_indices.append(None)
            else:
                qc_indices.append(last_index)
                last_index += 1
        return qc_indices

    def _update_qc_blocks_based_on_network_schedule(self, network_schedule):
        """
        In the future, each session might have a different QC block length.

        :param network_schedule:
        :return:
        """
        if network_schedule is None:
            return
        for i in range(self.n_blocks):
            if self.types[i] == "QC":
                self.durations[i] = network_schedule.QC_LENGTH

    def _calculate_time_lags(self, session_metadata: SessionMetadata):
        """
        In the most trivial version of time lags, the only parameter taken into account is the critical section.
        This means that blocks within the same critical section should be executed without any delays.
        The rest of maximum time lags are set to the time remaining before the application deadline
        (minus the execution time of all the blocks) -- this might not be a sufficient constraint on itself,
        but along with the constraint for the application deadline to be satisfied, it is adequate.

        :param session_metadata:
        :return:
        """
        # we define min/max time lags between consecutive pairs of blocks
        # d^{min}_{ij} can be accessed through d_min[j] (and similarly for d^{max}_{ij})

        if self.app_deadline is not None:
            remaining_time = self.app_deadline - sum(self.durations)

        d_max = []
        for i, block in enumerate(session_metadata.blocks):
            if block.CS is None or i == 0:
                d_max.append(None if self.app_deadline is None else remaining_time)
            else:
                d_max.append(0 if session_metadata.blocks[i-1].CS == block.CS else None)

        assert len(d_max) == self.n_blocks  # sanity check
        return d_max


class ActiveSet:

    def __init__(self):
        self.n_blocks = 0
        self.ids = []
        self.successors = []
        self.resource_reqs = []
        self.types = []
        self.durations = []
        self.d_max = []
        self.block_names = []
        self.gcd = None
        self.qc_indices = []
        self.cs_ids = []

    # TODO: something like this is needed for the test but do it in a pythonic way (kwargs)
    # def __init__(self, n_blocks, ids, succ, reqs, types, durations, d_min, d_max, block_names, gcd):
    #     self.n_blocks = n_blocks
    #     self.ids = ids
    #     self.successors = succ
    #     self.resource_reqs = reqs
    #     self.types = types
    #     self.durations = durations
    #     self.d_min = d_min
    #     self.d_max = d_max
    #     self.block_names = block_names
    #     self.gcd = gcd

    @staticmethod
    def create_active_set(dataset, role, network_schedule):
        active = ActiveSet()
        logger.debug(f"Your active set now has the following sessions:")
        last_session_id = 0
        for (config_file, number) in dataset.items():
            ids = list(range(last_session_id, last_session_id + number))
            logger.debug(f"\t{number} sessions of {config_file}_{role} with IDs {ids}")
            for id in ids:
                session_metadata = SessionMetadata(yaml_file=config_file + f"_{role}.yml", session_id=id)
                active._merge_activity_metadata(ActivityMetadata(session_metadata, network_schedule))
            last_session_id += number
        return active

    def get_gcd(self):
        return reduce(gcd, self.durations)

    def scale_down(self):
        gcd = self.get_gcd()
        scaled_durations = [int(d / gcd) for d in self.durations]
        # these might not be integers but as long as d_max is rounded up, it should be fine
        scaled_d_max = [floor(d / gcd) if d is not None else None for d in self.d_max]
        return scaled_durations, scaled_d_max

    def _merge_activity_metadata(self, other: ActivityMetadata):
        if self.n_blocks == 0:  # we don't need to reindex successors
            self.successors += other.successors
        else:
            new_successors = [x if len(x) == 0 else [x[0] + self.n_blocks] for x in other.successors]
            self.successors += new_successors
        self.ids += [other.session_id] * other.n_blocks
        self.n_blocks += other.n_blocks
        self.resource_reqs += other.resource_reqs
        self.block_names += other.block_names
        self.types += other.types
        self.durations += other.durations
        self.d_max += other.d_max
        self.qc_indices += other.qc_indices
        self.cs_ids += other.cs_ids
