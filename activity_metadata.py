from session_metadata import SessionMetadata, BlockMetadata
from network_schedule import NetworkSchedule
from math import gcd
from functools import reduce


class ActivityMetadata:

    def __init__(self, session_metadata: SessionMetadata, network_schedule=None):
        self.session_id = session_metadata.session_id
        self.app_deadline = session_metadata.app_deadline
        self.block_names = [b.name for b in session_metadata.blocks]

        self.n_blocks = len(session_metadata.blocks)
        self.successors = [[i + 1] for i in range(self.n_blocks - 1)]
        self.successors.append([])
        self.resource_reqs = [self._calculate_resource_reqs(b) for b in session_metadata.blocks]
        self.types = [b.type for b in session_metadata.blocks]

        self.durations = [b.duration for b in session_metadata.blocks]
        self.d_min, self.d_max = self._calculate_time_lags(session_metadata)

    @staticmethod
    def _calculate_resource_reqs(block_metadata: BlockMetadata):
        # resource requirements should be defined as [CPU, QPU]
        classical = block_metadata.type[0] == "C"
        quantum = block_metadata.type[0] == "Q"
        # sanity check that block is either fully classical or fully quantum
        assert (classical and not quantum) or (not classical and quantum)
        return [int(classical), int(quantum)]

    @staticmethod
    def _calculate_duration(block_metadata: BlockMetadata, gate_duration, cc_duration,
                            session_id, network_schedule: NetworkSchedule):
        # TODO: this needs to be changed if considering real-time units
        cl_duration = 1  # constant for a classical local instruction
        if network_schedule is None or not network_schedule.is_defined or block_metadata.instructions[0] == 0:
            qc = block_metadata.instructions[0]
        else:
            # Currently any block involving quantum communication operations is as long as its time slot in the
            # network schedule.
            return network_schedule.get_session_duration(session_id)
        ql = block_metadata.instructions[1] * gate_duration
        cc = block_metadata.instructions[2] * cc_duration
        cl = block_metadata.instructions[3] * cl_duration
        return qc + ql + cc + cl

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

        remaining_time = self.app_deadline - sum(self.durations)

        d_max = []
        for i, block in enumerate(session_metadata.blocks):
            if block.CS is None or i == 0:
                d_max.append(remaining_time)
            else:
                d_max.append(0 if session_metadata.blocks[i-1].CS == block.CS else remaining_time)

        assert len(d_max) == self.n_blocks  # sanity check
        return [0] * self.n_blocks, d_max


class ActiveSet:

    def __init__(self):
        self.n_blocks = 0
        self.ids = []
        self.successors = []
        self.resource_reqs = []
        self.types = []
        self.durations = []
        self.d_min = []
        self.d_max = []
        self.block_names = []
        self.smallest_time_unit = None

    @staticmethod
    def create_active_set(configs: [ActivityMetadata], session_ids, network_schedule=None):
        if len(configs) != len(session_ids):
            raise ValueError(f"Session IDs for all activities need to be defined. Currently, there are {len(configs)} "
                             f"different config files set but {len(session_ids)} different sets of IDs defined.")

        active = ActiveSet()
        print(f"Your active set now has the following sessions:")
        for i, ids in enumerate(session_ids):
            print(f"\t{configs[i]} -- {len(ids)} sessions with IDs {ids}")
            for ID in ids:
                session_metadata = SessionMetadata(yaml_file=configs[i], session_id=ID)
                if network_schedule is None:
                    network_schedule = NetworkSchedule()
                active._merge_activity_metadata(ActivityMetadata(session_metadata, network_schedule))

        return active

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
        self.d_min += other.d_min
        self.d_max += other.d_max

    def update_durations(self):
        # TODO: this should also update the app_deadline and maybe only after that calculate the d_min/max
        self.smallest_time_unit = reduce(gcd, self.durations)
        self.durations = [int(d/self.smallest_time_unit) for d in self.durations]
        # TODO: these might not actually be integers
        self.d_min = [int(d/self.smallest_time_unit) for d in self.d_min]
        self.d_max = [int(d/self.smallest_time_unit) for d in self.d_max]
