from session_metadata import SessionMetadata, BlockMetadata


class ActivityMetadata:

    def __init__(self, session_metadata: SessionMetadata):
        self.session_id = session_metadata.session_id
        self.app_deadline = session_metadata.app_deadline

        self.n_blocks = len(session_metadata.blocks)
        self.successors = [[i + 1] for i in range(self.n_blocks - 1)]
        self.successors.append([])
        self.resource_reqs = [self._calculate_resource_reqs(b) for b in session_metadata.blocks]

        self.durations = [self._calculate_duration(b, session_metadata.gate_duration, session_metadata.cc_duration)
                          for b in session_metadata.blocks]
        self.d_min, self.d_max = self._calculate_time_lags(session_metadata)

    @staticmethod
    def _calculate_resource_reqs(block_metadata: BlockMetadata):
        # resource requirements should be defined as [CPU, QPU]
        classical = block_metadata.comm_q == 0 and block_metadata.storage_q == 0 and\
                    (block_metadata.instructions[2] > 0 or block_metadata.instructions[3] > 0)
        quantum = (block_metadata.comm_q > 0 or block_metadata.storage_q > 0) and\
                  (block_metadata.instructions[0] > 0 or block_metadata.instructions[1] > 0)
        # sanity check that block is either fully classical or fully quantum
        assert (classical and not quantum) or (not classical and quantum)
        return [int(classical), int(quantum)]

    @staticmethod
    def _calculate_duration(block_metadata: BlockMetadata, gate_duration, cc_duration):
        cl_duration = 1  # constant for a classical local instruction, TODO: what should it be?
        qc = block_metadata.instructions[0]  # TODO: retrieve from network schedule???
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
            if block.CS_id is None or i == 0:
                d_max.append(remaining_time)
            else:
                d_max.append(0 if session_metadata.blocks[i-1].CS_id == block.CS_id else remaining_time)

        assert len(d_max) == self.n_blocks  # sanity check
        return [0] * self.n_blocks, d_max


class ActiveSet:

    def __init__(self):
        self.n_blocks = 0
        self.successors = []
        self.resource_reqs = []
        self.durations = []
        self.d_min = []
        self.d_max = []

    @staticmethod
    def create_active_set(configs: [ActivityMetadata], session_ids):
        if len(configs) != len(session_ids):
            raise ValueError(f"Session IDs for all activities need to be defined. Currently, there are {len(configs)} "
                             f"different config files set but {len(session_ids)} different sets of IDs defined.")

        active = ActiveSet()
        print(f"Your active set now has the following sessions:")
        for i, ids in enumerate(session_ids):
            print(f"\t{configs[i]} -- {len(ids)} sessions with IDs {ids}")
            for ID in ids:
                session_metadata = SessionMetadata(yaml_file=configs[i], session_id=ID)
                active._merge_activity_metadata(ActivityMetadata(session_metadata))

        return active

    def _merge_activity_metadata(self, other: ActivityMetadata):
        if self.n_blocks == 0:  # we don't need to reindex successors
            self.successors += other.successors
        else:
            new_successors = [x if len(x) == 0 else [x[0] + self.n_blocks] for x in other.successors]
            self.successors += new_successors
        self.n_blocks += other.n_blocks
        self.resource_reqs += other.resource_reqs
        self.durations += other.durations
        self.d_min += other.d_min
        self.d_max += other.d_max
