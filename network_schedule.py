import os
from argparse import ArgumentParser
from itertools import chain

import numpy as np
import pandas as pd

import datasets
from session_metadata import SessionMetadata


class NetworkSchedule:

    QC_LENGTH = 380_000  # QC duration constant set by Qoala

    def __init__(self, dataset_id, n_sessions, sessions=None, start_times=None, filename=None, save=False, seed=42):
        """
        Network schedule class. Dataset ID and total number of sessions must be given. If sessions and start_times
        are not define, a network schedule will be randomly generated. Filename for saving the network schedule
        can be set for testing purposes.

        :param dataset_id: ID of the dataset specifying a combination of applications
        :param n_sessions: Total number of sessions to be scheduled
        :param sessions: List of sessions for which QC timeslots are scheduled
        :param start_times: Start times of QC timeslots being scheduled
        :param filename: Optional filename for saving the network schedule
        """
        self.dataset_id = dataset_id
        self.n_sessions = n_sessions
        self.dataset = datasets.create_dataset(id=dataset_id, n_sessions=n_sessions)
        self.length = NetworkSchedule._calculate_length(self.dataset)
        self.id = None

        if start_times is None and sessions is None:
            if save:
                path = "network_schedules"
                if not os.path.exists(path):
                    os.makedirs(path)
                if filename is None:
                    filename = f"network-schedule_{self.n_sessions}-sessions_dataset-{self.dataset_id}"
                self.id = (max([int(f.split(".")[0].split("-")[-1]) for f in os.listdir(path) if filename in f]) + 1) \
                    if len([f for f in os.listdir(path) if filename in f]) > 0 else 0
                self.generate_random_network_schedule(seed=self.id)
                self.save_network_schedule(filename=filename)
            else:
                self.id = seed
                self.generate_random_network_schedule(seed=self.id)
        else:
            assert len(sessions) == len(start_times)
            self.sessions = sessions
            self.start_times = start_times

    def save_network_schedule(self, filename):
        """
        Saves the network schedule. First checks if network_schedules folder is created.

        :param filename: Filename for saving the network schedule instance
        :param id: Index of the network schedule also used as a seed for random generation
        :return:
        """
        path = "network_schedules"
        df = pd.DataFrame(data={"session": self.sessions,
                                "start_time": self.start_times})
        df.to_csv(f"{path}/{filename}_id-{self.id}.csv", index=False)

    def generate_random_network_schedule(self, seed):
        """
        A method for generating a random network schedule. Depends on the probabilities that a session should be
        scheduled. Each session has a probability `p` assigned. At each decision point, a choice is randomly
        made between scheduling a timeslot for a specific session or not assigning the NS timeslot.

        :param seed: Seed for `numpy.random`
        :return:
        """
        np.random.seed(seed)

        p = self._calculate_probabilities()
        self.sessions = []
        self.start_times = []

        cs = {k.split("/")[-1]:
                  len(set([b.CS for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks if b.type == "QC"]))
              for (k, v) in self.dataset.items()}

        print(set([b.CS for b in SessionMetadata("configs/qkd_alice.yml", session_id=1).blocks if b.type == "QC"]))
        print("critical sections ", cs)

        min_sep = {  # TODO: maybe at some point this could be read out from the yml config?
            "bqc": 607000,  # note that bob has 0, it's max of the min sep
            "pingpong": 807000,
            "qkd": 15000,
        }

        # assign all timeslots
        for start_time in range(500, self.length - self.QC_LENGTH, self.QC_LENGTH):
            choices = list(p.keys())
            probabilities = list(p.values())
            session = np.random.choice(choices, p=probabilities)
            self.sessions.append(session)
            self.start_times.append(start_time)

        # pick a subset
        start = True
        conflicts = False
        # check for constraints. if they don't work, pick a different subset. how to do this with seeds?
        while conflicts or start:
            start = False
            conflicts = False
            picked_timeslots = {}
            for (k, v) in self.dataset.items():
                session = k.split("/")[-1]
                indices = [i for i in range(len(self.sessions)) if self.sessions[i] == session]
                required_cs = cs[session] * v
                picked_timeslots.update({session: list(np.random.choice(indices, required_cs, replace=False))})

            if "qkd" in picked_timeslots.keys():
                for i in picked_timeslots["qkd"]:
                    if i - 1 in picked_timeslots["qkd"]:
                        conflicts = True
                        self.id += 1
                        np.random.seed(self.id)
                        # todo: can you `pass` here

            # assign remaining QC blocks
            picked_start_times = []
            picked_sessions = []
            for (k, l) in picked_timeslots.items():
                for i in l:
                    picked_start_times.append(self.start_times[i])
                    picked_sessions.append(k)

            # adding the other QC block in the critical section
            for (k, l) in picked_timeslots.items():
                for i in l:
                    if k == "bqc" or k == "pingpong":
                        new_start_time = self.start_times[i] + self.QC_LENGTH + (0 if min_sep[k] == 0
                                                                                 else (int(min_sep[k] / self.QC_LENGTH) + 1) * self.QC_LENGTH)
                        # TODO: also check that you're not putting it after the end of the network schedule
                        for overlapping in range(self.start_times[i] + self.QC_LENGTH, new_start_time + 1, self.QC_LENGTH):
                            if overlapping in picked_start_times:
                                conflicts = True
                                self.id += 1
                                np.random.seed(self.id)
                        picked_start_times.append(new_start_time)
                        picked_sessions.append(k)

        zipped = list(zip(picked_start_times, picked_sessions))
        zipped.sort()
        self.start_times = [s for (s, _) in zipped]
        self.sessions = [s for (_, s) in zipped]

    def _calculate_probabilities(self):
        """
        A method calculating the probabilities with which NS timeslots should be assigned to sessions. Based on the
        minimum separation time between QC blocks in a session and how many QC blocks are being scheduled.

        :return:
        """
        number_of_blocks = {}
        for k, v in self.dataset.items():
            # calculate number of blocks for session k
            n_blocks = sum([1 for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks if b.type == "QC"]) * v
            # assign to dictionary
            number_of_blocks[k.split("/")[-1]] = n_blocks

        p = {}
        for k, v in self.dataset.items():
            session = k.split("/")[-1]
            probability = (number_of_blocks[session] / sum(number_of_blocks.values()))
            p.update({session: probability})
        return p

    @staticmethod
    def _calculate_length(dataset):
        """
        A method calculating the total length of a network schedule. This depends on the session types and number
        of sessions being scheduled. For a session between Alice and Bob, we consider the total length of a program
        and take the longer one. This is then multiplied by however many sessions are being scheduled. Finally,
        we add a factor of 2 to allow for some delays in the node schedule due to network schedule constraints.

        :param dataset: Dataset for which this network schedule is being constructed
        :return:
        """
        length = 0
        for k, v in dataset.items():
            total_alice = sum([b.duration for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks])
            total_bob = sum([b.duration for b in SessionMetadata(k + "_bob.yml", session_id=1).blocks])
            length += (max(total_alice, total_bob) * v)
        print("Total length for dataset is ", length)
        return int(length) * 2

    def rewrite_sessions(self, dataset):
        # TODO: make this prettier
        session_ids = {}
        last_session_id = 0
        for (config_file, number) in dataset.items():
            ids = list(range(last_session_id, last_session_id + number))
            session_ids.update({config_file.split("/")[-1]: ids})
            last_session_id += number

        new_d = {}
        for (session, ids) in session_ids.items():
            if session == "qkd":
                # yes you're going to regret this
                new_d.update({session: list(chain(*zip(
                    zip(ids, [0] * len(ids)),
                    zip(ids, [1] * len(ids)),
                    zip(ids, [2] * len(ids)),
                    zip(ids, [3] * len(ids)),
                    zip(ids, [4] * len(ids))
                ))) * 10})
            else:
                new_d.update({session: list(chain(*zip(
                    zip(ids, [0] * len(ids)), # first QC blocks
                    zip(ids, [1] * len(ids))  # second QC blocks
                ))) * 10})

        rewrite = []
        for sesh in self.sessions:
            rewrite.append(new_d[sesh].pop(0))

        self.sessions = rewrite

    def get_session_start_times(self, session):
        start_times = []
        for i, (s, _) in enumerate(self.sessions):
            if s == session:
                start_times.append(self.start_times[i])
        if len(start_times) > 0:
            return start_times
        else:
            return None

    def get_qc_block_start_times(self, qc_index):
        start_times = []
        for i, (_, qc_i) in enumerate(self.sessions):
            if qc_i == qc_index:
                start_times.append(self.start_times[i])
        if len(start_times) > 0:
            return start_times
        else:
            return None


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument('-d', '--dataset-id', required=True, type=int,
                        help="Dataset of sessions to schedule.")
    # number of sessions in a dataset
    parser.add_argument('-s', '--n_sessions', required=True, type=int,
                        help="Total number of sessions in a dataset.")
    # T/F save schedule
    parser.add_argument('-save', "--save_schedule", dest="save", default=False, action="store_true",
                        help="Save the network schedule.")
    # filename for saving schedule
    parser.add_argument('-ssf', '--save_schedule_filename', required=False, type=str, default=None,
                        help="Filename for saving the schedule.")
    # seed
    parser.add_argument('-seed', '--seed', required=False, type=int, default=42,
                        help="Seed for randomly generating the network schedule.")
    args, unknown = parser.parse_known_args()

    NetworkSchedule(dataset_id=args.dataset_id, n_sessions=args.n_sessions, save=args.save,
                    filename=args.save_schedule_filename, seed=args.seed)
