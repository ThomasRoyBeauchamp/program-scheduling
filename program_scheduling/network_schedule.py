import logging
import os
from argparse import ArgumentParser
from itertools import chain

import numpy as np
import pandas as pd

from program_scheduling.datasets import create_dataset
from program_scheduling.session_metadata import SessionMetadata
from setup_logging import setup_logging

logger = logging.getLogger("program_scheduling")


class NetworkSchedule:
    QC_LENGTH = 380_000  # QC duration constant set by Qoala

    def __init__(self, dataset_id, n_sessions, sessions=None, start_times=None, filename=None, save=True, seed=None,
                 length_factor=3):
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
        self.dataset = create_dataset(dataset_id=dataset_id, n_sessions=n_sessions)
        self.length_factor = length_factor
        self.length = NetworkSchedule._calculate_length(self.dataset, length_factor=length_factor)
        self.id = None

        if start_times is None and sessions is None:
            if save:
                folder_path = os.path.dirname(__file__).rstrip("program_scheduling") + "network_schedules"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                self.id = self.calculate_id(folder_path)
                self.generate_random_network_schedule(seed=self.id)
                self.save_network_schedule(filename=filename)
            else:
                assert seed is not None
                self.id = seed
                self.generate_random_network_schedule(seed=self.id)
        else:
            assert len(sessions) == len(start_times)
            self.sessions = sessions
            self.start_times = start_times

    def calculate_id(self, folder_path):
        relevant_ids = []
        filename_part = NetworkSchedule.get_name(dataset_id=self.dataset_id, n_sessions=self.n_sessions,
                                                 length_factor=self.length_factor)
        for f in os.listdir(folder_path):
            if filename_part in f:
                relevant_ids.append(int(f.split(".")[0].split("-")[-1]))
        return max(relevant_ids) + 1 if len(relevant_ids) > 0 else 0

    @staticmethod
    def scale_down(network_schedule, factor):
        network_schedule.start_times = list(map(lambda t: int(t / factor), network_schedule.start_times))
        network_schedule.length = int(network_schedule.length / factor)  # TODO: is this always int?
        return network_schedule

    def save_network_schedule(self, filename):
        """
        Saves the network schedule. First checks if network_schedules folder is created.

        :param filename: Filename for saving the network schedule instance
        :param id: Index of the network schedule also used as a seed for random generation
        :return:
        """
        folder_path = os.path.dirname(__file__).rstrip("program_scheduling") + "network_schedules"
        if filename is None:
            filename = NetworkSchedule.get_name(dataset_id=self.dataset_id, n_sessions=self.n_sessions,
                                                ns_id=self.id, length_factor=self.length_factor)
        df = pd.DataFrame(data={"session": self.sessions,
                                "start_time": self.start_times})
        df.to_csv(f"{folder_path}/{filename}.csv", index=False)

    def _get_all_timeslots(self, seed):
        # to make sure no scheduled timeslot is longer than the actual network schedule
        np.random.seed(seed)
        all_timeslots = list(range(500, self.length - self.QC_LENGTH, self.QC_LENGTH))

        assigned_timeslots = []

        p = self._calculate_probabilities()

        if "qkd" in p.keys():
            parity = np.random.choice([0, 1])
            qkd_timeslots = [j for i, j in enumerate(all_timeslots) if i % 2 == parity]
            for qkd_timeslot in qkd_timeslots:
                assigned_timeslots.append((qkd_timeslot, "qkd"))
                all_timeslots.remove(qkd_timeslot)
            # remaining sessions (if any) will be scheduled in the remaining timeslots
            p.pop("qkd")

        # fill up the rest of the timeslots by the other
        if len(p.keys()) > 0:
            for start_time in all_timeslots:
                choices = list(p.keys())
                probabilities = list(p.values())
                session = np.random.choice(choices, p=probabilities)
                assigned_timeslots.append((start_time, session))

        return assigned_timeslots

    def _pick_timeslots(self, timeslots, seed):
        np.random.seed(seed)

        n_critical_sections = {k.split("/")[-1]:
                               len(set([b.CS for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks
                                        if b.type == "QC"]))
                               for (k, v) in self.dataset.items()}

        picked_timeslots = []
        for (k, v) in self.dataset.items():
            session = k.split("/")[-1]
            indices = [i for i, timeslot in enumerate(timeslots) if timeslot[1] == session]
            required_cs = n_critical_sections[session] * v
            picked_indices = np.random.choice(indices, required_cs, replace=False)
            picked_timeslots += [timeslots[i] for i in picked_indices]

        return picked_timeslots

    def _add_cs_timeslots(self, suggested_ns):
        added = []
        for (start_time, session) in suggested_ns:
            if session == "bqc":
                # bqc needs to have 2 timeslots inbetween
                t = 3 * self.QC_LENGTH
                added.append((start_time + t, session))
            if session == "pingpong":
                # pingpong needs to have 3 timeslots inbetween
                t = 4 * self.QC_LENGTH
                added.append((start_time + t, session))

        ns = suggested_ns + added
        ns.sort()
        return ns

    def feasible_network_schedule(self, suggested_ns):
        # check for length
        for (start_time, _) in suggested_ns:
            if start_time > (self.length - self.QC_LENGTH):
                return False

        # check for crit sections
        for (start_time, session) in suggested_ns:
            if session == "bqc":
                new_start_time = start_time + 3 * self.QC_LENGTH
                if new_start_time > (self.length - self.QC_LENGTH):
                    return False
                for blocked in range(start_time + self.QC_LENGTH, new_start_time + 1 + 3 * self.QC_LENGTH, self.QC_LENGTH):
                    if blocked in [t for (t, _) in suggested_ns]:
                        return False
            if session == "pingpong":
                new_start_time = start_time + 4 * self.QC_LENGTH
                if new_start_time > (self.length - self.QC_LENGTH):
                    return False
                for blocked in range(start_time + self.QC_LENGTH, new_start_time + 1 + 2 * self.QC_LENGTH, self.QC_LENGTH):
                    if blocked in [t for (t, _) in suggested_ns]:
                        return False
            if session == "qkd":
                if start_time + self.QC_LENGTH in [t for (t, _) in suggested_ns]:
                    return False

        # check for setup time (pingpong)
        for (start_time, session) in suggested_ns:
            if session == "pingpong":
                if start_time < 20000:
                    return False
                if start_time - self.QC_LENGTH in [t for (t, _) in suggested_ns]:
                    return False

        return True

    def generate_random_network_schedule(self, seed):
        """
        A method for generating a random network schedule. Depends on the probabilities that a session should be
        scheduled. Each session has a probability `p` assigned. At each decision point, a choice is randomly
        made between scheduling a timeslot for a specific session or not assigning the NS timeslot.

        :param seed: Seed for `numpy.random`
        :return:
        """
        timeslots_to_pick_from = self._get_all_timeslots(seed)
        suggested_ns = self._pick_timeslots(timeslots_to_pick_from, seed)

        while not self.feasible_network_schedule(suggested_ns):
            if seed % 100 == 0:
                logger.debug("Trying out seed", seed)
            seed += 1
            suggested_ns = self._pick_timeslots(timeslots_to_pick_from, seed)

        ns_timeslots = self._add_cs_timeslots(suggested_ns)
        self.id = seed
        logger.info(f"Generated network schedule with seed={seed}")
        for i, timeslot in enumerate(ns_timeslots):
            if i == 0:
                logger.debug(f"\t{timeslot}")
            else:
                logger.debug(f"\t{timeslot} with sep {(timeslot[0] - ns_timeslots[i - 1][0])/self.QC_LENGTH - 1} "
                             f"timeslots since the end of the execution of the previous")
        self.start_times = [start_time for (start_time, _) in ns_timeslots]
        self.sessions = [session for (_, session) in ns_timeslots]

    def _calculate_probabilities(self):
        """
        A method calculating the probabilities with which NS timeslots should be assigned to sessions. Based on the
        minimum separation time between QC blocks in a session and how many QC blocks are being scheduled.

        :return:
        """
        number_of_blocks_wo_qkd = {}
        for k, v in self.dataset.items():
            if k.split("/")[-1] != "qkd":
                # calculate number of blocks for session k
                n_blocks = sum(
                    [1 for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks if b.type == "QC"]) * v
                # assign to dictionary
                number_of_blocks_wo_qkd[k.split("/")[-1]] = n_blocks

        p = {}
        for k, v in self.dataset.items():
            session = k.split("/")[-1]
            if session != "qkd":
                probability = (number_of_blocks_wo_qkd[session] / sum(number_of_blocks_wo_qkd.values()))
                p.update({session: probability})
            elif session == "qkd":
                p.update({session: 1.0})

        return p

    @staticmethod
    def _calculate_length(dataset, length_factor=3):
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
        # TODO: how to decide on the factor
        return int(length) * length_factor

    @staticmethod
    def get_name(dataset_id, n_sessions, length_factor, ns_id=None):
        if ns_id is None:
            return f"network-schedule_sessions-{n_sessions}_dataset-{dataset_id}_length-{length_factor}"
        else:
            return f"network-schedule_sessions-{n_sessions}_dataset-{dataset_id}_length-{length_factor}_id-{ns_id}"

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
                    zip(ids, [0] * len(ids)),  # first QC blocks
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
    parser.add_argument('-seed', '--seed', required=False, type=int, default=None,
                        help="Seed for randomly generating the network schedule.")
    # length factor
    parser.add_argument('-l', '--length_factor', required=False, type=int, default=3,
                        help="Seed for randomly generating the network schedule.")
    # logging
    parser.add_argument('--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    args, unknown = parser.parse_known_args()

    setup_logging(args.loglevel)

    NetworkSchedule(dataset_id=args.dataset_id, n_sessions=args.n_sessions, save=args.save,
                    filename=args.save_schedule_filename, seed=args.seed, length_factor=args.length_factor)
