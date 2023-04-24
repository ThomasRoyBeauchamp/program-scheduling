import os

import numpy as np
import pandas as pd

import datasets
from session_metadata import SessionMetadata


class NetworkSchedule:

    QC_LENGTH = 500000  # QC duration constant set by Qoala (TODO: check if it makes sense)

    def __init__(self, dataset_id, n_sessions, sessions=None, start_times=None, filename=None, save=True, seed=42):
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

        if start_times is None and sessions is None:
            if save:
                path = os.path.dirname(__file__) + "/network_schedules"
                if not os.path.exists(path):
                    os.makedirs(path)
                if filename is None:
                    filename = f"network-schedule_{self.n_sessions}-sessions_dataset-{self.dataset_id}"
                count = len([f for f in os.listdir(path) if filename in f])
                self.generate_random_network_schedule(seed=count)
                self.save_network_schedule(filename=filename, id=count)
            else:
                self.generate_random_network_schedule(seed=seed)
        else:
            assert len(sessions) == len(start_times)
            self.sessions = sessions
            self.start_times = start_times

    def save_network_schedule(self, filename, id):
        """
        Saves the network schedule. First checks if network_schedules folder is created.

        :param filename: Filename for saving the network schedule instance
        :param id: Index of the network schedule also used as a seed for random generation
        :return:
        """
        path = os.path.dirname(__file__) + "/network_schedules"
        df = pd.DataFrame(data={"session": self.sessions,
                                "start_time": self.start_times})
        df.to_csv(f"{path}/{filename}_id-{id}.csv", index=False)

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
        for start_time in range(0, self.length - self.QC_LENGTH, self.QC_LENGTH):
            choices = list(p.keys()) + [None]
            probabilities = list(p.values()) + [1 - sum(p.values())]
            session = np.random.choice(choices, p=probabilities)
            if session is not None:
                self.sessions.append(session)
                self.start_times.append(start_time)

    def _calculate_probabilities(self):
        """
        A method calculating the probabilities with which NS timeslots should be assigned to sessions. Based on the
        minimum separation time between QC blocks in a session and how many QC blocks are being scheduled.

        :return:
        """
        # TODO: update when programs are finalised
        min_sep = {  # TODO: maybe at some point this could be read out from the yml config?
            "bqc": 20000,  # note that bob has 0, it's max of the min sep
            "pingpong": 222000,
            "qkd": 15000,
        }

        number_of_blocks = {}
        for k, v in self.dataset.items():
            # calculate number of blocks for session k
            n_blocks = sum([1 for b in SessionMetadata(k + "_alice.yml", session_id=1).blocks if b.type == "QC"]) * v
            # assign to dictionary
            number_of_blocks[k.split("/")[-1]] = n_blocks

        p = {}
        for k, v in self.dataset.items():
            session = k.split("/")[-1]
            t_sep = min_sep[session]
            rate = 1 / ((self.QC_LENGTH + t_sep) / self.QC_LENGTH)  # average number of QC slots for this session
            probability = rate * (number_of_blocks[session]/sum(number_of_blocks.values()))
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
        return int(length) * 2

    def get_session_start_times(self, session):
        try:
            self.sessions.index(session)
        except AttributeError:
            return None
        except ValueError:
            raise ValueError(f"Network schedule is incomplete and does not define a time slot for session {session}")
        start_times = []
        for i, s in enumerate(self.sessions):
            if s == session:
                start_times.append(self.start_times[i])
        return start_times

