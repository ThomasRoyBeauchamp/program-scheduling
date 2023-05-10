import logging
import os
import time

import pandas as pd
from pycsp3 import VarArray, Cumulative, satisfy, minimize, Maximum, compile, solver, ACE, \
    status, clear, SAT, OPTIMUM, UNSAT, UNKNOWN, solution
from termcolor import cprint

from program_scheduling.datasets import create_dataset
from program_scheduling.activity_metadata import ActiveSet
from program_scheduling.network_schedule import NetworkSchedule

logger = logging.getLogger("program_scheduling")


class NodeSchedule:

    def __init__(self, dataset_id, n_sessions, ns_id, role, schedule_type="HEU", save_schedule=True, save_metrics=True,
                 ns_length_factor=3, start_times=None, filename=None):
        self.dataset_id = dataset_id
        self.n_sessions = n_sessions
        self.ns_id = ns_id
        self.schedule_type = schedule_type
        self.role = role
        self.length_factor = 2 if ns_id is None else ns_length_factor

        dataset = create_dataset(dataset_id, n_sessions)
        if ns_id is not None:
            ns_name = NetworkSchedule.get_name(dataset_id=dataset_id, n_sessions=n_sessions, ns_id=ns_id,
                                               length_factor=ns_length_factor)
            folder_path = os.path.dirname(__file__).rstrip("program_scheduling") + "network_schedules"
            ns_csv = pd.read_csv(f"{folder_path}/{ns_name}.csv")

            network_schedule = NetworkSchedule(dataset_id=dataset_id, n_sessions=n_sessions,
                                               sessions=list(ns_csv["session"]),
                                               start_times=list(map(lambda x: int(x), ns_csv["start_time"])),
                                               length_factor=ns_length_factor)
            network_schedule.rewrite_sessions(dataset)
        else:
            network_schedule = None

        self.active_set = ActiveSet.create_active_set(dataset=dataset, role=role, network_schedule=network_schedule)

        if start_times is not None:
            self.start_times = start_times
            solve_time = None
        else:
            self.status, self.start_times, solve_time = self.construct_node_schedule(network_schedule=network_schedule,
                                                                                     schedule_type=schedule_type)

        self.makespan = None
        self.PUF_both = None
        self.PUF_CPU = None
        self.PUF_QPU = None

        if save_schedule and self.start_times is not None:
            if filename is None:
                filename = NodeSchedule.get_name(dataset_id=dataset_id, n_sessions=n_sessions, ns_id=ns_id,
                                                 length_factor=self.length_factor, schedule_type=schedule_type,
                                                 role=role)
            self.save_node_schedule(filename=filename)
        if save_metrics and self.start_times is not None:
            self.save_success_metrics(solve_time=solve_time)

    @staticmethod
    def get_name(dataset_id, n_sessions, schedule_type, length_factor=None, ns_id=None, role=None):
        if length_factor is None and ns_id is None and role is None:
            return f"node-schedule_sessions-{n_sessions}_dataset-{dataset_id}_schedule-{schedule_type}"
        else:
            return f"node-schedule_sessions-{n_sessions}_dataset-{dataset_id}_schedule-{schedule_type}_" \
               f"length-{length_factor}_NS-{ns_id}_role-{role}"

    @staticmethod
    def get_relevant_node_schedule_names(dataset_id, n_sessions, schedule_type):
        relevant_node_schedule_names = []
        folder_path = os.path.dirname(__file__).rstrip("program_scheduling") + "node_schedules"
        filename_part = NodeSchedule.get_name(dataset_id=dataset_id, n_sessions=n_sessions, schedule_type=schedule_type)
        for f in os.listdir(folder_path):
            if filename_part in f:
                relevant_node_schedule_names.append(f[:-8] if "bob" in f else f[:-10])
        unique = list(set(relevant_node_schedule_names))
        unique.sort()
        return unique

    def construct_node_schedule(self, network_schedule, schedule_type):
        scaled_durations, scaled_d_max = self.active_set.scale_down()
        if network_schedule is not None:
            scaled_network_schedule = NetworkSchedule.scale_down(network_schedule, self.active_set.get_gcd())

        # in case a session with post-processing is scheduled right at the end of NS (currently dominated by QKD)
        extra_margin = int(35_000_000 / self.active_set.get_gcd())
        schedule_size = scaled_network_schedule.length + extra_margin if network_schedule is not None \
            else self.length_factor * int(sum(scaled_durations))
        logger.debug(f"Length of network schedule is {schedule_size}")
        capacities = [1, 1]  # capacity of [CPU, QPU]

        # x[i] is the starting time of the ith job
        x = VarArray(size=self.active_set.n_blocks, dom=range(schedule_size))

        # taken from http://pycsp.org/documentation/models/COP/RCPSP/
        def cumulative_for(k):
            # TODO: this doesn't work if session is purely quantum or purely classical
            origins, lengths, heights = zip(*[(x[i], scaled_durations[i], self.active_set.resource_reqs[i][k])
                                              for i in range(self.active_set.n_blocks) if
                                              self.active_set.resource_reqs[i][k] > 0])
            return Cumulative(origins=origins, lengths=lengths, heights=heights)

        def get_CS_indices():
            forward = list(zip(self.active_set.ids, self.active_set.cs_ids))
            backwards = list(zip(self.active_set.ids, self.active_set.cs_ids))
            backwards.reverse()

            CS_indices = []
            for (session_id, cs_id) in set(zip(self.active_set.ids, self.active_set.cs_ids)):
                if cs_id is not None:
                    CS_indices.append((session_id, cs_id, forward.index((session_id, cs_id)),
                                       len(forward) - backwards.index((session_id, cs_id)) - 1))

            return CS_indices

        # constraints
        satisfy(
            # precedence constraints
            [x[i] + scaled_durations[i] <= x[j] for i in range(self.active_set.n_blocks) for j in
             self.active_set.successors[i]],
            # resource constraints
            [cumulative_for(k) <= capacity for k, capacity in enumerate(capacities)],
            # constraints for max time lags
            [(x[i + 1] - (x[i] + scaled_durations[i])) <= scaled_d_max[i + 1] for i in
             range(self.active_set.n_blocks - 1)
             if (self.active_set.types[i + 1] != "QC" or scaled_d_max[i] is None) and scaled_d_max[i + 1] is not None],
            [(x[i] < x[start]) | (x[end] < x[i]) for (session_id, cs_id, start, end) in get_CS_indices()
             for i in range(self.active_set.n_blocks - 1) if
             (self.active_set.ids[i] != session_id and self.active_set.cs_ids[i] != cs_id)]
        )

        def get_QC_indices(without=None):
            indices = [i for i in range(0, self.active_set.n_blocks - 1) if self.active_set.types[i] == "QC"]
            if without is not None:
                for remove in without:
                    indices.remove(remove)
            return indices

        if network_schedule is not None:
            satisfy(
                [x[i] in set(scaled_network_schedule.get_session_start_times(self.active_set.ids[i])) for i in
                 get_QC_indices()],
                # order of a qc block is correct
                [x[i] in set(scaled_network_schedule.get_qc_block_start_times(self.active_set.qc_indices[i])) for i in
                 get_QC_indices()]
            )
        else:
            satisfy(
                [(x[i + 1] - (x[i] + scaled_durations[i])) <= scaled_d_max[i + 1] for i in
                 range(self.active_set.n_blocks - 1)
                 if self.active_set.types[i + 1] == "QC" and scaled_d_max[i + 1] is not None]
            )

        if schedule_type == "NAIVE":
            satisfy(
                [x[i] < x[i + 1] for i in range(self.active_set.n_blocks - 1)],
            )

        # optional objective function
        if schedule_type == "OPT":
            minimize(
                Maximum([x[i] + scaled_durations[i] for i in range(self.active_set.n_blocks)])
            )

        instance = compile()
        ace = solver(ACE)

        # https://github.com/xcsp3team/pycsp3/blob/master/docs/optionsSolvers.pdf
        # here you can possibly define other heuristics to use
        heuristics = {}

        start = time.time()
        result = ace.solve(instance, dict_options=heuristics)
        end = time.time()

        stat = None
        start_times = None
        solve_time = end - start

        if status() is SAT or status() is OPTIMUM:
            start_times = [s * self.active_set.get_gcd() for s in solution().values]
            stat = "SAT"

            logger.info(f"Found node schedule for {self.role} with {self.n_sessions} sessions of "
                        f"dataset {self.dataset_id} in {round(solve_time, 2)} seconds.")

            # remove PyCSP log files
            for filename in os.listdir():
                if filename.endswith(".log") or filename.endswith(".xml"):
                    os.remove(filename)

        elif status() is UNKNOWN:
            stat = "UNKNOWN"
            logger.info("The solver cannot find a solution. (The problem might be too large.) "
                        "Time taken to finish: %.4f seconds" % solve_time)
        elif status() is UNSAT:
            stat = "UNSAT"
            logger.info("No feasible node schedule can be found. Time taken to finish: %.4f seconds" % solve_time)
        else:
            logger.info("Something else went wrong. Time taken to finish: %.4f seconds" % solve_time)

        clear()
        return stat, start_times, solve_time

    def save_success_metrics(self, solve_time):
        # there is one results file for each combination of n_sessions and schedule_type
        filename = f"static-results-node-schedule_sessions-{self.n_sessions}_schedule-{self.schedule_type}"
        path = os.path.dirname(__file__).rstrip("program_scheduling") + "/results"
        # create a pandas dataframe
        dataset = create_dataset(self.dataset_id, self.n_sessions, only_session_name=True)
        metadata = {
            "dataset_id": self.dataset_id,
            "n_sessions": self.n_sessions,
            "ns_id": self.ns_id,
            "length_factor": self.length_factor,
            "schedule_type": self.schedule_type,
            "role": self.role,
            "bqc_sessions": dataset.get("bqc", 0),
            "pingpong_sessions": dataset.get("pingpong", 0),
            "qkd_sessions": dataset.get("qkd", 0),
            "solve_time": solve_time
        }
        success_metrics = {
            "makespan": self.get_makespan(),
            "PUF_both": self.get_PUF_both(),
            "PUF_CPU": self.get_PUF_CPU(),
            "PUF_QPU": self.get_PUF_QPU()
        }
        df = pd.DataFrame(data={**metadata, **success_metrics},
                          columns=list(metadata.keys()) + list(success_metrics.keys()), index=[0])

        if os.path.isfile(f"{path}/{filename}.csv"):  # if the file exists, append
            old_df = pd.read_csv(f"{path}/{filename}.csv")
            new_df = pd.concat([old_df, df], ignore_index=True)
            new_df.to_csv(f"{path}/{filename}.csv", index=False)
        else:  # otherwise make new file
            df.to_csv(f"{path}/{filename}.csv", index=False)

    def print(self):
        cprint(f"Success metrics: makespan={self.get_makespan()}, PUF_CPU={round(self.get_PUF_CPU() * 100, 2)}%, "
               f"PUF_QPU={round(self.get_PUF_QPU() * 100, 2)}%, PUF_both={round(self.get_PUF_both() * 100, 2)}% \n",
               "magenta")
        cprint("CPU schedule:", "light_yellow")
        c_ops = [b for b in zip(self.start_times, self.active_set.block_names, self.active_set.types, self.active_set.durations) if b[2][0] == "C"]
        c_ops.sort()
        for b in c_ops:
            cprint(f"\tt={b[0]}: {b[1]} ({b[2]}) -- (duration = {b[3]} -> end time = {b[0] + b[3]})", "light_yellow")
        cprint("QPU schedule:", "light_blue")
        q_ops = [b for b in zip(self.start_times, self.active_set.block_names, self.active_set.types, self.active_set.durations) if b[2][0] == "Q"]
        q_ops.sort()
        for b in q_ops:
            cprint(f"\tt={b[0]}: {b[1]} ({b[2]}) -- (duration = {b[3]} -> end time = {b[0] + b[3]})", "light_blue")

    def save_node_schedule(self, filename):
        path = os.path.dirname(__file__).rstrip("program_scheduling") + "/node_schedules"
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.isfile(f"{path}/{filename}.csv"):
            logger.warning("An older node schedule is being overwritten.")
        df = pd.DataFrame(data={"index": list(range(self.active_set.n_blocks)),
                                "type": self.active_set.types,
                                "start_time": self.start_times,
                                "duration": self.active_set.durations})
        df.to_csv(f"{path}/{filename}.csv", index=False)

    def _calculate_activities(self, resource_index):
        activities = [-1] * self.get_makespan()
        for activity, start_time in enumerate(self.start_times):
            if self.active_set.resource_reqs[activity][resource_index] == 1:
                if self.active_set.durations[activity] == 1:
                    activities[start_time] = activity
                else:
                    end_time = start_time + self.active_set.durations[activity]
                    activities[start_time:end_time] = [activity] * self.active_set.durations[activity]
        return activities

    def get_makespan(self):
        if self.makespan is None:
            makespan = -1
            for activity, start_time in enumerate(self.start_times):
                end_time = start_time + self.active_set.durations[activity]
                if end_time > makespan:
                    makespan = end_time
            self.makespan = makespan
        return self.makespan

    def get_PUF_CPU(self):
        if self.PUF_CPU is None:
            CPU_duration = sum([self.active_set.durations[i] for i in range(self.active_set.n_blocks)
                                if self.active_set.types[i][0] == "C"])
            self.PUF_CPU = CPU_duration / self.get_makespan()
        return self.PUF_CPU

    def get_PUF_QPU(self):
        if self.PUF_QPU is None:
            QPU_duration = sum([self.active_set.durations[i] for i in range(self.active_set.n_blocks)
                                if self.active_set.types[i][0] == "Q"])
            self.PUF_QPU = QPU_duration / self.get_makespan()
        return self.PUF_QPU

    def get_PUF_both(self):
        if self.PUF_both is None:
            total_duration = 0
            pairs = list(zip(self.start_times, self.active_set.durations))
            pairs.sort()
            c = 0
            for (start, duration) in pairs:
                if c <= start:
                    c += duration
                    total_duration += duration
                else:
                    end_time = start + duration
                    if end_time > c:
                        total_duration = total_duration + end_time - c
                        c = end_time
            self.PUF_both = total_duration / self.get_makespan()
        return self.PUF_both
