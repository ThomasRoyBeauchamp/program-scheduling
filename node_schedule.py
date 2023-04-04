from termcolor import cprint
import numpy as np
import pandas as pd
import yaml
import os


class NodeSchedule:

    def __init__(self, active_set, start_times):
        self.n_activities = active_set.n_blocks
        self.start_times = start_times
        self.durations = active_set.durations
        self.resources = active_set.resource_reqs
        self.types = active_set.types
        self.block_names = active_set.block_names

        self.makespan = None
        self.PUF_both = None
        self.PUF_CPU = None
        self.PUF_QPU = None

        self._CPU_activities = self._calculate_activities(0)
        self._QPU_activities = self._calculate_activities(1)

    @staticmethod
    def how_many_sessions(dataset, protocol):
        f = list(filter(lambda x: protocol in x[0], list(dataset.items())))
        return sum([n for (_, n) in f])

    @staticmethod
    def get_role(dataset):
        f = list(filter(lambda x: x[1] > 0, list(dataset.items())))
        return "Alice" if "alice" in f[0][0] else "Bob"

    def save_success_metrics(self, name, filename, network_schedule, dataset):
        # create a pandas dataframe
        ns = False if network_schedule is None or not network_schedule.is_defined else True
        qkd = self.how_many_sessions(dataset, "qkd")
        bqc = self.how_many_sessions(dataset, "bqc")
        pp = self.how_many_sessions(dataset, "pingpong")
        df = pd.DataFrame([[name, self.get_role(dataset), ns, qkd, bqc, pp, self.get_makespan(),
                            self.get_PUF_both(), self.get_PUF_CPU(), self.get_PUF_QPU()]],
                          columns=["name", "node", "network_schedule", "# QKD", "# BQC", "# PP", "makespan",
                                   "PUF_both", "PUF_CPU", "PUF_QPU"])

        if os.path.isfile(filename):  # if the file exists, append
            old_df = pd.read_csv(filename)
            new_df = pd.concat([old_df, df], ignore_index=True)
            new_df.to_csv(filename, index=False)
        else:  # otherwise make new file
            df.to_csv(filename, index=False)

    def print(self):
        if self.get_makespan() < 40:
            CPU_str = self._prep_PU_for_print(self._CPU_activities)
            QPU_str = self._prep_PU_for_print(self._QPU_activities)
            timeline = "".join(str(t) + "   " if t < 10 else str(t) + "  " for t in range(self.get_makespan()))

            cprint("\nNode schedule:", "light_magenta")
            cprint(f"Success metrics: makespan={self.get_makespan()}, PUF_CPU={round(self.get_PUF_CPU() * 100, 2)}%, "
                   f"PUF_QPU={round(self.get_PUF_QPU() * 100, 2)}%, PUF_both={round(self.get_PUF_both() * 100, 2)}% \n",
                   "magenta")
            cprint("CPU: " + CPU_str, "light_yellow")
            cprint("QPU: " + QPU_str, "light_blue")
            cprint("     " + "|   " * self.get_makespan(), "dark_grey")
            cprint("     " + timeline, "dark_grey")

        else:
            cprint(f"Success metrics: makespan={self.get_makespan()}, PUF_CPU={round(self.get_PUF_CPU() * 100, 2)}%, "
                   f"PUF_QPU={round(self.get_PUF_QPU() * 100, 2)}%, PUF_both={round(self.get_PUF_both() * 100, 2)}% \n",
                   "magenta")
            cprint("CPU schedule:", "light_yellow")
            c_ops = [b for b in zip(self.start_times, self.block_names, self.types, self.durations) if b[2][0] == "C"]
            c_ops.sort()
            for b in c_ops:
                cprint(f"\tt={b[0]}: {b[1]} ({b[2]}) -- (duration = {b[3]} -> end time = {b[0] + b[3]})", "light_yellow")
            cprint("QPU schedule:", "light_blue")
            q_ops = [b for b in zip(self.start_times, self.block_names, self.types, self.durations) if b[2][0] == "Q"]
            q_ops.sort()
            for b in q_ops:
                cprint(f"\tt={b[0]}: {b[1]} ({b[2]}) -- (duration = {b[3]} -> end time = {b[0] + b[3]})", "light_blue")

    def save_starting_times(self, filename):
        # TODO: check that node_schedules folder exists and if not, create it
        with open('../node_schedules/' + filename + '.yml', 'w+') as outfile:
            yaml.dump([t * self.stu for t in self.start_times], outfile, default_flow_style=False, sort_keys=False)

    def save_sorted_indices(self, filename):
        # TODO: check that node_schedules folder exists and if not, create it
        pairs = [(t * self.stu, i) for i, t in enumerate(self.start_times)]
        pairs.sort()
        with open('../node_schedules/' + filename + '.yml', 'w+') as outfile:
            yaml.dump([i for (_, i) in pairs], outfile, default_flow_style=False, sort_keys=False)

    def _prep_PU_for_print(self, activities):
        temp = ""
        for i, a in enumerate(activities):
            if a == -1:
                temp += "----"
            else:
                if self.durations[a] == 1:
                    temp += "[" + ("0" + str(a) if a < 10 else str(a)) + "]"
                else:
                    if activities.index(a) < i:
                        continue
                    d = (self.durations[a] - 1) * 2
                    temp += "[" + " " * d + ("0" + str(a) if a < 10 else str(a)) + " " * d + "]"
        return temp

    def _calculate_activities(self, resource_index):
        activities = [-1] * self.get_makespan()
        for activity, start_time in enumerate(self.start_times):
            if self.resources[activity][resource_index] == 1:
                if self.durations[activity] == 1:
                    activities[start_time] = activity
                else:
                    end_time = start_time + self.durations[activity]
                    activities[start_time:end_time] = [activity] * self.durations[activity]
        return activities

    def get_makespan(self):
        if self.makespan is None:
            makespan = -1
            for activity, start_time in enumerate(self.start_times):
                end_time = start_time + self.durations[activity]
                if end_time > makespan:
                    makespan = end_time
            self.makespan = makespan
        return self.makespan

    def get_PUF_CPU(self):
        if self.PUF_CPU is None:
            self.PUF_CPU = (self.get_makespan() - self._CPU_activities.count(-1)) / self.get_makespan()
        return self.PUF_CPU

    def get_PUF_QPU(self):
        if self.PUF_QPU is None:
            self.PUF_QPU = (self.get_makespan() - self._QPU_activities.count(-1)) / self.get_makespan()
        return self.PUF_QPU

    def get_PUF_both(self):
        if self.PUF_both is None:
            temp_cpu = [False if i == -1 else True for i in self._CPU_activities]
            temp_qpu = [False if i == -1 else True for i in self._QPU_activities]
            self.PUF_both = (self.get_makespan() - list(np.logical_or(temp_cpu, temp_qpu)).count(False)) \
                / self.get_makespan()
        return self.PUF_both
