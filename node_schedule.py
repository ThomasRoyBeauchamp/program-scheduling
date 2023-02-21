from termcolor import cprint


class NodeSchedule:

    def __init__(self, n_activities, start_times, durations, resources):
        self.n_activities = n_activities
        self.start_times = start_times
        self.durations = durations
        self.resources = resources

        self.makespan = None
        self.PUF_both = None
        self.PUF_CPU = None
        self.PUF_QPU = None

        self._CPU_activities = self._calculate_activities(0)
        self._QPU_activities = self._calculate_activities(1)

    def print(self):
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
            self.PUF_both = (self.get_makespan() - (temp_cpu or temp_qpu).count(False)) / self.get_makespan()
        return self.PUF_both
