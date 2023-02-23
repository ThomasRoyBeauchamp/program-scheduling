import yaml


class BlockMetadata:

    def __init__(self, config):
        for key in ["comm_q", "storage_q", "instructions", "CS_id"]:
            if key not in config.keys():
                raise ValueError(f"Key {key} is not defined in block configuration")
        if len(config.get("instructions")) != 4:
            raise ValueError(f"Four types of instructions should be defined, but there were "
                             f"{len(config.get('instructions'))}")

        self.comm_q = config.get("comm_q")
        self.storage_q = config.get("storage_q")
        self.instructions = config.get("instructions")
        self.CS_id = config.get("CS_id")

    def __str__(self):
        return f"(#Q_C = {self.comm_q}, #Q_S = {self.storage_q}, " \
               f"#I_QC = {self.instructions[0]}, #I_QL = {self.instructions[1]}, " \
               f"#I_CC = {self.instructions[2]}, #I_CL = {self.instructions[3]}, " \
               f"critical section ID = {self.CS_id})"


class SessionMetadata:

    def __init__(self, yaml_file, session_id=None):
        with open(yaml_file, 'r') as file_handle:
            config = yaml.load(file_handle, yaml.SafeLoader) or {}

        for key in ["session_id", "app_deadline", "blocks"]:
            if key not in config.keys():
                raise ValueError(f"Key {key} is not defined in session configuration")

        default_params = {"T1": None, "T2": None, "gate_duration": 1, "gate_fidelity": 1.0, "cc_duration": 1}
        for key in ["T1", "T2", "gate_duration", "gate_fidelity", "cc_duration"]:
            if key not in config.keys():
                print(f"Value for {key} is not defined, it will be set to its default value ({default_params[key]})")

        if session_id is not None:
            config["session_id"] = session_id

        self.session_id = config.get("session_id")
        self.app_deadline = config.get("app_deadline")

        self.T1 = config.get("T1", default_params["T1"])
        self.T2 = config.get("T2", default_params["T2"])
        self.gate_duration = config.get("gate_duration", default_params["gate_duration"])
        self.gate_fidelity = config.get("gate_fidelity", default_params["gate_fidelity"])
        self.cc_duration = config.get("cc_duration", default_params["cc_duration"])

        # note that each block is a nested dictionary with an arbitrary block name
        self.blocks = [BlockMetadata(block_config.get(list(block_config.keys())[0]))
                       for block_config in config.get("blocks")]

    def __str__(self):
        s = f"Session {self.session_id}:" \
            f"\n\tParameters:" \
            f"\n\t\tDeadline = {self.app_deadline}" \
            f"\n\t\t[T1, T2] = [{self.T1}, {self.T2}]" \
            f"\n\t\tGate duration = {self.gate_duration}ns, Gate fidelity = {self.gate_fidelity}ns" \
            f"\n\t\tclassical communication duration = {self.cc_duration}ns" \
            f"\n\tBlocks:"
        for b in self.blocks:
            s += "\n\t\t" + str(b)
        return s


if __name__ == '__main__':
    sm = SessionMetadata("qkd.yaml")
    print(sm)
