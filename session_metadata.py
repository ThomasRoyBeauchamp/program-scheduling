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

        if session_id is not None:
            config["session_id"] = session_id

        # TODO: check required params are present and

        self.session_id = config.get("session_id")
        self.app_deadline = config.get("app_deadline")

        self.T1 = config.get("T1")
        self.T2 = config.get("T2")
        self.gate_duration = config.get("gate_duration")
        self.gate_fidelity = config.get("gate_fidelity")
        self.cc_duration = config.get("cc_duration")

        # note that each block is a nested dictionary with an arbitrary block name
        # TODO: try to figure out if there's a better way to do this
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
