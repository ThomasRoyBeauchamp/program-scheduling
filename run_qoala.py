# from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns
import pandas as pd
from netsquid.qubits import ketstates
from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import LatenciesConfig, ProcNodeConfig, ProcNodeNetworkConfig, TopologyConfig
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.schedule import TaskSchedule, TaskScheduleEntry
from qoala.sim.build import build_network
from qoala.sim.procnode import ProcNode
from qoala.util.math import has_state

from datasets import create_dataset


def create_network_info(names: List[str]) -> NetworkInfo:
    env = NetworkInfo.with_nodes({i: name for i, name in enumerate(names)})
    env.set_global_schedule([0, 1, 2])
    env.set_timeslot_len(1e6)
    return env


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=100_000, qnos_instr_time=1000
        ),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def create_batch(
    program: QoalaProgram,
    unit_module: UnitModule,
    inputs: List[ProgramInput],
    num_iterations: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        unit_module=unit_module,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
    )


def create_task_schedule(tasks, node_schedule_config):
    """
    To fix the length of QC blocks, this method can in the future also take in network schedule (or have the relevant
    information in the node schedule) and return the changed tasks as well.
    """
    node_schedule = pd.read_csv(node_schedule_config)

    end_times = [node_schedule["start_time"][i] + node_schedule["duration"][i] for i in range(len(node_schedule["index"]))]
    temp = []
    for i in range(len(node_schedule["index"])):
        if end_times.count(node_schedule["start_time"][i]) == 1:
            prev_index = end_times.index(node_schedule["start_time"][i])
            if node_schedule["type"][i][0] != node_schedule["type"][prev_index][0]:
                temp.append((node_schedule["start_time"][i], node_schedule["index"][i], prev_index))
            else:
                # there is only one task which finishes before this one but it's the same type
                temp.append((node_schedule["start_time"][i], node_schedule["index"][i], None))
        else:
            # there is no prev task
            temp.append((node_schedule["start_time"][i], node_schedule["index"][i], None))
    # sort tuples according to start time
    temp.sort()

    # indices of tasks in the order they start in the node schedule
    indices = [i for (_, i, _) in temp]
    # prev[i] is an index of tasks that needs to be defined as previous to task i
    prev = [p for (_, _, p) in temp]

    schedule = TaskSchedule([
        TaskScheduleEntry(tasks[indices[i]], prev=tasks[prev[i]] if prev[i] is not None else None)
        for i in range(len(tasks))])

    return schedule


def execute_node_schedule(dataset, node_schedule_name, **kwargs):
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["alice", "bob"])
    alice_id = network_info.get_node_id("alice")
    bob_id = network_info.get_node_id("bob")

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=380_000
    )
    network = build_network(network_cfg, network_info)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    for (path, num_iterations) in dataset.items():
        alice_program = load_program(path + "_alice.iqoala")
        session = path.split("/")[-1]

        # different inputs based on program to be used
        if session == "qkd":
            # theta should be either 0 (Z basis meas) or 24 (X basis measurement) for rot Y rotation
            alice_inputs = [ProgramInput(
                # TODO: use kwargs or default values?
                {"bob_id": bob_id, "theta0": 0, "theta1": 0, "theta2": 0, "theta3": 0, "theta4": 0}
            ) for _ in range(num_iterations)]
        elif session == "pingpong":
            alice_inputs = [ProgramInput({"bob_id": bob_id}) for _ in range(num_iterations)]
        elif session == "bqc":
            alice_inputs = [ProgramInput(
                # TODO: use kwargs or default values?
                {"bob_id": bob_id, "alpha": 8, "beta": 8, "theta1": 0, "theta2": 0}
            ) for _ in range(num_iterations)]
        else:
            raise ValueError(f"Unknown session type {session}")

        alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
        alice_batch = create_batch(alice_program, alice_unit_module, alice_inputs, num_iterations)
        alice_procnode.submit_batch(alice_batch)

    alice_procnode.initialize_processes()
    alice_tasks = alice_procnode.scheduler.get_tasks_to_schedule()

    alice_schedule = create_task_schedule(alice_tasks, "node_schedules/" + node_schedule_name + "-alice.csv")
    print(f"\nAlice's schedule:\n{alice_schedule}")
    alice_procnode.scheduler.upload_schedule(alice_schedule)

    for (path, num_iterations) in dataset.items():
        bob_program = load_program(path + "_bob.iqoala")
        session = path.split("/")[-1]

        if session == "qkd":
            # theta should be either 0 (Z basis meas) or 24 (X basis measurement) for rot Y rotation
            bob_inputs = [ProgramInput(
                # TODO: use kwargs or default values?
                {"alice_id": alice_id, "theta0": 0, "theta1": 0, "theta2": 0, "theta3": 0, "theta4": 0}
            ) for _ in range(num_iterations)]
        elif session == "pingpong":
            bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]
        elif session == "bqc":
            bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]
        else:
            raise ValueError(f"Unknown session type {session}")

        bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
        bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, num_iterations)
        bob_procnode.submit_batch(bob_batch)

    bob_procnode.initialize_processes()
    bob_tasks = bob_procnode.scheduler.get_tasks_to_schedule()

    bob_schedule = create_task_schedule(bob_tasks, "node_schedules/" + node_schedule_name + "-bob.csv")
    print(f"\nBob's schedule:\n{bob_schedule}")
    bob_procnode.scheduler.upload_schedule(bob_schedule)

    network.start()
    ns.sim_run()

    alice_results = alice_procnode.scheduler.get_batch_results()
    bob_results = bob_procnode.scheduler.get_batch_results()
    makespan = ns.sim_time()

    return NodeScheduleResult(alice_results, bob_results, alice_procnode, makespan)


@dataclass
class NodeScheduleResult:
    alice_results: Dict[int, BatchResult]
    bob_results: Dict[int, BatchResult]
    alice_procnode: ProcNode
    makespan: float


def evaluate_node_schedule(node_schedule_name):
    # node_schedule_name is `6-sessions_dataset-1_NS-1_schedule-HEU_node`
    parts = node_schedule_name.split("_")
    dataset_id = int(parts[1].split("-")[1])
    n_sessions = int(parts[0].split("-")[0])
    dataset = create_dataset(id=dataset_id, n_sessions=n_sessions)

    result = execute_node_schedule(dataset, node_schedule_name)
    return result

    # # you need to somehow zip the sessions that were run
    # if session_type == "BQC":
    #     # Effective computation: measure in Z the following state:
    #     # H Rz(beta) H Rz(alpha) |+>
    #     # m2 should be this outcome
    #
    #     # angles are in multiples of pi/16
    #     # check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=12)
    #     bob_batch_results = bqc_result.bob_results
    #     for _, batch_results in bob_batch_results.items():
    #         program_results = batch_results.results
    #         m2s = [result.values["m2"] for result in program_results]
    #         assert all(m2 == expected for m2 in m2s)
    # elif session_type == "pingpong":
    #     alice_batch_results = result.alice_results
    #     for _, batch_results in alice_batch_results.items():
    #         program_results = batch_results.results
    #         outcomes = [result.values["outcome"] for result in program_results]
    #         assert all(outcome == 1 for outcome in outcomes)
    #     q0 = result.alice_procnode.qdevice.get_local_qubit(0)
    #     assert has_state(q0, ketstates.s1, margin=1 - fidelity_threshold)
    # elif session_type == "qkd":
    #     qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    #     alice_results = qkd_result.alice_results.results
    #     bob_results = qkd_result.bob_results.results
    #
    #     print(alice_results)
    #     print(bob_results)
    #
    #     assert len(alice_results) == num_iterations
    #     assert len(bob_results) == num_iterations
    #
    #     alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    #     bob_outcomes = [bob_results[i].values for i in range(num_iterations)]
    #
    #     for alice, bob in zip(alice_outcomes, bob_outcomes):
    #         assert alice["m0"] == bob["m0"]


if __name__ == "__main__":
    start = time.time()

    # res = evaluate_node_schedule("6-sessions_dataset-0_NS-128_schedule-HEU_node")
    # res = evaluate_node_schedule("6-sessions_dataset-1_NS-169_schedule-HEU_node")
    res = evaluate_node_schedule("6-sessions_dataset-2_NS-0_schedule-HEU_node")

    end = time.time()
    print(end - start)
