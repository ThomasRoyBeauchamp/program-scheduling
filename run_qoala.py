# from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.schedule import TaskSchedule, TaskScheduleEntry
from qoala.sim.build import build_network
from qoala.util.logging import LogManager


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


@dataclass
class PingPongResult:
    alice_results: Dict[int, BatchResult]
    bob_results: Dict[int, BatchResult]


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


def run_pingpong(num_iterations: int) -> PingPongResult:
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["bob", "alice"])
    alice_id = network_info.get_node_id("alice")
    bob_id = network_info.get_node_id("bob")

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=500_000
    )
    network = build_network(network_cfg, network_info)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_program = load_program("configs/pingpong_alice.iqoala")
    alice_inputs = [ProgramInput({"bob_id": bob_id}) for _ in range(num_iterations)]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(alice_program, alice_unit_module, alice_inputs, num_iterations)
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_tasks = alice_procnode.scheduler.get_tasks_to_schedule()
    print("Alice tasks:")
    print([str(t) for t in alice_tasks])
    # TODO: make into a CL argument
    node_schedule_config = "node_schedules/temp_pingpong_alice_12.csv"
    alice_schedule = create_task_schedule(alice_tasks, node_schedule_config)
    # alice_schedule = TaskSchedule.consecutive(alice_tasks)
    print("\nAlice schedule:")
    print(alice_schedule)
    alice_procnode.scheduler.upload_schedule(alice_schedule)

    bob_program = load_program("configs/pingpong_bob.iqoala")
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, num_iterations)
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_tasks = bob_procnode.scheduler.get_tasks_to_schedule()
    print("\n\nBob tasks:")
    print([str(t) for t in bob_tasks])
    node_schedule_config = "node_schedules/temp_pingpong_bob_12.csv"
    bob_schedule = create_task_schedule(bob_tasks, node_schedule_config)
    # bob_schedule = TaskSchedule.consecutive(bob_tasks)
    print("\nBob schedule:")
    print(bob_schedule)
    bob_procnode.scheduler.upload_schedule(bob_schedule)

    network.start()
    ns.sim_run()

    alice_results = alice_procnode.scheduler.get_batch_results()
    bob_results = bob_procnode.scheduler.get_batch_results()
    print("End of execution: " + str(ns.sim_time()))

    return PingPongResult(alice_results, bob_results)


@dataclass
class QkdResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_qkd(num_iterations: int, alice_file: str, bob_file: str):
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["alice", "bob"])
    alice_id = network_info.get_node_id("alice")
    bob_id = network_info.get_node_id("bob")

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network = build_network(network_cfg, network_info)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_program = load_program(alice_file)
    # theta should be either 0 (Z basis meas) or 24 (X basis measurement) for rot Y rotation
    alice_inputs = [ProgramInput({"bob_id": bob_id, "theta0": 0, "theta1": 0, "theta2": 0, "theta3": 0,
                                  "theta4": 0}) for _ in range(num_iterations)]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(
        alice_program, alice_unit_module, alice_inputs, num_iterations
    )
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_tasks = alice_procnode.scheduler.get_tasks_to_schedule()
    print("Alice tasks:")
    print([str(t) for t in alice_tasks])
    alice_schedule = TaskSchedule.consecutive(alice_tasks)

    with open("node_schedules/temp_qkd_ck_alice_indices.yml", 'r') as file_handle:
        indices = yaml.load(file_handle, yaml.SafeLoader) or {}
    alice_schedule = TaskSchedule([TaskScheduleEntry(alice_tasks[indices[i]]) for i in range(len(indices))])

    print("\nAlice schedule:")
    print(alice_schedule)
    alice_procnode.scheduler.upload_schedule(alice_schedule)

    bob_program = load_program(bob_file)
    # theta should be either 0 (Z basis meas) or 24 (X basis measurement) for rot Y rotation
    bob_inputs = [ProgramInput({"alice_id": alice_id, "theta0": 0, "theta1": 0, "theta2": 0, "theta3": 0,
                                "theta4": 0}) for _ in range(num_iterations)]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, num_iterations)
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_tasks = bob_procnode.scheduler.get_tasks_to_schedule()
    print("\n\nBob tasks:")
    print([str(t) for t in bob_tasks])
    bob_schedule = TaskSchedule.consecutive(bob_tasks)
    with open("node_schedules/temp_qkd_ck_bob_indices.yml", 'r') as file_handle:
        indices = yaml.load(file_handle, yaml.SafeLoader) or {}
    bob_schedule = TaskSchedule([TaskScheduleEntry(bob_tasks[indices[i]]) for i in range(len(indices))])

    print("\nBob schedule:")
    print(bob_schedule)
    bob_procnode.scheduler.upload_schedule(bob_schedule)

    network.start()
    ns.sim_run()

    # only one batch (ID = 0), so get value at index 0
    alice_results = alice_procnode.scheduler.get_batch_results()[0]
    bob_results = bob_procnode.scheduler.get_batch_results()[0]
    print("End of execution: " + str(ns.sim_time()))

    return QkdResult(alice_results, bob_results)


def pingpong():
    # LogManager.set_log_level("DEBUG")

    def check(num_iterations):
        ns.sim_reset()
        result = run_pingpong(num_iterations)
        print("finished running!!")
        assert len(result.alice_results) > 0
        assert len(result.bob_results) > 0

        alice_batch_results = result.alice_results
        for _, batch_results in alice_batch_results.items():
            program_results = batch_results.results
            outcomes = [result.values["outcome"] for result in program_results]
            assert all(outcome == 1 for outcome in outcomes)

    check(12)


def qkd_ck():
    # LogManager.set_log_level("DEBUG")
    ns.sim_reset()

    num_iterations = 2
    alice_file = "configs/qkd_ck_alice.iqoala"
    bob_file = "configs/qkd_ck_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_results.results
    bob_results = qkd_result.bob_results.results

    print(alice_results)
    print(bob_results)

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


if __name__ == "__main__":
    pingpong()
    # qkd_ck()
