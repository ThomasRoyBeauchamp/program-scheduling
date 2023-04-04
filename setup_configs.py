import os
import yaml
from typing import List
from argparse import ArgumentParser

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ProcNodeNetworkConfig,
    ProcNodeConfig,
    TopologyConfig,
    LatenciesConfig
)
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import ProgramInput, BatchInfo
from qoala.sim.build import build_network

sourcecode = "https://github.com/QuTech-Delft/qoala-sim/blob/" \
             "635728c60ded03e84f254ab413071a6392e19a7c/tests/integration/pingpong/test_pingpong.py"


# copied from `sourcecode`
def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


# copied from `sourcecode`
def create_network_info(names: List[str]) -> NetworkInfo:
    env = NetworkInfo.with_nodes({i: name for i, name in enumerate(names)})
    env.set_global_schedule([0, 1, 2])
    env.set_timeslot_len(1e6)
    return env


# copied from `sourcecode`
def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=100_000, qnos_instr_time=1000
        ),
    )


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


def retrieve_tasks(config, save_filename=None, num_qubits=3):
    if save_filename is None:
        save_filename = config

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

    alice_program = load_program("configs/" + config + "_alice.iqoala")
    # this is not necessarily correct but should not matter for retrieving the blocks information
    alice_inputs = [ProgramInput({"bob_id": bob_id})]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(
        alice_program, alice_unit_module, alice_inputs, 1
    )
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_tasks = alice_procnode.scheduler.get_tasks_to_schedule()

    with open('configs/' + save_filename + '_alice.yml', 'w') as outfile:
        yaml.dump({"session_id": "TODO",
                   "app_deadline": "TODO",
                   "blocks": [{bt.block_name: {"type": bt.typ.name, "duration": int(bt.duration), "CS": "TODO"}}
                              for bt in alice_tasks]}, outfile, default_flow_style=False, sort_keys=False)

    bob_program = load_program("configs/" + config + "_bob.iqoala")
    bob_inputs = [ProgramInput({"alice_id": alice_id})]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, 1)
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_tasks = bob_procnode.scheduler.get_tasks_to_schedule()

    with open('configs/' + save_filename + '_bob.yml', 'w') as outfile:
        yaml.dump({"session_id": "TODO",
                   "app_deadline": "TODO",
                   "blocks": [{bt.block_name: {"type": bt.typ.name, "duration": int(bt.duration), "CS": "TODO"}}
                              for bt in bob_tasks]}, outfile, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help="Name of the iqoala program (without the `_alice` or `_bob` suffix).")
    parser.add_argument('-s', '--save_filename', required=False, type=str, default=None,
                        help="Name of the file to save results in.")

    args, unknown = parser.parse_known_args()
    retrieve_tasks(args.config, args.save_filename)
