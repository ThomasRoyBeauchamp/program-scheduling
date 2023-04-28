from argparse import ArgumentParser
from copy import deepcopy

from create_node_schedule import create_node_schedule
from datasets import create_dataset
from network_schedule import NetworkSchedule

if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument('-d', '--dataset-id', required=True, type=int,
                        help="Dataset of sessions to schedule.")
    # number of sessions in a dataset
    parser.add_argument('-s', '--n_sessions', required=True, type=int,
                        help="Total number of sessions in a dataset.")
    parser.add_argument('-ns', '--n_ns', required=False, default=1, type=int,
                        help="Number of random network schedules to generate.")
    args, unknown = parser.parse_known_args()

    for i in range(args.n_ns):
        success = False
        while not success:
            ns = NetworkSchedule(dataset_id=args.dataset_id, n_sessions=args.n_sessions, save=True)
            ns2 = deepcopy(ns)

            dataset = create_dataset(id=args.dataset_id, n_sessions=args.n_sessions)
            alice_res = create_node_schedule(dataset, "alice", network_schedule=ns, dataset_id=args.dataset_id,
                                             save_schedule=True)
            bob_res = create_node_schedule(dataset, "bob", network_schedule=ns2, dataset_id=args.dataset_id,
                                           save_schedule=True)
            if alice_res == "SAT" and bob_res == "SAT":
                success = True
