import logging
import time
from argparse import ArgumentParser
from copy import deepcopy

from create_node_schedule import create_node_schedule, create_dataset
from network_schedule import NetworkSchedule
from setup_logging import setup_logging

if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument('-d', '--dataset-id', required=False, default=6, type=int,
                        help="Dataset of sessions to schedule.")
    # number of sessions in a dataset
    parser.add_argument('-s', '--n_sessions', required=False, default=6, type=int,
                        help="Total number of sessions in a dataset.")
    parser.add_argument("--all", dest="all", default=False, action="store_true",
                        help="Run all datasets.")
    parser.add_argument("-n", '--n_ns', required=False, default=10, type=int,
                        help="How many networks schedules should be created.")
    # logging
    parser.add_argument('-log', '--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    args, unknown = parser.parse_known_args()

    setup_logging(args.loglevel)
    logger = logging.getLogger(__name__)

    # args.dataset_id = 3
    # args.n_ns = 1

    start = time.time()
    datasets = range(7) if args.all else [args.dataset_id]
    for d in datasets:
        ns_ids = []
        correct_node_schedules = []

        last_id = -1
        for i in range(args.n_ns):
            ns = NetworkSchedule(dataset_id=d, n_sessions=args.n_sessions, save=False, seed=last_id + 1)
            # ns = NetworkSchedule(dataset_id=d, n_sessions=args.n_sessions)
            ns2 = deepcopy(ns)

            ns_ids.append(ns.id)
            last_id = ns.id

            dataset = create_dataset(id=d, n_sessions=args.n_sessions)
            alice_res = create_node_schedule(dataset, "alice", network_schedule=ns, dataset_id=args.dataset_id)
            bob_res = create_node_schedule(dataset, "bob", network_schedule=ns2, dataset_id=args.dataset_id)
            if alice_res == "SAT" and bob_res == "SAT":
                correct_node_schedules.append(ns.id)

        logger.debug(f"Dataset {d}: {args.n_ns} network schedules up to ID {ns_ids[-1]} "
                      f"({round(args.n_ns/ns_ids[-1],4) * 100 if ns_ids[-1] != 0 else 100}%, ids: {ns_ids}). "
                      f"Out of that, {len(correct_node_schedules)} network schedules resulted in "
                      f"feasible node schedules ({round(len(correct_node_schedules)/args.n_ns, 2) * 100}% success, "
                      f"ids: {correct_node_schedules}).")

    end = time.time()
    logger.info("Time taken to finish: %.4f seconds" % (end - start))
