import logging
import time
from argparse import ArgumentParser

from program_scheduling.network_schedule import NetworkSchedule
from program_scheduling.node_schedule import NodeSchedule
from setup_logging import setup_logging

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset-id', required=False, type=int,
                        help="Dataset ID which specifies which sessions to schedule.")
    parser.add_argument("--all", dest="all", default=False, action="store_true",
                        help="Generate schedules for all datasets at once.")
    parser.add_argument('-s', '--n_sessions', required=False, default=6, type=int,
                        help="Total number of sessions in a dataset.")
    parser.add_argument("-n", '--n_ns', required=False, default=100, type=int,
                        help="How many networks schedules should be created.")
    parser.add_argument('--opt', dest="opt", action="store_true",
                        help="Use optimal scheduling approach (i.e. use an objective function to minimise makespan).")
    parser.add_argument('--naive', dest="naive", action="store_true",
                        help="Use naive scheduling approach (i.e. schedule all blocks consecutively).")
    parser.add_argument('--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    args, unknown = parser.parse_known_args()

    setup_logging(args.loglevel)
    logger = logging.getLogger("program_scheduling")

    if args.dataset_id is None and not args.all:
        raise ValueError("No dataset ID was specified and the `--all` flag is not set.")

    dataset_ids = range(7) if args.all else [args.dataset_id]
    schedule_type = "OPT" if args.opt else ("NAIVE" if args.naive else "HEU")

    start = time.time()
    for dataset_id in dataset_ids:
        logger.info(f"Creating schedules for dataset {dataset_id}.")

        if args.n_ns == 0:
            for role in ["alice", "bob"]:
                node_schedule = NodeSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions, ns_id=None,
                                             role=role, schedule_type=schedule_type)
        else:
            # TODO: decide on the length factor for network schedules
            network_schedule_length_factor = {6: 3, 12: 5}.get(args.n_sessions)
            for i in range(args.n_ns):
                network_schedule = NetworkSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                   length_factor=network_schedule_length_factor)

                alice_node_schedule = NodeSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                   ns_id=network_schedule.id, role="alice", schedule_type=schedule_type,
                                                   ns_length_factor=network_schedule_length_factor)
                bob_node_schedule = NodeSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                 ns_id=network_schedule.id, role="bob", schedule_type=schedule_type,
                                                 ns_length_factor=network_schedule_length_factor)

                if alice_node_schedule.status != "SAT" or bob_node_schedule.status != "SAT":
                    logger.warning(f"Network schedule with id {network_schedule.id} did not result in feasible "
                                   f"node schedules.")

    end = time.time()
    logger.info("Time taken to finish: %.4f seconds" % (end - start))
