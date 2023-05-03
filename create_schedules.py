import logging
import time
from argparse import ArgumentParser

# from program_scheduling.create_node_schedule import create_node_schedule, create_dataset
from program_scheduling.network_schedule import NetworkSchedule
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
                        help="Use naive scheduling approach (i.e. schedule all blocks consecutively.")
    parser.add_argument('--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    args, unknown = parser.parse_known_args()

    setup_logging(args.loglevel)
    logger = logging.getLogger(__name__)

    if args.dataset_id is None and not args.all:
        raise ValueError("No dataset ID was specified and the `--all` flag is not set.")

    dataset_ids = range(7) if args.all else [args.dataset_id]

    start = time.time()
    for dataset_id in dataset_ids:
        logger.info(f"Creating schedules for dataset {dataset_id}.")

        if args.n_ns == 0:
            # TODO: immediately create node schedules
            pass
        else:
            # TODO: decide on the length factor for network schedule
            network_schedule_length_factor = None
            for i in range(args.n_ns):
                network_schedule_id = NetworkSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                      length_factor=network_schedule_length_factor).id
                alice_node_schedule_result = None  # TODO: call the right method
                bob_node_schedule_result = None  # TODO: call the right method
                if alice_node_schedule_result != "SAT" or bob_node_schedule_result != "SAT":
                    logger.warning(f"Network schedule with id {network_schedule_id} did not result in feasible "
                                   f"node schedules.")

    end = time.time()
    logger.info("Time taken to finish: %.4f seconds" % (end - start))
