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
    parser.add_argument("-n", '--n_ns', required=False, default=0, type=int,
                        help="How many networks schedules should be created.")
    parser.add_argument("--existing_ns", dest="existing_ns", action="store_true",
                        help="Include network schedule constraints but use existing network schedules.")
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

    if args.existing_ns and args.n_ns != 0:
        logger.warning("Ignoring the number of NS to create and using existing IDS.")

    dataset_ids = range(7) if args.all else [args.dataset_id]
    schedule_type = "OPT" if args.opt else ("NAIVE" if args.naive else "HEU")

    start = time.time()

    for dataset_id in dataset_ids:
        d_start = time.time()
        logger.info(f"CREATING SCHEDULES FOR DATASET {dataset_id}:")
        ns_length_factor = {6: 3, 12: 5}.get(args.n_sessions)

        # determine network schedule ids
        if args.n_ns == 0 and not args.existing_ns:
            ns_ids = [None]
        elif args.existing_ns:
            logger.info("Using existing network schedules.")
            ns_ids = NetworkSchedule.get_relevant_ids(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                      length_factor=ns_length_factor)
        else:
            ns_ids = []
            for i in range(args.n_ns):
                network_schedule = NetworkSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions,
                                                   length_factor=ns_length_factor)
                ns_ids.append(network_schedule.id)

        for ns_id in ns_ids:
            length_factor = {6: 3, 12: 5}.get(args.n_sessions) if ns_id is not None else 2
            ns_string = "no network schedule" if ns_id is None else f"network schedule with id {ns_id}"
            logger.info(f"Creating node schedules for dataset {dataset_id} with {args.n_sessions} sessions in "
                        f"{schedule_type} approach based on {ns_string}.")
            for role in ["alice", "bob"]:
                node_schedule = NodeSchedule(dataset_id=dataset_id, n_sessions=args.n_sessions, ns_id=ns_id,
                                             role=role, schedule_type=schedule_type,
                                             ns_length_factor=length_factor)
                if node_schedule.status != "SAT":
                    logger.warning(f"Network schedule with id {ns_id} did not result in "
                                   f"a feasible node schedule for {role}.")
                    break  # we don't need to create a node schedule for bob if there is no feasible schedule for alice

        d_end = time.time()
        logger.info(f"Time taken to finish: {round(d_end - d_start, 4)} seconds for dataset {dataset_id}.\n")
    end = time.time()
    logger.info("Time taken to finish: %.4f seconds" % (end - start))
