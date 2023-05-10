from argparse import ArgumentParser

from program_scheduling.node_schedule import NodeSchedule
from setup_logging import setup_logging

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset-id', required=True, type=int,
                        help="Dataset of sessions to schedule.")
    parser.add_argument('-s', '--n_sessions', required=True, type=int,
                        help="Total number of sessions in a dataset.")
    parser.add_argument('-ns', '--ns_id', required=False, type=int, default=None,
                        help="Total number of sessions in a dataset.")
    parser.add_argument('-role', '--role', required=True, type=str)
    parser.add_argument('--opt', dest="opt", action="store_true",
                        help="Use optimal scheduling approach (i.e. use an objective function to minimise makespan).")
    parser.add_argument('--naive', dest="naive", action="store_true",
                        help="Use naive scheduling approach (i.e. schedule all blocks consecutively).")
    parser.add_argument('-l', '--length_factor', required=False, type=int, default=0,
                        help="Seed for randomly generating the network schedule.")
    parser.add_argument('--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    args, unknown = parser.parse_known_args()

    schedule_type = "OPT" if args.opt else ("NAIVE" if args.naive else "HEU")
    length_factor = {6: 3, 12: 5}.get(args.n_sessions) if args.length_factor == 0 else args.length_factor

    setup_logging(args.loglevel)

    NodeSchedule(dataset_id=args.dataset_id, n_sessions=args.n_sessions, ns_id=args.ns_id, role=args.role,
                 schedule_type=schedule_type, ns_length_factor=length_factor)
