import logging
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from pycsp3 import *
from node_schedule import NodeSchedule
from activity_metadata import ActiveSet
from network_schedule import NetworkSchedule
from datasets import create_dataset
from setup_logging import setup_logging


# TODO: should this be moved to node_schedule.py?
def create_node_schedule(dataset, role, network_schedule=None, network_schedule_name=None, schedule_type="HEU",
                         save_schedule=False, save_schedule_filename=None, save_metrics=False, save_metrics_name=None,
                         save_metrics_filename=None, dataset_id=None):
    # TODO: return a status code? would be nice to know if a node schedule was not created because NS is infeasible
    """

    :param dataset:
    :param role:
    :param network_schedule:
    :param schedule_type:
    :param save_schedule:
    :param save_schedule_filename:
    :param save_metrics:
    :param save_metrics_name:
    :param save_metrics_filename:
    :param dataset_id
    :return:
    """
    logger = logging.getLogger("program_scheduling")

    if network_schedule is not None:
        network_schedule_name = f"network-schedule_{network_schedule.n_sessions}-sessions_dataset-" \
                                f"{network_schedule.dataset_id}_id-{network_schedule.id}.csv"
    elif network_schedule_name is not None:
        path = os.path.dirname(__file__) + "network_schedules/"
        csv = pd.read_csv(path + network_schedule_name)
        parts = network_schedule_name.split("_")
        assert int(parts[2].split("-")[1]) == dataset_id
        assert int(parts[1].split("-")[0]) == sum(dataset.values())
        network_schedule = NetworkSchedule(dataset_id=dataset_id, n_sessions=sum(dataset.values()),
                                           sessions=list(csv["session"]),
                                           start_times=list(map(lambda x: int(x), csv["start_time"])))
    else:
        network_schedule = None

    active = ActiveSet.create_active_set(dataset=dataset, role=role, network_schedule=network_schedule)
    active.scale_down()
    if network_schedule is not None:
        network_schedule.rewrite_sessions(dataset)
        network_schedule.start_times = np.divide(network_schedule.start_times, active.gcd)
        network_schedule.start_times = list(map(lambda x: int(x), network_schedule.start_times))
        logger.debug(f"Network schedule length={network_schedule.length}")
        network_schedule.length = network_schedule.length / active.gcd

    # TODO: how to decide on the extra length?
    schedule_size = int(network_schedule.length) + 100000 if network_schedule is not None else 2 * int(sum(active.durations))
    capacities = [1, 1]  # capacity of [CPU, QPU]

    # x[i] is the starting time of the ith job
    x = VarArray(size=active.n_blocks, dom=range(schedule_size))

    # taken from http://pycsp.org/documentation/models/COP/RCPSP/
    def cumulative_for(k):
        # TODO: this doesn't work if session is purely quantum or purely classical
        origins, lengths, heights = zip(*[(x[i], active.durations[i], active.resource_reqs[i][k])
                                          for i in range(active.n_blocks) if active.resource_reqs[i][k] > 0])
        return Cumulative(origins=origins, lengths=lengths, heights=heights)

    def get_CS_indices():
        forward = list(zip(active.ids, active.cs_ids))
        backwards = list(zip(active.ids, active.cs_ids))
        backwards.reverse()

        CS_indices = []
        for (session_id, cs_id) in set(zip(active.ids, active.cs_ids)):
            if cs_id is not None:
                CS_indices.append((session_id, cs_id, forward.index((session_id, cs_id)),
                                  len(forward) - backwards.index((session_id, cs_id)) - 1))

        return CS_indices

    # constraints
    satisfy(
        # precedence constraints
        [x[i] + active.durations[i] <= x[j] for i in range(active.n_blocks) for j in active.successors[i]],
        # resource constraints
        [cumulative_for(k) <= capacity for k, capacity in enumerate(capacities)],
        # constraints for max time lags
        [(x[i + 1] - (x[i] + active.durations[i])) <= active.d_max[i + 1] for i in range(active.n_blocks - 1)
         if (active.types[i + 1] != "QC" or active.d_max[i] is None) and active.d_max[i + 1] is not None],
        # TODO: make a constraint that nothing is scheduled inbetween the first and last Q block in a critical section
        [(x[i] < x[start]) | (x[end] < x[i]) for (session_id, cs_id, start, end) in get_CS_indices()
         for i in range(active.n_blocks - 1) if (active.ids[i] != session_id and active.cs_ids[i] != cs_id)]
    )

    def get_QC_indices(without=None):
        indices = [i for i in range(0, active.n_blocks - 1) if active.types[i] == "QC"]
        if without is not None:
            for remove in without:
                indices.remove(remove)
        return indices

    if network_schedule is not None:
        satisfy(
            [x[i] in set(network_schedule.get_session_start_times(active.ids[i])) for i in get_QC_indices()],
            # order of a qc block is correct
            [x[i] in set(network_schedule.get_qc_block_start_times(active.qc_indices[i])) for i in get_QC_indices()]
        )
    else:
        satisfy(
            [(x[i + 1] - (x[i] + active.durations[i])) <= active.d_max[i + 1] for i in range(active.n_blocks - 1)
             if active.types[i + 1] == "QC" and active.d_max[i + 1] is not None]
        )

    if schedule_type == "NAIVE":
        satisfy(
            [x[i] < x[i + 1] for i in range(active.n_blocks - 1)],
        )

    # optional objective function
    if schedule_type == "OPT":
        minimize(
            Maximum([x[i] + active.durations[i] for i in range(active.n_blocks)])
        )

    instance = compile()
    ace = solver(ACE)

    # https://github.com/xcsp3team/pycsp3/blob/master/docs/optionsSolvers.pdf
    # here you can possibly define other heuristics to use
    heuristics = {}

    start = time.time()
    result = ace.solve(instance, dict_options=heuristics)
    end = time.time()

    if status() is SAT or status() is OPTIMUM:
        active.scale_up()
        start_times = [s * active.gcd for s in solution().values]
        ns = NodeSchedule(active, start_times)
        # TODO: print ns if debugging?

        # name of a node schedule needs to include: n_sessions, dataset, NS ID, schedule type, alice/bob
        network_schedule_id = None if network_schedule is None else network_schedule_name.split(".")[0].split("-")[-1]
        name = f"{sum(dataset.values())}-sessions_dataset-{dataset_id if dataset_id is not None else 'unknown'}_" \
               f"NS-{network_schedule_id}_schedule-{schedule_type}_node-{role}"
        if save_schedule:
            save_schedule_filename = f"{name}.csv" if save_schedule_filename is None \
                else f"{save_schedule_filename}-{role}.csv"
            ns.save_node_schedule(filename=save_schedule_filename)

        if save_metrics:
            save_metrics_name = name if save_metrics_name is None else save_metrics_name
            save_metrics_filename = "../node_schedule_results.csv" if save_metrics_filename is None \
                else save_metrics_filename
            ns.save_success_metrics(name=save_metrics_name, filename=save_metrics_filename, role=role,
                                    type=schedule_type, network_schedule=network_schedule, dataset=dataset,
                                    solve_time=end - start)
        logger.info("Found node schedule successfully.")
        logger.info("Time taken to finish: %.4f seconds" % (end - start))
        clear()

        for filename in os.listdir():
            if filename.endswith(".log") or filename.endswith(".xml"):
                os.remove(filename)

        return "SAT"
    elif status() is UNKNOWN:
        logger.info("\nThe solver cannot find a solution. The problem is probably too large.")
        logger.info("Time taken to finish: %.4f seconds" % (end - start))
        clear()
        return "UNKNOWN"
    elif status() is UNSAT:
        logger.info("\nNo feasible node schedule can be found. "
                     "Consider making the length of node schedule longer or finding a better network schedule.")
        logger.info("Time taken to finish: %.4f seconds" % (end - start))
        clear()
        return "UNSAT"
    else:
        logger.info("\nSomething else went wrong.")
        logger.info("Time taken to finish: %.4f seconds" % (end - start))
        clear()


def get_dataset(dataset_id, n_sessions, n_bqc, n_qkd, n_pp):
    if dataset_id is None and (n_bqc + n_qkd + n_pp) == 0:
        raise ValueError("No sessions are being scheduled. Please define either a dataset or "
                         "number of specific sessions to schedule.")
    if dataset_id is not None:
        return create_dataset(dataset_id, n_sessions)
    else:
        d = {}
        for n, c in list(zip([n_bqc, n_qkd, n_pp], ["../configs/bqc", "../configs/qkd", "../configs/pingpong"])):
            if n > 0:
                d.update({c: n})
        return d


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument('-d', '--dataset', required=False, type=int, default=None,
                        help="Dataset of sessions to schedule.")
    # number of sessions in a dataset
    parser.add_argument('-s', '--n_sessions', required=False, type=int, default=18,
                        help="Total number of sessions in a dataset.")
    # number of sessions bqc
    parser.add_argument('-bqc', '--n_bqc_sessions', required=False, type=int, default=0,
                        help="Number of BQC sessions to schedule.")
    # number of sessions qkd
    parser.add_argument('-qkd', '--n_qkd_sessions', required=False, type=int, default=0,
                        help="Number of QKD sessions to schedule.")
    # number of sessions pingpong
    parser.add_argument('-pp', '--n_pp_sessions', required=False, type=int, default=0,
                        help="Number of PingPong sessions to schedule.")
    # network schedule file
    parser.add_argument('-ns', '--network_schedule_name', required=False, type=str, default=None,
                        help="File with network schedule.")
    # role --> if None, node schedule will be created for both alice and bob
    parser.add_argument('-role', '--role', required=False, type=str, default=None,
                        help="File with network schedule.")
    # T/F for opt
    parser.add_argument('-opt', "--optimal", dest="opt", action="store_true",
                        help="Create an optimal schedule (i.e. use an objective function to minimise makespan).")
    # T/F for naive
    parser.add_argument('-naive', "--naive", dest="naive", action="store_true",
                        help="Create a naive schedule (i.e. schedule all blocks consecutively.")
    # T/F save schedule
    parser.add_argument('-ss', "--save_schedule", dest="save_schedule", action="store_true",
                        help="Save the node schedule.")
    # filename for saving schedule
    parser.add_argument('-ssf', '--save_schedule_filename', required=False, type=str, default=None,
                        help="Filename for saving the schedule.")
    # T/F save metrics
    parser.add_argument('-sm', "--save_metrics", dest="save_metrics", action="store_true",
                        help="Save the success metrics.")
    # name of schedule
    parser.add_argument('-smn', '--save_metrics_name', required=False, type=str, default=None,
                        help="Name for saving the schedule success metrics.")
    # filename for saving metrics
    parser.add_argument('-smf', '--save_metrics_filename', required=False, type=str, default=None,
                        help="Filename for saving the schedule success metrics.")
    # logging
    parser.add_argument('--log', dest='loglevel', type=str, required=False, default="INFO",
                        help="Set log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    args, unknown = parser.parse_known_args()

    setup_logging(args.loglevel)

    dataset = get_dataset(args.dataset, args.n_sessions,
                          args.n_bqc_sessions, args.n_qkd_sessions, args.n_pp_sessions)

    if args.opt:
        schedule_type = "OPT"
    elif args.naive:
        schedule_type = "NAIVE"
    else:
        schedule_type = "HEU"

    for r in ["alice", "bob"] if args.role is None else [args.role]:
        create_node_schedule(dataset=dataset, role=r, network_schedule_name=args.network_schedule_name,
                             schedule_type=schedule_type, save_schedule=args.save_schedule,
                             save_schedule_filename=args.save_schedule_filename, save_metrics=args.save_metrics,
                             save_metrics_name=args.save_metrics_name, save_metrics_filename=args.save_metrics_filename,
                             dataset_id=args.dataset)
