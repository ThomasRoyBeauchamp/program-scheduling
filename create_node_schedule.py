import time
from argparse import ArgumentParser

from pycsp3 import *
from node_schedule import NodeSchedule
from activity_metadata import ActiveSet
from network_schedule import NetworkSchedule
from datasets import create_dataset


def create_node_schedule(dataset, role, network_schedule=None, schedule_type="HEU", save_schedule=False,
                         save_schedule_filename=None, save_metrics=False, save_metrics_name=None,
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
    if network_schedule is None:
        network_schedule = NetworkSchedule()
    # TODO: read out network schedule if the config is defined

    active = ActiveSet.create_active_set(dataset=dataset, role=role, network_schedule=network_schedule)
    active.scale_down()
    """
        TODO: How to define length of node_schedule?
        If it's too low, there might not be any feasible solution. 
        If it's too high, we are unnecessarily solving a more complex problem. 
    """
    schedule_size = 2 * int(sum(active.durations))  # TODO: retrieve from length of network schedule
    capacities = [1, 1]  # capacity of [CPU, QPU]

    # x[i] is the starting time of the ith job
    x = VarArray(size=active.n_blocks, dom=range(schedule_size))

    # taken from http://pycsp.org/documentation/models/COP/RCPSP/
    def cumulative_for(k):
        # TODO: this doesn't work if session is purely quantum or purely classical
        origins, lengths, heights = zip(*[(x[i], active.durations[i], active.resource_reqs[i][k])
                                          for i in range(active.n_blocks) if active.resource_reqs[i][k] > 0])
        return Cumulative(origins=origins, lengths=lengths, heights=heights)

    # constraints
    satisfy(
        # precedence constraints
        [x[i] + active.durations[i] <= x[j] for i in range(active.n_blocks) for j in active.successors[i]],
        # resource constraints
        [cumulative_for(k) <= capacity for k, capacity in enumerate(capacities)],
        # constraints for max time lags
        [(x[i + 1] - (x[i] + active.durations[i])) <= active.d_max[i + 1] for i in range(active.n_blocks - 1)]
    )

    # TODO: the above max time lags constraint does not apply to QC if network schedule is defined
    # TODO: make a new max time lags constraint that applies to QC if network schedule is defined

    if network_schedule.is_defined:
        # TODO: this needs to be fixed when we allow for multiple QC blocks in a session (also rescale)
        # satisfy(
        # [(x[i] == network_schedule.get_session_start_time(active.ids[i]) for i in range(active.n_blocks - 1)
        #   if network_schedule.is_defined and active.types[i] == "QC")]
        # )
        pass

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

    print(f"\nTrying to construct a node schedule of length {schedule_size} ({schedule_size * active.gcd})")
    start = time.time()
    result = ace.solve(instance, dict_options=heuristics)
    end = time.time()

    if status() is SAT or status() is OPTIMUM:
        active.scale_up()
        start_times = [s * active.gcd for s in solution().values]
        ns = NodeSchedule(active, start_times)

        ns.print()

        name = f"dataset_{dataset_id if dataset_id is not None else 'unknown'}_{role}_{schedule_type}"
        if save_schedule:
            save_schedule_filename = f"../node_schedules/{name}.csv" if save_schedule_filename is None \
                else save_schedule_filename
            ns.save_node_schedule(filename=save_schedule_filename)

        if save_metrics:
            save_metrics_name = name if save_metrics_name is None else save_metrics_name
            save_metrics_filename = "../node_schedule_results.csv" if save_metrics_filename is None \
                else save_metrics_filename
            ns.save_success_metrics(name=save_metrics_name, filename=save_metrics_filename, role=role,
                                    type=schedule_type, network_schedule=network_schedule, dataset=dataset,
                                    solve_time=end - start)
    elif status() is UNKNOWN:
        print("\n The solver cannot find a solution. The problem is probably too large.")
    elif status() is UNSAT:
        print("\nNo feasible node schedule can be found. "
              "Consider making the length of node schedule longer or finding a better network schedule.")
    else:
        print("\nSomething else went wrong.")
    print("\nTime taken to finish: %.4f seconds" % (end - start))
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
    parser.add_argument('-ns', '--network_schedule', required=False, type=str, default=None,
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

    args, unknown = parser.parse_known_args()

    dataset = get_dataset(args.dataset, args.n_sessions,
                          args.n_bqc_sessions, args.n_qkd_sessions, args.n_pp_sessions)

    if args.opt:
        schedule_type = "OPT"
    elif args.naive:
        schedule_type = "NAIVE"
    else:
        schedule_type = "HEU"

    for r in ["alice", "bob"] if args.role is None else [args.role]:
        create_node_schedule(dataset=dataset, role=r, network_schedule=args.network_schedule,
                             schedule_type=schedule_type, save_schedule=args.save_schedule,
                             save_schedule_filename=args.save_schedule_filename, save_metrics=args.save_metrics,
                             save_metrics_name=args.save_metrics_name, save_metrics_filename=args.save_metrics_filename,
                             dataset_id=args.dataset)
