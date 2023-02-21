import time

from pycsp3 import *
from node_schedule import NodeSchedule
from activity_metadata import ActiveSet

if __name__ == '__main__':
    # TODO: make this input into a command-line argument
    active = ActiveSet.create_active_set(["../session_configs/qkd.yaml"], [[1, 2, 3, 4, 5]])
    # active = ActiveSet.create_active_set(["../session_configs/qkd.yaml", "../session_configs/bqc-client.yaml"],
    #                                      [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    schedule_size = 80  # TODO: how to define schedule size
    capacities = [1, 1]

    # x[i] is the starting time of the ith job
    x = VarArray(size=active.n_blocks, dom=range(schedule_size))
    # TODO: already set the initial domains more efficiently??

    # taken from http://pycsp.org/documentation/models/COP/RCPSP/
    def cumulative_for(k):
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
        [(x[i+1] - (x[i] + active.durations[i])) <= active.d_max[i+1] for i in range(active.n_blocks - 1)]
        # constraint for min time lags, TODO: this currently doesn't make sense I think, double-check the time lags
        # [active.d_min[i+1] <= (x[i+1] - (x[i] + active.durations[i])) for i in range(active.n_blocks - 1)]
    )

    # optional objective function
    if variant("makespan"):
        minimize(
            Maximum([x[i] + active.durations[i] for i in range(active.n_blocks)])
        )

    instance = compile()
    ace = solver(ACE)

    # TODO: this is where you define heuristics I think
    start = time.time()
    # https://github.com/xcsp3team/pycsp3/blob/master/docs/optionsSolvers.pdf
    result = ace.solve(instance, dict_options={"valh": "max"})
    # result = ace.solve(instance)
    end = time.time()

    if status() is SAT:
        ns = NodeSchedule(active.n_blocks, solution().values, active.durations, active.resource_reqs)
        ns.print()
        print("\nTime taken to finish: %.4f seconds" % (end - start))
