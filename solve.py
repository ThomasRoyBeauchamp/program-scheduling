from pycsp3 import *
from node_schedule import NodeSchedule
from activity_metadata import ActiveSet

if __name__ == '__main__':
    # TODO: make this input into a command-line argument
    active = ActiveSet.create_active_set(["../session_configs/qkd.yaml"], [[1, 2, 3, 4, 5]])

    schedule_size = 30  # TODO: how to define schedule size
    capacities = [1, 1]

    # x[i] is the starting time of the ith job
    x = VarArray(size=active.n_blocks, dom=range(schedule_size))

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
        [cumulative_for(k) <= capacity for k, capacity in enumerate(capacities)]
        # TODO: constraints for min and max time lags
    )

    # optional objective function
    if variant("makespan"):
        minimize(
            Maximum([x[i] + active.durations[i] for i in range(active.n_blocks)])
        )

    instance = compile()
    ace = solver(ACE)

    # TODO: this is where you define heuristics I think
    result = ace.solve(instance)

    if status() is SAT:
        ns = NodeSchedule(active.n_blocks, solution().values, active.durations, active.resource_reqs)
        ns.print()
