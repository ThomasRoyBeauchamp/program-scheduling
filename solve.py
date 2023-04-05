import time

from pycsp3 import *
from node_schedule import NodeSchedule
from activity_metadata import ActiveSet
from network_schedule import NetworkSchedule

if __name__ == '__main__':
    # TODO: network_schedule and input for create_active_set as command line arguments
    # network_schedule = NetworkSchedule([1, 10], [3, 3], [1, 2])
    network_schedule = NetworkSchedule()

    # active = ActiveSet.create_active_set(["../session_configs/teleportation.yaml"], [[1, 2]], network_schedule)
    # active = ActiveSet.create_active_set(["../session_configs/qkd.yaml", "../session_configs/bqc-client.yaml"],
    #                                      [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    dataset = {  # TODO: use this for argument to create_active_set
        "../configs/pingpong_alice.yml": 30
    }
    active = ActiveSet.create_active_set(["../configs/pingpong_alice.yml"], [list(range(30))],
                                         network_schedule)
    active.scale_down()
    """
        TODO: How to define length of node_schedule?
        If it's too low, there might not be any feasible solution. 
        If it's too high, we are unnecessarily solving a more complex problem. 
    """
    print(sum(active.durations))
    schedule_size = 2 * int(sum(active.durations))
    capacities = [1, 1]  # capacity of [CPU, QPU]

    # x[i] is the starting time of the ith job
    x = VarArray(size=active.n_blocks, dom=range(schedule_size))
    # TODO: already set the initial domains more efficiently??

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
        # constraint for min time lags
        # TODO: this does not allow for concurrent executions of sessions of teleportation, why
        # [active.d_min[i+1] <= (x[i+1] - (x[i] + active.durations[i])) for i in range(active.n_blocks - 1)],
        # network-schedule constraints (all quantum communication blocks adhere to network schedule if it's defined)
        # TODO: this needs to be fixed when we allow for multiple QC blocks in a session (also rescale)
        # [(x[i] == network_schedule.get_session_start_time(active.ids[i]) for i in range(active.n_blocks - 1)
        #   if network_schedule.is_defined and active.types[i] == "QC")]
    )

    # optional objective function
    if variant("makespan"):
        minimize(
            Maximum([x[i] + active.durations[i] for i in range(active.n_blocks)])
        )

    instance = compile()
    ace = solver(ACE)

    # https://github.com/xcsp3team/pycsp3/blob/master/docs/optionsSolvers.pdf
    # heuristics = {"valh": "max"}
    heuristics = {}

    print(f"\nTrying to construct a node schedule of length {schedule_size}")
    start = time.time()
    result = ace.solve(instance, dict_options=heuristics)
    end = time.time()

    if status() is SAT:
        active.scale_up()
        start_times = [s * active.gcd for s in solution().values]
        ns = NodeSchedule(active, start_times)

        ns.print()
        # TODO: make into a CL argument
        save_node_schedule = False
        if save_node_schedule:
            # TODO: make into a CL argument
            name = "temp_results"
            # TODO: make filename into a CL argument
            ns.save_success_metrics(name, filename="../node_schedule_results.csv", network_schedule=network_schedule,
                                    dataset=dataset, solve_time=end-start)

        # TODO: make into a CL argument
        save_node_schedule = False
        if save_node_schedule:
            ns.save_node_schedule("../node_schedules/temp_pingpong_bob_12.csv")

        print("\nTime taken to finish: %.4f seconds" % (end - start))
    else:
        print("\nNo feasible node schedule was found. "
              "Consider making the length of node schedule longer or finding a better network schedule.")
        print("\nTime taken to finish: %.4f seconds" % (end - start))
