
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time


def create_data_model(map_size, num_vehicles, depot, not_scanned_map):
    """Stores the data for the problem."""
    data = {}
    data['num_vehicles'] = num_vehicles    # vehicle/drone number
    data['depot'] = depot           # base station index
    data['target_map_size'] = map_size

    baseStation_x = 0
    baseStation_y = 0
    cell_number = map_size * map_size + 1  # including base station
    map_x_start = baseStation_x+1
    map_x_end = map_size+1
    map_y_start = baseStation_y+1
    map_y_end = map_size+1

    # distances
    # In this test file, locations are tuple
    location_set = [(baseStation_x, baseStation_y)]
    for x in range(map_x_start, map_x_end):
        for y in range(map_y_start, map_y_end):
            # trajectory only consider cells not scanned, not_scanned_map index start from 0
            if not_scanned_map[x-map_x_start, y-map_y_start]:
                location_set.append((x, y))
    data['locations'] = location_set

    # calculate distance
    # distance_set = np.zeros((cell_number, cell_number))
    # # TODO: solve distance not exist problem when some cells are scanned (create path only depend on given cells)
    # # print('shape', distance_set.shape)
    # # print("distance_set", distance_set)
    # i_index = 0
    # for i_x, i_y in location_set:
    #     j_index = 0
    #     for j_x, j_y in location_set:
    #         # if not_scanned_map[i_index, j_index]:  # trajectory only consider cells not scanned
    #         distance_set[i_index][j_index] = np.linalg.norm([i_x - j_x, i_y - j_y])
    #         j_index += 1
    #     i_index += 1
    #
    # # since the libarary will convert float to integer, Multiple 100 here for keeping two decimals.
    # data['distance_matrix'] = distance_set * 100
    return data

def compute_euclidean_distance_matrix(locations, variable_precision):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))*variable_precision))   # multiple variable_precision for keeping two decimals
    return distances



def print_solution(data, manager, routing, solution, variable_precision):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()/variable_precision}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            # print("distance", manager.IndexToNode(index), manager.IndexToNode(previous_index), data['distance_matrix'][manager.IndexToNode(index)][manager.IndexToNode(previous_index)])
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance/variable_precision)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance/variable_precision))


def draw_map(data, manager, routing, solution):
    location_set = data['locations']
    map_size = data['target_map_size']
    colors_order = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for vehicle_id in range(data['num_vehicles']):
        # print("vehicle_id", vehicle_id)
        # color = (random.random(), random.random(), random.random())

        index = routing.Start(vehicle_id)
        route_distance = 0

        [i_x, i_y] = [None, None]
        while not routing.IsEnd(index):
            # print("IndexToNode", manager.IndexToNode(index))
            # print("[i_x, i_y]", [i_x, i_y])
            [j_x, j_y] = location_set[manager.IndexToNode(index)]
            if i_x is not None:
                plt.arrow(i_x, i_y, j_x - i_x, j_y - i_y, width=0.05, color = colors_order[vehicle_id], length_includes_head=True)
            [i_x, i_y] = [j_x, j_y]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # print("IndexToNode", manager.IndexToNode(index))
        # print("[i_x, i_y]", [i_x, i_y])
        [j_x, j_y] = location_set[manager.IndexToNode(index)]
        if i_x is not None:
            plt.arrow(i_x, i_y, j_x - i_x, j_y - i_y, width=0.05, color = colors_order[vehicle_id], length_includes_head=True)
        plt.plot(j_x, j_y, 'ro')
        # print(solution.Value(routing.NextVar(index)))
        # print("route_distance", route_distance)

    # draw grid
    for i in range(map_size + 1):
        plt.axvline(x=i + 0.5, linewidth=0.5, color='gray')
        plt.axhline(y=i + 0.5, linewidth=0.5, color='gray')

    plt.xlim(-0.5, map_size + 1 + 0.5)
    plt.ylim(-0.5, map_size + 1 + 0.5)
    plt.show()




def generate_route_array(data, manager, routing, solution):
    location_set = data['locations']
    route_array = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_array[vehicle_id] = np.array([location_set[manager.IndexToNode(index)]]).astype(float)
        while not routing.IsEnd(index):
            # print("location_set[manager.IndexToNode(index)]", location_set[manager.IndexToNode(index)])
            # print("route_array[vehicle_id]", route_array[vehicle_id])
            index = solution.Value(routing.NextVar(index))
            route_array[vehicle_id] = np.concatenate((route_array[vehicle_id],[location_set[manager.IndexToNode(index)]]), axis=0)


        # route_array[vehicle_id] = route_array[vehicle_id][1:]

    return route_array





def MD_path_plan_main(num_MD, map_size, not_scanned_map):
    """Entry point of the program."""

    # target_map_size = 5
    num_vehicles = num_MD  # vehicle/drone number
    depot = 0  # base station index
    variable_precision = 100 # 100 means keep two decimal

    # Instantiate the data problem.
    data = create_data_model(map_size, num_vehicles, depot, not_scanned_map)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # create distance for all nodes.
    distance_matrix = compute_euclidean_distance_matrix(data['locations'], variable_precision)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
        # return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        (map_size*variable_precision)*(map_size*variable_precision),  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(map_size*variable_precision)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC) # algorithm may change here)

    # Setting second solution
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    # search_parameters.time_limit.seconds = 100
    # search_parameters.log_search = True

    # Solve the problem (with time counter).
    start = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    end = time.time()
    print("\ntime spent:{:.2f}\n".format(end - start))

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution, variable_precision)
        draw_map(data, manager, routing, solution)
    else:
        print('No solution found !')

    return generate_route_array(data, manager, routing, solution)


if __name__ == '__main__':
    num_MD = 3
    map_size = 6
    not_scanned_map = np.ones((map_size,map_size), dtype=bool)
    not_scanned_map[0,5] = False
    not_scanned_map[5, 0] = False
    print(not_scanned_map)
    MD_path_plan_main(num_MD, map_size, not_scanned_map)


