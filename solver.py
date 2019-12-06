import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

from student_utils import *

#Google OR Tools
#from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

"""
======================================================================
  Complete the following function.
======================================================================
"""


def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    loc_map = {}
    drop_off_dict = {}
    num_home_visited = 0

    """
    for i in range(len(list_of_locations)):
        loc_map[i] = list_of_locations[0]
    """

    home_indexes = convert_locations_to_indices(list_of_homes, list_of_locations)
    start = list_of_locations.index(starting_car_location)
    graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    num_homes = len(list_of_homes)

    car_path = []
    all_paths = dict(nx.all_pairs_dijkstra(graph))
    visited = set()

    #print(start)
    car_path.append(start)
    current_node = start

    if start in home_indexes:
        visited.add(start)
        drop_off_dict[start] = [start]
        num_home_visited += 1

    while num_home_visited < num_homes:
        dist_dict = all_paths.get(current_node)[0]
        paths_dict = all_paths.get(current_node)[1]

        dist_dict = {k:v for (k,v) in dist_dict.items() if k not in visited and k in home_indexes}
        min_dist = min(dist_dict.values())
        min_list = [k for k in dist_dict.keys() if dist_dict[k] <= min_dist]
        #print(dist_dict.values())
        target = min_list[0]
        drop_off_dict[target] = [target]
        #print(target+1)
        #print(target)
        car_path.pop()
        car_path.extend(paths_dict[target])

        visited.add(target)
        current_node = target
        num_home_visited += 1

    paths_dict = all_paths.get(current_node)[1]
    car_path.pop()
    car_path.extend(paths_dict[start])
    #print((drop_off_dict.keys()))
    #car_path = [start, ...., start]
    #drop_off_dict = {drop_off_loc: [home1, home2, ...] }

    return car_path, drop_off_dict


def create_data_model(list_of_homes, starting_location):
    """Stores the data for the problem."""
    data = {}
    # Locations in block units
    #print(list_of_homes)
    data['locations'] = list_of_homes
    data['num_vehicles'] = 1
    data['depot'] = starting_location
    return data

def solve_tsp(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """Entry point of the program."""
    drop_off_dict = {}
    car_path = []
    home_map = {}
    home_indexes = convert_locations_to_indices(list_of_homes, list_of_locations)

    start = list_of_locations.index(starting_car_location)
    graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    all_paths = dict(nx.all_pairs_dijkstra(graph))

    start_in_home = start in home_indexes
    if start in home_indexes:
        home_indexes.remove(start)
    home_indexes.insert(0, start)
    home_count = 0;

    for home in home_indexes:
        #print(home, end = " ")
        home_map[home_count] = home
        home_count += 1
    # Instantiate the data problem.
    #print(len(home_map))
    data = create_data_model(home_indexes, 0)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])

    #print(manager.NodeToIndex(15))
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        #print(home_map[to_index], end = " ")
        from_index = manager.IndexToNode(from_index)
        to_index = manager.IndexToNode(to_index)
        dist_to = all_paths.get(home_map[from_index])[0][home_map[to_index]]
        #if from_index >= 25 or to_index >= 25:
        #    print("from" if from_index >= 25 else "to", end = " ")
        #dist_to = all_paths[from_index][0][to_index]
        return dist_to

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    """
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    """

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 8
    #search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # if assignment:
    #     print_solution(manager, routing, assignment)
    # Print solution on console.

    if start in home_indexes:
        drop_off_dict[start] = [start]


    index = routing.Start(0)
    car_path.append(start)

    while not routing.IsEnd(index):
        previous_index = manager.IndexToNode(index)
        index = assignment.Value(routing.NextVar(index))

        car_path.pop();
        to_index = manager.IndexToNode(index)
        path_to = all_paths.get(home_map[previous_index])[1][home_map[to_index]]
        drop_off_dict[home_map[to_index]] = [home_map[to_index]]
        #print(to_index, end = ' ')
        car_path.extend(path_to)
        #route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    # for i in car_path:
    #      print(i)
    if start in drop_off_dict.keys() and not start_in_home:
        drop_off_dict.pop(start, None)

    return car_path, drop_off_dict

def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    print('Objective: {}'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(index)
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Objective: {}m\n'.format(route_distance)

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)

    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    car_path2, drop_offs2 = solve_tsp(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    ad_graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    cost1, msg = cost_of_solution(ad_graph, car_path2, drop_offs2)
    cost2, msg = cost_of_solution(ad_graph, car_path, drop_offs)
    if cost1 < cost2:
        car_path = car_path2
        drop_offs = drop_offs2
    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
