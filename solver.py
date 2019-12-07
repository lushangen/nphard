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
def compute_centrality(v, home_indexes, shortest_paths):
    numHomes = len(home_indexes)
    sum = 0
    vpath = shortest_paths[v][0]
    for home in home_indexes:
        sum += vpaths[home]
    return (n-1)/sum

def compute_group(v, home_indexes, shortest_paths, epsilon):
    #Could home_indexes be empty?
    #REMOVES home if passed in
    if not home_indexes:
        return 0, []
    numHomes = len(home_indexes)
    sum = 0
    vpath = shortest_paths[v][0]
    minDist = vpath[home_indexes[0]]
    homeList = [home_indexes[0]]
    if v in home_indexes:
        home_indexes.remove(v)
    for home in home_indexes:
        if vpath[home] < minDist:
            minDist = vpath[home]
            homeList = [home]
    for home in home_indexes:
        homeDist = vpath[home]
        if homeDist <= minDist * epsilon:
            sum += homeDist
            homeList.append(home)
    return sum/(pow(1.3, len(homeList))*len(homeList)), homeList[1:]

def solve_tsp_grouped(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """Entry point of the program."""
    drop_off_dict = {}
    car_path = []
    home_map = {}
    group_score = {}
    center_map = {}
    epsilon = 1.2
    home_indexes = convert_locations_to_indices(list_of_homes, list_of_locations)
    orig_home_indexes = home_indexes[:]
    cluster_map = {}

    start = list_of_locations.index(starting_car_location)
    graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    all_paths = dict(nx.all_pairs_dijkstra(graph))

    for v in range(len(list_of_locations)):
        group_score[v], cluster_map[v] = compute_group(v, home_indexes[:], all_paths, epsilon)

    sorted_v = sorted([k for k in group_score.keys() if group_score[k] > 0], key = lambda x: group_score[x])
    min_group_score = group_score[sorted_v[0]]
    #print(min_group_score)
    delta = 6

    high_centrality_homes = set()  #LOW GROUP SCORE VERTICES (CLUSTER)
    used_homes = set()
    newHome = home_indexes[:]

    while newHome and group_score[sorted_v[0]] < delta * min_group_score:

        v = sorted_v[0]
        high_centrality_homes.add(v)
        usedList = cluster_map[v]
        #print(usedList)
        if v in newHome and v not in usedList:
            usedList.append(v)
        if v in center_map.keys():
            center_map[v].extend(usedList)
        else:
            center_map[v] = usedList
            #if v in home_indexes:
                #center_map[v].append(v)
        for vert in usedList:
            used_homes.add(vert)
            if vert in newHome:
                newHome.remove(vert)
        used_homes.add(v)
        while v in newHome:
            newHome.remove(v)
        for x in range(len(list_of_locations)):
            group_score[x], cluster_map[x] = compute_group(x, newHome[:], all_paths, epsilon)
        sorted_v = sorted([k for k in group_score.keys() if group_score[k] > 0], key = lambda x: group_score[x])

    for home in newHome:
        if home not in center_map.keys():
            center_map[home] = [home]
    """START TIM

    for home in high_centrality_homes:
        center_map[home] = [home]
        dist_dict = all_paths.get(home)[0]
        paths_dict = all_paths.get(home)[1]
        dist_dict = {k:v for (k,v) in dist_dict.items() if k not in used_homes and k in home_indexes} #distance dict of all remaing homes

        min_dist = min(dist_dict.values()) #closest home to high centrality home
        dist_dict = {k:v for (k,v) in dist_dict.items() if dist_dict[k] <= min_dist*epsilon}

        for cluster_home in dist_dict.keys():
            center_map[home].append(cluster_home)
            home_indexes.remove(cluster_home)
            used_homes.add(cluster_home)

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
    END TIM """
    tspInput = list(high_centrality_homes)
    tspInput.extend(newHome)
    if start in tspInput:
        tspInput.remove(start)
    tspInput.insert(0, start)
    home_map.clear()
    for i in range(len(tspInput)):
        home_map[i] = tspInput[i]
    data = create_data_model(tspInput, 0)

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
    search_parameters.time_limit.seconds = 3
    #search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # if assignment:
    #     print_solution(manager, routing, assignment)
    # Print solution on console.

    #if start in home_indexes:
    #    drop_off_dict[start] = [start]


    index = routing.Start(0)
    car_path.append(start)

    while not routing.IsEnd(index):
        previous_index = manager.IndexToNode(index)
        index = assignment.Value(routing.NextVar(index))
        to_index = manager.IndexToNode(index)
        car_path.append(home_map[to_index])
        #path_to = all_paths.get(home_map[previous_index])[1][home_map[to_index]]
        #drop_off_dict[home_map[to_index]] = [home_map[to_index]]
        #print(to_index, end = ' ')
        #car_path.extend(path_to)
        #route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    # for i in car_path:
    #      print(i)

    #print(car_path)
    drop_off_dict = center_map
    new_path = [start]
    previous_index = start


    #print(center_map)
    for to_index in car_path[1:]:
        new_path.pop()
        path_to = all_paths.get(previous_index)[1][to_index]
        new_path.extend(path_to)
        previous_index = to_index
    keys = drop_off_dict.keys()
    singleadd = []
    for i in keys:
        if i not in orig_home_indexes:
            vals = drop_off_dict[i]
            for v in vals:
                if v in new_path:
                    if v in keys:
                        drop_off_dict[v].append(v)
                        drop_off_dict[i].remove(v)
                    else:
                        singleadd.append(v)
                        drop_off_dict[i].remove(v)
    for item in singleadd:
        drop_off_dict[item] = [item]

    for i in drop_off_dict.keys():
        for k in drop_off_dict.keys():
            if i != k:
                if i in drop_off_dict[k]:
                    drop_off_dict[k].remove(i)
                    drop_off_dict[i].append(i)

    removeIt = set()
    for i in drop_off_dict.keys():
        if not drop_off_dict[i]:
            removeIt.add(i)

    for i in removeIt:
        drop_off_dict.pop(i, None)

    car_path = new_path
    return car_path, drop_off_dict



def centrality_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    loc_map = {}
    drop_off_dict = {}
    num_home_visited = 0
    for i in range(len(list_of_locations)):
        loc_map[i] = list_of_locations[0]


    home_indexes = convert_locations_to_indices(list_of_homes, list_of_locations)
    start = list_of_locations.index(starting_car_location)
    graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    num_homes = len(list_of_homes)

    car_path = []
    all_paths = dict(nx.all_pairs_dijkstra(graph))
    visited = set()
    visited.add(start)
    #print(start)
    car_path.append(start)
    current_node = start

    cent_map = nx.closeness_centrality(graph);
    max = 0;
    maxnode = 0;
    for k in cent_map.keys():
        if cent_map[k] > max:
            max = cent_map[k]
            maxnode = k

    paths_dict = all_paths.get(start)[1]
    car_path = paths_dict[maxnode]
    rev = car_path[:]
    rev.reverse()
    car_path.extend(rev[1:])
    print(car_path)
    drop_off_dict[maxnode] = home_indexes

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
    search_parameters.time_limit.seconds = 3
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

    #car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    #car_path2, drop_offs2 = solve_tsp_grouped(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    #car_path3, drop_offs3 = solve_tsp(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    car_path, drop_offs = solve_tsp_grouped(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    #ad_graph, msg = adjacency_matrix_to_graph(adjacency_matrix)
    #cost2, msg2 = cost_of_solution(ad_graph, car_path2, drop_offs2)
    #cost1, msg1 = cost_of_solution(ad_graph, car_path, drop_offs)
    #cost3, msg3 = cost_of_solution(ad_graph, car_path3, drop_offs3)
    #print(cost2)
    #print(cost1)
    #print(cost3)
    """
    if cost2 < cost1:
       car_path = car_path2
        drop_offs = drop_offs2
    """
    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)

def tests(input_data, output_data, params=[]):
        number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix = data_parser(input_data)
        message = ''
        cost = -1
        car_cycle = output_data[0]
        num_dropoffs = int(output_data[1][0])
        targets = []
        dropoffs = {}
        for i in range(num_dropoffs):
            dropoff = output_data[i + 2]
            dropoff_index = list_of_locations.index(dropoff[0])
            dropoffs[dropoff_index] = convert_locations_to_indices(dropoff[1:], list_of_locations)

        G, msg = adjacency_matrix_to_graph(adjacency_matrix)
        car_cycle = convert_locations_to_indices(car_cycle, list_of_locations)
        cost, message = cost_of_solution(G, car_cycle, dropoffs)

        return cost, message, car_cycle, dropoffs


def solve_from_file_2(input_file, output_file2, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    output_data = utils.read_file(output_file2)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    #car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    #car_path2, drop_offs2 = solve_tsp_grouped(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    #car_path3, drop_offs3 = solve_tsp(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    car_path, drop_offs = solve_tsp_grouped(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    ad_graph, msg = adjacency_matrix_to_graph(adjacency_matrix)

    cost1, msg1 = cost_of_solution(ad_graph, car_path, drop_offs)
    cost2, msg2, car_path2, drop_offs2 = tests(input_data, output_data)

    #cost2, msg2 = cost_of_solution(ad_graph, car_path2, drop_offs2)
    #cost1, msg1 = cost_of_solution(ad_graph, car_path, drop_offs)
    #cost3, msg3 = cost_of_solution(ad_graph, car_path3, drop_offs3)
    #print(cost2)
    #print(cost1)
    #print(cost3)

    if cost2 < cost1:
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
        temp_input = input_file[:]
        output_file = input_file.replace('in', 'out')
        input_file = temp_input
        solve_from_file_2(input_file, output_file, output_directory, params=params)


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
