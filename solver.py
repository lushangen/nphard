import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

from student_utils import *
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

    if start in home_indexes:
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
