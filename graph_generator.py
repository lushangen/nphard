import networkx as nx
import numpy as np
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from queue import Queue
from student_utils import *
import random as r
from UF import *
import scipy

def random_connected_graph(sd, numvert):
    r.seed(sd)
    adj = []
    unioner = UF(numvert)
    for x in range(numvert):
        row = []
        for y in range(numvert):
            row.append(0)    
        adj.append(row)
            
    q = Queue(maxsize = numvert*numvert) 
    q.put(0)
    
    #While weighted union is not size n
    while unioner.sizeOfIndex(0) < numvert: 
        numEdge = r.randint(1, numvert//3)
        fromVert = q.get()
        for c in range(numEdge):
            toVert = r.randint(0, numvert - 1)
            while toVert == fromVert:
                toVert = r.randint(0, numvert - 1)
            q.put(toVert)
            unioner.union(toVert,fromVert)
            adj[fromVert][toVert] = r.randint(1,5)
            adj[toVert][fromVert] = r.randint(1,5)
    return adj
def valid_graph(sd, numvert): 
    r.seed(sd)
    A = random_connected_graph(sd, numvert)
    G = adjacency_matrix_to_graph(A)
    print_graph(G) 
    while not is_metric(G):
        A = random_connected_graph(r.randint(0, 100000), numvert)
        G = adjacency_matrix_to_graph(A)
    return A
    #as adjacency matrix
#G = valid_graph(3, 10)
G = random_connected_graph(4, 5)
#print(G)
for i in range(5):
    print(G[i])
