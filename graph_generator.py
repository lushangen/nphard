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
import matplotlib.pyplot as plt

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
            edge = r.randint(1,5)
            if adj[fromVert][toVert] == 0:
                G, message = adjacency_matrix_to_graph(adj)
                dist = dict(nx.floyd_warshall(G))
                if dist[fromVert][toVert] > 0 :
                    while edge >= dist[fromVert][toVert]:
                        edge = edge - 1
             
                if edge != 0: 
                    q.put(toVert)
                    adj[fromVert][toVert] = edge
                    adj[toVert][fromVert] = edge
                    unioner.union(toVert,fromVert)
                    
    return adj

def valid_graph(sd, numvert): 
    r.seed(sd)
    A = random_connected_graph(sd, numvert)
    G, message = adjacency_matrix_to_graph(A)
    #print_graph(G) 
    if not is_metric(G):
        print("you fucked up")
        nx.draw(G)
        plt.show()
        #A = random_connected_graph(r.randint(0, 100000), numvert)
        #G, message = adjacency_matrix_to_graph(A)
    return G
    #as adjacency matrix
#G = valid_graph(3, 10)
G = valid_graph(4, 50)
print_graph(G)
#print(G)
#for i in range(5):
    #print(G[i])
