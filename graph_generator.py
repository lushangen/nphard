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

def branching_graph(numvert):
    adj = []
    for x in range(numvert):
        row = [0]*numvert
        adj.append(row)

    q = Queue(maxsize = numvert)
    q2 = Queue(maxsize = numvert)
    q3 = Queue(maxsize = numvert)


    for i in range(1,numvert):
        q.put(i)
    q3.put(0)

    while not q.empty():
        numEdges = r.randint(1, numvert//10)
        set = {}
        fromVert = q3.get()

        for i in range(numEdges):
            if q.empty():
                break
            else:
                y = q.get()
                q2.put(y)
                q3.put(y)
        while not q2.empty():
            toVert = q2.get()
            length =  r.randint(1,5)
            #print(fromVert, toVert)
            adj[fromVert][toVert] = length
            adj[toVert][fromVert] = length
    return adj



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
    G, msg = adjacency_matrix_to_graph(A)
    #print_graph(G)
    while not is_metric(G):
        A = random_connected_graph(r.randint(0, 100000), numvert)
        G, msg = adjacency_matrix_to_graph(A)
    return A
    #as adjacency matrix

#G = valid_graph(2, 50)
"""
G = nx.full_rary_tree(3,10)
G = nx.adjacency_matrix(G)
G = adjacency_matrix_to_graph(G)
print_graph(G)
print(is_metric(G))
"""
nums = 200
G = branching_graph(nums)
print(nums)
print(nums//2)
for i in range(nums):
    print(i, end = " ")
print()
rv = random_homes(nums, nums//2)
for i in rv:
    print(i, end = " ")
print()
print("0")

"""
for i in range(25):
    print(G[i])
"""
for i in range(nums):
    for x in range(len(G[i])):
        if G[i][x] == 0:
            if (x == len(G[i])-1):
                print("x")
            else:
                print("x", end = " ")
        else:
            if (x == len(G[i])-1):
                print(G[i][x])
            else:
                print(G[i][x], end = " ")


G, msg = adjacency_matrix_to_graph(G)

nx.draw(G)
plt.show()
#G = random_connected_graph(4, 10)
