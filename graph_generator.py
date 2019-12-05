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

def branching_graph(numvert, start, adj, adj_bool):
    if not adj_bool:
        for x in range(numvert):
            row = [0]*numvert
            adj.append(row)
    else:
        for x in adj:
            lst = [0] * numvert
            x += lst
        for x in range(numvert):
            n = numvert*2
            lst = [0] * n
            #print(len(lst))
            adj.append(lst)
    #print(len(adj))


    q = Queue(maxsize = numvert)
    q2 = Queue(maxsize = numvert)
    q3 = Queue(maxsize = numvert)


    for i in range(start+1,start+numvert):
        q.put(i)
    q3.put(start)

    while not q.empty():
        numEdges = r.randint(1, numvert//10)
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

    adj[0][start] = 2
    adj[start][0] = 2
    adj[55][start+15] = 2
    adj[start+15][55] = 2
    adj[32][start+20] = 2
    adj[start+20][32] = 2
    adj[82][start+73] = 2
    adj[start+73][82] = 2
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
        numEdge = r.randint(1, numvert//5)

        fromVert = q.get()
        for c in range(numEdge):
            toVert = r.randint(0, numvert - 1)
            while toVert == fromVert:
                toVert = r.randint(0, numvert - 1)
            edge = r.randint(1,10)

            if adj[fromVert][toVert] == 0:
                G, message = adjacency_matrix_to_graph(adj)
                if not is_metric(G):
                    print_graph(G)
                dist = dict(nx.floyd_warshall(G))
                dist2 = dist[fromVert][toVert]
                if dist2 > 0:
                    if not (dist2 == float('inf')):
                        edge = dist2-1
                    #print("edge: ", edge)

                if edge != 0:

                    q.put(toVert)
                    adj[fromVert][toVert] = edge
                    adj[toVert][fromVert] = edge
                    unioner.union(toVert,fromVert)
                    G, message = adjacency_matrix_to_graph(adj)
                    while not is_metric(G):
                        adj[fromVert][toVert] = adj[fromVert][toVert] + 1
                        adj[toVert][fromVert] = adj[toVert][fromVert] + 1
                        G, message = adjacency_matrix_to_graph(adj)

                        """
                        print_graph(G)
                        print("this edge is bad: ")
                        print(edge)
                        """

        if q.empty():
            q.put(r.randint(0, numvert - 1))
    return adj


def valid_graph(sd, numvert):
    r.seed(sd)
    A = random_connected_graph(sd, numvert)
    G, message = adjacency_matrix_to_graph(A)
    #print_graph(G)
    if not is_metric(G):
        print("you fucked up")
    """
    else:
        nx.draw(G)
        plt.show()
    """

        #nx.draw(G)
        #plt.show()

        #A = random_connected_graph(r.randint(0, 100000), numvert)
        #G, message = adjacency_matrix_to_graph(A)
    return A
    #as adjacency matrix
#A = valid_graph(3, 10)
nums = 200
A = valid_graph(42, 100)
G2 = branching_graph(100, nums//2, A, True)
G, msg = adjacency_matrix_to_graph(G2)
nx.draw(G, with_labels = True)
plt.show()
#nx.draw(H)
#plt.show()
#print(is_metric(G2))


#print(G)
#for i in range(5):
    #print(G[i])
print(is_metric(G))
"""
nums = 200
G = branching_graph(nums)
"""
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



for i in range(nums):
    for x in range(len(G2[i])):
        if G2[i][x] == 0:
            if (x == len(G2[i])-1):
                print("x")
            else:
                print("x", end = " ")
        else:
            if (x == len(G2[i])-1):
                print(G2[i][x])
            else:
                print(G2[i][x], end = " ")
