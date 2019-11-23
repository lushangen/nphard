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
    dist = []
    unioner = UF(numvert)
    for x in range(numvert):
        row = []
        row2 = []
        for y in range(numvert):
            row.append(0)    
            row2.append(0)
        adj.append(row)
        dist.append(row2)
            
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
                if dist[fromVert][toVert] > 0 :
                    while edge >= dist[fromVert][toVert]:
                        edge = edge - 1
             
                if edge != 0: 
                    for i in range(numvert):
                        distA = dist[i][fromVert]
                        distB = dist[i][toVert]
                                            #start
                        for x in range(numvert): 
                            xi = dist[x][i]
                            xfV = dist[fromVert][x]
                            xtV = dist[toVert][x]  
                            if distA > 0:
                                if distB > 0:
                                    if xi > 0:
                                        if xfV > 0:
                                            if xtV > 0:
                                                dist[x][i] = min(xi, xfV + distB + edge, distA + xfV, distB + xtV)
                                                dist[i][x] = min(xi, xfV + distB + edge, distA + xfV, distB + xtV)
                                            else: 
                                                dist[i][x] = min(xi, xfV + distB + edge, distA + xfV)
                                                dist[x][i] = min(xi, xfV + distB + edge, distA + xfV)
                                        else: 
                                            if xtV > 0:
                                                dist[i][x] = min(xi,  distB + xtV)
                                                dist[x][i] = min(xi,  distB + xtV)
                                    else: 
                                        if xfV > 0:
                                            if xtV > 0:
                                                dist[x][i] = min(xfV + distB + edge, distA + xfV, distB + xtV)
                                                dist[i][x] = min(xfV + distB + edge, distA + xfV, distB + xtV)
                                            else: 
                                                dist[x][i] = min(xfV + distB + edge, distA + xfV)
                                                dist[i][x] = min(xfV + distB + edge, distA + xfV)
                                        else: 
                                            if xtV > 0:
                                                dist[x][i] = distB + xtV
                                                dist[i][x] = distB + xtV
                                else:
                                    if xi > 0:
                                        if xfV > 0:
                                            if xtV > 0:
                                                dist[x][i] = min(xi, distA + xfV)
                                                dist[i][x] = min(xi, distA + xfV)
                                            else: 
                                                dist[i][x] = min(xi, distA + xfV)
                                                dist[x][i] = min(xi, distA + xfV)
                                    else: 
                                        if xfV > 0:
                                            dist[x][i] = distA + xfV
                                            dist[i][x] = distA + xfV
                            else:
                                if distB > 0:
                                    if xi > 0:
                                        if xfV > 0:
                                            if xtV > 0:
                                                dist[x][i] = min(xi, xfV + distB + edge, distB + xtV)
                                                dist[i][x] = min(xi, xfV + distB + edge, distB + xtV)
                                            else: 
                                                dist[i][x] = min(xi, xfV + distB + edge)
                                                dist[x][i] = min(xi, xfV + distB + edge)
                                        else: 
                                            if xtV > 0:
                                                dist[i][x] = min(xi,  distB + xtV)
                                                dist[x][i] = min(xi,  distB + xtV)
                                    else: 
                                        if xfV > 0:
                                            if xtV > 0:
                                                dist[x][i] = min(xfV + distB + edge, distB + xtV)
                                                dist[i][x] = min(xfV + distB + edge, distB + xtV)
                                            else: 
                                                dist[x][i] = xfV + distB + edge
                                                dist[i][x] = xfV + distB + edge
                                        else: 
                                            if xtV > 0:
                                                dist[x][i] = distB + xtV
                                                dist[i][x] = distB + xtV
                    #end
                    """
                            dist[i][fromVert] = min(dist[i][fromVert], dist[i][toVert] + edge)
                            dist[i][toVert] = min(dist[i][toVert], dist[i][fromVert] + edge)
                            dist[fromVert][i] = min(dist[i][fromVert], dist[i][toVert] + edge)
                            dist[toVert][i] = min(dist[i][toVert], dist[i][fromVert] + edge)
                    """
                    q.put(toVert)
                    adj[fromVert][toVert] = edge
                    adj[toVert][fromVert] = edge
                    unioner.union(toVert,fromVert)
                    dist[fromVert][toVert] = edge
                    dist[toVert][fromVert] = edge
                    
    return adj

def valid_graph(sd, numvert): 
    r.seed(sd)
    A = random_connected_graph(sd, numvert)
    G, message = adjacency_matrix_to_graph(A)
    #print_graph(G) 
    while not is_metric(G):
        print("you fucked up")
        A = random_connected_graph(r.randint(0, 100000), numvert)
        G, message = adjacency_matrix_to_graph(A)
    return G
    #as adjacency matrix
#G = valid_graph(3, 10)
G = valid_graph(4, 50)
print_graph(G)
#print(G)
#for i in range(5):
    #print(G[i])
