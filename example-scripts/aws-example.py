from StorageTopology.StorageTopology import StorageTopology
import sympy as sp
import matplotlib.pyplot as plt
import csv, math, random, numpy, networkx
from itertools import combinations

#Specify latencies & capacities
caps = [1, 1, 1, 1]

#Note that edges are not directed
#Instead took average of directed edges between nodes
lats = [129.1, 246.0, 132.42, 122.1, 194.8, 100.5]

#Instantiate the network
topology = StorageTopology(capacities=caps, num_nodes=4, num_objects=3, weights=lats)

#Generate all allocations
allocations = topology.generate_feasile_replicative_allocations()

#Instantiate average latency variable and allocation
avg_lat = numpy.inf
alloc = []

#Loop through all allocations and find the optimal one with regard to average latency
#Finally print the optimal allocation
for allocation in allocations:
    topology.allocate_coded_objects(allocation)
    latency = topology.calculate_global_average_latency()
    if latency < avg_lat:
        avg_lat = latency
        alloc = topology.get_allocation()
        print(alloc)