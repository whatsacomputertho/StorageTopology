from StorageTopology import Node, Edge, Object, StorageTopology
import sympy as sp
import matplotlib.pyplot as plt
import csv, math, random, numpy, networkx
from itertools import combinations

#Specify latencies & capacities
caps = [1, 1, 1, 1]
lats = [129.1, 246.0, 132.42, 122.1, 194.8, 100.5]

#Check every replication scheme
topology = StorageTopology(capacities=caps, num_nodes=4, num_objects=3, weights=lats)
allocations = topology.filter_feasible_replicative_allocations(topology.generate_all_replicative_allocations())
avg_lat = numpy.inf
alloc = []
for allocation in allocations:
    topology.allocate_coded_objects(allocation)
    latency = topology.calculate_average_coded_latency()
    if latency < avg_lat:
        avg_lat = latency
        alloc = topology.get_allocation()
        print(alloc)