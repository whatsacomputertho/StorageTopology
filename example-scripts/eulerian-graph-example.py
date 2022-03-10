from StorageTopology import Node, Edge, Object, StorageTopology
import sympy as sp
import matplotlib.pyplot as plt
import csv, math, random, numpy, networkx
from itertools import combinations

#Variables controlling sizes of the cycles
nodes = 0
cycle_lengths = [4, 5]
for i in range(len(cycle_lengths)):
    nodes = nodes + cycle_lengths[i]
nodes = nodes - (len(cycle_lengths) - 1)

#TODO: Generate latencies based on cycle lengths & capacities
#lats = [1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
#lats = [1.0, 2.0, 3.0, 4.0, 3.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 2.0]
lats = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 1.0, 3.0, 2.0]
caps = []
for i in range(nodes):
    caps.append(1)

#TODO: Check every replication scheme
topology = StorageTopology(capacities=caps, num_nodes=nodes, num_objects=3, weights=lats)
#latency = numpy.inf
#alloc = []
#alloc_string = ""
#for allocation in topology.filter_feasible_replicative_allocations(topology.generate_all_replicative_allocations()):
#    topology.allocate_coded_objects(allocation)
#    avg_lat = topology.calculate_average_coded_latency()
#    if avg_lat < latency:
#        latency = avg_lat
#        alloc = allocation
#        alloc_string = topology.get_allocation()
#        print("Better average latency found: " + str(avg_lat))
#        print("Better allocation found: " + str(topology.get_allocation()))
#print("Optimal average latency of replication: " + str(latency))
#print("Lower bound average latency of replication: " + str(topology.calculate_global_lower_bound_average_latency()))
#print("Optimal allocation: " + str(alloc_string))

#TODO: Create best erasure coded allocation
allocation = [[1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]
topology.allocate_coded_objects(allocation)
avg_lat = topology.calculate_average_coded_latency()
print("Erasure coded average latency: " + str(avg_lat))
print("Erasure coded allocation: " + str(topology.get_allocation()))