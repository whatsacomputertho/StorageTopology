from StorageTopology.StorageTopology import StorageTopology
import sympy as sp
import matplotlib.pyplot as plt
import csv, math, random, numpy, networkx
from itertools import combinations

#The goal is to generate a number sequence S_n which represents the distance from
#the lower bound for n nodes and n-k objects such that 0 < k <= n.  We would like
#to do so for as many values n as possible.

#Prep file
f = open("output.csv", "a")
f.write("nodes,objects,chromatic_number_of_g_prime,lower_bound_avg,optimal_avg,difference_in_numerator\n")
f.close()

#Generate storage topology parameters.
for j in range(3, 20):
    #i is the number of nodes in the topology
    print("Analyzing m = " + str(j))
    for i in range(j, 2*j):
        #j is the number of objects in the topology
        print("Analyzing n = " + str(i))
        
        #Generate the node capacities
        c = []
        for k in range(i):
            c.append(1)
        print("Capacities = " + str(c))

        #Generate the edge weights
        w = []
        for k in range(i - 1):
            w.append(float(-abs(k-((i/2) - 1)) + (i/2)))
        pattern = w.copy()
        l = i-2
        while l > 0:
            for m in range(l):
                w.append(pattern[m])
            l = l - 1
        print("Weights = " + str(w))
            

        #Instantiate the StorageTopology object
        topology = StorageTopology(capacities=c, num_nodes=i, num_objects=j, weights=w)

        #Calculate lower-bound global average latency
        lower_bound = topology.calculate_global_lower_bound_average_latency()
        print("Lower bound average latency = " + str(lower_bound))
        graph = topology.generate_modified_retrieval_graph()
        chromatic_number = topology.calculate_chromatic_number(graph)
        print("Chromatic number of modified retrieval graph = " + str(chromatic_number))
        
        #Debug
        #networkx.draw(graph)
        #plt.show()

        #Setup
        index = 0
        index_found = 0
        average = numpy.inf

        #Generate optimal average latency for the data store
        base_allocation = []
        additional_allocations = []
        for k in range(j):
            node_allocation = []
            additional_node_allocation = []
            for l in range(i):
                node_allocation.append(0)
                additional_node_allocation.append(0)
            node_allocation[k] = 1
            additional_node_allocation[k] = 1
            base_allocation.append(node_allocation)
            additional_allocations.append(additional_node_allocation)
        k_combinations = combinations(additional_allocations, i-j)
        allocations = []
        for k_combination in k_combinations:
            allocation = base_allocation.copy()
            for element in k_combination:
                allocation.append(element)
            allocations.append(allocation)
        
        #Check all replicative allocations for optimal allocation
        #allocations = topology.generate_feasile_replicative_allocations()
        for allocation in allocations:
            topology.allocate_coded_objects(allocation)
            #alloc_id = index
            #alloc = topology.get_allocation()
            #print(allocation)
            #wc_lat = topology.calculate_global_worst_case_latency()
            avg_lat = topology.calculate_global_average_latency()
            #line = "{alloc_id},{allocation},{wc_lat},{avg_lat}".format(alloc_id = alloc_id, allocation = topology.get_allocation(), wc_lat = wc_lat, avg_lat = avg_lat)
            if avg_lat <= average:
                index_found = index
                print("Another good average found at index " + str(index_found) + ": " + str(avg_lat))
                average = avg_lat
            if (index - index_found) <= pow(j, (i-1)):
                index = index + 1
            else:
                break

        #Calculate distance from lower bound
        avg_num = average * len(topology.nodes) * len(topology.objects)
        lb_num = lower_bound * len(topology.nodes) * len(topology.objects)
        diff = avg_num - lb_num
        print("Numerator of the average is: " + str(avg_num))
        print("Numerator of the lower bound is: " + str(lb_num))

        #Write to csv
        #f = open("output.csv", "a")
        #f.write(str(i) + "," + str(j) + "," + str(chromatic_number) + "," + str(lower_bound) + "," + str(average) + "," + str(diff) + "\n")
        #f.close()

print("done")