from StorageTopology.StorageTopology import StorageTopology

#Specify edge latencies
latencies = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]

#Instantiate the network
topology = StorageTopology(num_nodes=4, num_objects=3, capacities=[1, 1, 1, 1, 1], weights=latencies)

#Specify the allocation
alloc = [
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1]
]

#Allocate the objects
topology.allocate_coded_objects(alloc)

#Check the worst-case and average latency
wc_lat = topology.calculate_global_worst_case_latency()
avg_lat = topology.calculate_global_average_latency()

#Print to command line
print("Worst case latency: " + str(wc_lat))
print("Average latency: " + str(avg_lat))