import numpy as np
from StorageTopology.StorageTopology import StorageTopology

#Specify edge latencies
latencies = [5.1, 5.0, 5.4, 3.6, 3.6, 8.1, 8.1, 5.1, 6.3, 3.2]

#Instantiate the network
topology = StorageTopology(num_nodes=5, num_objects=3, capacities=[1, 1, 1, 1, 1], weights=latencies)

#Generate all replicative allocations
allocations = topology.generate_feasile_replicative_allocations()

#Instantiate average and worst-case latency variable & allocation variable
wc_lat = np.inf
avg_lat = np.inf
alloc = []

#Minimize by average latency
for allocation in allocations:
    topology.allocate_coded_objects(allocation)
    alat = topology.calculate_global_average_latency()
    wlat = topology.calculate_global_worst_case_latency()
    if alat < avg_lat:
        avg_lat = alat
        alloc = allocation
    if wlat < wc_lat:
        wc_lat = wlat

#Check if allocation is minimal with regard to worst-case latency
topology.allocate_coded_objects(alloc)
if topology.calculate_global_average_latency() == avg_lat:
    print("Allocation is optimal with regard to average latency")
    print("Average Latency: " + str(avg_lat))
if topology.calculate_global_worst_case_latency() == wc_lat:
    print("Allocation is optimal with regard to worst-case latency")
    print("Wrost-Case Latency: " + str(wc_lat))