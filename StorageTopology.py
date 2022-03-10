import numpy as np
import sympy as sp
import networkx, grinpy
import math
from itertools import combinations
from itertools import permutations

#Specifies the propoerties of a node in a distributed data storage network topology
class Node:
    #Constructor for node
    def __init__(self, capacity):
        self.coefficients = []
        self.objects = []
        self.capacity = capacity

    #Getter for capacity
    def get_capacity(self):
        return self.capacity

    #Setter for capacity
    def set_capacity(self, capacity):
        self.capacity = capacity

    #Getter for list of objects
    def get_objects(self):
        return self.objects

    #Setter for list of objects
    def set_objects(self, objects):
        self.objects = objects

    #Getter for list of object coefficients
    def get_coefficients(self):
        return self.coefficients

    #Setter for list of object coefficients
    def set_coefficients(self, coefficients):
        self.coefficients = coefficients

#Specifies the properties of an edge in a distributed data storage network topology
class Edge:
    #Constructor for edge
    def __init__(self, weight, nodes):
        self.weight = weight
        self.nodes = nodes

    #Getter for edge weight
    def get_weight(self):
        return self.weight

    #Setter for edge weight
    def set_weight(self, weight):
        self.weight = weight

    #Getter for the nodes incident with the edge (i.e. edge endpoints)
    def get_nodes(self):
        return self.nodes

    #Setter for the nodes incident with the edge (i.e. edge endpoints)
    def set_nodes(self, nodes):
        self.nodes = nodes

#Specifies the properties of a data object (yes, it's poorly named but
#wanted to remain consistent with the paper's terminology)
class Object:
    #Constructor for object
    def __init__(self, name):
        self.name = name

    #Getter for object name (i.e. one might name it "X1", "X2", etc.)
    def get_name(self):
        return self.name

    #Setter for object name
    def set_name(self, name):
        self.name = name

    #Getter for object size in bits.  Property currently unused.
    def get_size(self):
        return self.size

    #Setter for object size in bits.  Property currently unused.
    def set_size(self, size):
        self.size = size

#StorageTopology class represents a distributed data store as defined in the paper
class StorageTopology:
    #Constructor for StorageTopology class
    def __init__(self, capacities, num_nodes, num_objects, weights):
        #Initialize list of nodes
        self.nodes = []

        #Initialize list of objects
        self.objects = []

        #Initialize list of edges
        self.edges = []

        #Initialize list of matrices
        self.matrices = []

        #Initialize list of retrieval sets
        self.retrieval_sets = []

        #For the number of nodes passed in, create unit nodes for its capacity
        #(i.e. if we pass in a capacity of 2, it creates 2 nodes)
        for i in range(num_nodes):
            for j in range(capacities[i]):
                self.nodes.append(Node(1))

        #For the number of objects passed in, create a new object with name X_i counting from i=0 upward
        for i in range(num_objects):
            o_name = "X{index:d}".format(index = i)
            self.objects.append(Object(o_name))
        k = 0

        #Generate a complete graph and specify the edge weights for each edge by the list of weights passed in
        for i in range(num_nodes * capacities[i]):
            for j in range(i + 1, (num_nodes * capacities[i])):
                e = [self.nodes[i], self.nodes[j]]
                self.edges.append(Edge(weights[k], e))
                k += 1

    #Static method allowing the programmer to read wieghts from a simple text file with weights separated by line
    def read_weights(filepath):
        weights=[]
        file = open(filepath)
        for line in file.readlines():
            weights.append(float(line))
        return weights
        
    read_weights = staticmethod(read_weights)

    #Static method allowing the programmer to read capacities from a simple text file with capacities separated by line
    def read_capacities(filepath):
        capacities=[]
        file = open(filepath)
        for line in file.readlines():
            capacities.append(int(np.floor(float(line))))
        return capacities
        
    read_capacities = staticmethod(read_capacities)

    #Static method allowing the programmer to permute an allocation in every possible way to find optimal mapping given
    #an allocation pattern
    def permute(allocation):
        return permutations(allocation)

    permute = staticmethod(permute)

    #Private method that generates all replicative allocation patterns (i.e. length-n, base-m numbers/vectors)
    def __generate_all_replicative_allocation_patterns(self):
        patterns = []
        for i in range(len(self.objects)**len(self.nodes)):
            dec_pattern = i
            pattern = []
            j = len(self.nodes) - 1
            while j >= 0:
                base_n_place_value = 0
                power = len(self.objects)**j
                while dec_pattern >= power:
                    dec_pattern -= power
                    base_n_place_value += 1
                pattern.append(base_n_place_value)
                j -= 1
            patterns.append(pattern)
        return patterns
    
    #Using the previous function, map every replicative allocation pattern to an allocation matrix
    def __generate_all_replicative_allocations(self):
        allocations = []
        for pattern in self.__generate_all_replicative_allocation_patterns():
            allocation = []
            for i in range(len(pattern)):
                row = []
                for j in range(len(pattern)):
                    row.append(0)
                row[pattern[i]] = 1
                allocation.append(row)
            allocations.append(allocation)
        return allocations

    #Check the rank of each allocation matrix in a list of them which is passed in as an argument
    def __filter_feasible_replicative_allocations(self, all_allocations):
        feasible_allocations = []
        for allocation in all_allocations:
            if np.linalg.matrix_rank(allocation) == len(self.objects):
                feasible_allocations.append(allocation)
        return feasible_allocations

    #Public method allowing the programmer to generate all feasible replicative allocations on a distributed
    #data store.
    def generate_feasile_replicative_allocations(self):
        return self.__filter_feasible_replicative_allocations(self.__generate_all_replicative_allocations())

    #Public method allowing the programmer to generate a formatted string of the current allocation
    def get_allocation(self):
        allocation = ""
        for node in self.nodes:
            item = ""
            for i in range(len(node.coefficients)):
                if node.coefficients[i] == 0:
                    item += ""
                elif node.coefficients[i] == 1:
                    item += self.objects[i].name
                    item += " + "
                else:
                    item += "({coef}){name}".format(coef = node.coefficients[i], name = self.objects[i].name)
                    item += " + "
            if item[-3:] == " + ":
                item = item[:-3]
            if self.nodes.index(node) < (len(self.nodes) - 1):
                allocation += "{st}, ".format(st = item)
            else:
                allocation += item
        return allocation

    #Private method which checks the feasibility of the current allocation (may be coded or replicative)
    def __check_coded_storage_feasibility(self):
        coefficient_matrix = []
        for node in self.nodes:
            coefficient_matrix.append(node.coefficients)
        if np.linalg.matrix_rank(coefficient_matrix) == len(self.objects):
            return True
        else:
            return False

    #Private method which generates allocation submatrices formed using k nodes.  Function is used in the process 
    #of generating the retrieval sets of the current allocation.
    def __generate_coded_storage_coefficient_matrices(self, k):
        k_combinations = combinations(self.nodes, k)
        matrices = []
        for combination in k_combinations:
            matrix = []
            for node in combination:
                matrix.append(node.coefficients)
            matrices.append(matrix)
        return matrices

    #Private method which generates all allocation submatrices (for any k).  Function is used in the process
    #of generating the retrieval sets of the current allocation.
    def __generate_all_coded_storage_coefficient_matrices(self):
        self.matrices.clear()
        matrices = []
        for k in range(1, len(self.nodes) + 1):
            for matrix in self.__generate_coded_storage_coefficient_matrices(k):
                matrices.append(matrix)
        self.matrices = matrices

    #Private method which checks whether a given matrix is solvable for a particular index.  Function is used
    #in the process of generating the retrieval sets of the current allocation.
    def __check_matrix_solvable(self, matrix, index):
        for i in range(matrix.shape[0]):
            indices = []
            for j in range(len(matrix.row(i))):
                if matrix.row(i)[j] != 0:
                    indices.append(j)
            if len(indices) == 1 and indices[0] == index:
                return  True
        return False

    #Private function which translates a list of allocation submatrices into the node objects to which
    #the matrices' rows belong.  Function is used in the process of generating the retrieval sets of the
    #current allocation.
    def __get_node_set_from_coefficient_matrix(self, matrices):
        sets = []
        for i in range(len(matrices)):
            nodes = []
            for j in range(len(matrices[i])):
                for node in self.nodes:
                    if node.coefficients is matrices[i][j]:
                        nodes.append(node)
            sets.append(nodes)
        return sets

    #Private function which generates the retrieval sets of an object in the current allocation.
    def __generate_coded_storage_retrieval_set_by_object(self, obj):
        solvable_matrices = []
        obj_index = self.objects.index(obj)
        for i in range(len(self.matrices)):
            if self.__check_matrix_solvable(sp.Matrix(self.matrices[i]).rref()[0], obj_index):
                not_superlist = True
                for j in range(len(solvable_matrices)):
                    if(all(row in self.matrices[i] for row in solvable_matrices[j]) and (len(self.matrices[i]) != len(solvable_matrices[j]))):
                        not_superlist = False
                if not_superlist:
                    solvable_matrices.append(self.matrices[i])
        return self.__get_node_set_from_coefficient_matrix(solvable_matrices)

    #Private function which generates the retrieval sets of every object in the current allocation.
    #The retrieval sets are stored in memory as a property of the StorageTopology object on which it operates.
    def __generate_coded_retrieval_sets(self):
        sets = []
        for obj in self.objects:
            set = self.__generate_coded_storage_retrieval_set_by_object(obj)
            sets.append(set)
        self.retrieval_sets = sets

    #Public function which allos the user to allocate objects according to a provided allocation.  It then
    #calls the previous private methods to configure the StorageTopology object accordingly.
    def allocate_coded_objects(self, allocation):
        for i in range(len(allocation)):
            self.nodes[i].coefficients = allocation[i]
        self.__generate_all_coded_storage_coefficient_matrices()
        self.__generate_coded_retrieval_sets()

    #Private function that returns minimum number of nodes needed to retrieve all objects
    #or throws an exception if capacities are not uniform
    def __calculate_min_nodes_for_retrieval(self):
        capacity = self.nodes[0].capacity
        for i in range(len(self.nodes)):
            if self.nodes[i].capacity != capacity:
                raise Exception('Node capacities are not uniform')
        return math.ceil(len(self.objects) / capacity) - 1

    #Private function that returns the minimum-weight edges incident with a given node
    def __calculate_minimum_weight_edges(self, node):
        min_retrieval = self.__calculate_min_nodes_for_retrieval()
        edges = []
        if min_retrieval == 0:
            return edges
        else:
            for edge in self.edges:
                if node in edge.nodes:
                    if len(edges) < min_retrieval:
                        edges.append(edge)
                    else:
                        edges.sort(key=lambda e: e.weight, reverse=True)
                        if edge.weight < edges[0].weight:
                            edges.remove(edges[0])
                            edges.append(edge)
        return edges

    #Public function which calculates the local lower-bound average latency of a given node
    def calculate_local_lower_bound_average_latency(self, node):
        minimum_weight_edges = self.__calculate_minimum_weight_edges(node)
        weight_sum = 0
        for edge in minimum_weight_edges:
            weight_sum = weight_sum + edge.weight
        return weight_sum / (len(minimum_weight_edges) + 1)

    #Public function which calculates the global lower-bound average latency of the data store
    def calculate_global_lower_bound_average_latency(self):
        weight_sum = 0
        for node in self.nodes:
            weight_sum = weight_sum + self.calculate_local_lower_bound_average_latency(node)
        return weight_sum / len(self.nodes)

    #Public function that generates G' as a networkx graph
    def generate_modified_retrieval_graph(self):
        g = networkx.Graph()
        g.add_nodes_from(self.nodes)
        for node in self.nodes:
            nodes = [node]
            min_edges = self.__calculate_minimum_weight_edges(node)
            for edge in min_edges:
                for n in edge.nodes:
                    if n != node:
                        nodes.append(n)
            for i in range(len(nodes) - 1):
                for j in range(i+1, len(nodes)):
                    g.add_edge(nodes[i], nodes[j])
        return g

    #Public function that returns the chromatic number of a provided graph (presumably G')
    #See Algorithm 4.3.1 in Storage Latency Tradeoffs in Geographically Distributed Storage Settings for info.
    def calculate_chromatic_number(self, modified_retrieval_graph):
        return grinpy.chromatic_number(modified_retrieval_graph)

    #Private function that calculates the retrieval latency of a given node which hopes to retrieve a given object.
    def __calculate_coded_retrieval_latencies_by_node_and_object(self, node, obj):
        retrieval_latencies = []
        retrieval_sets = self.retrieval_sets[self.objects.index(obj)]
        for retrieval_set in retrieval_sets:
            set_latencies = []
            for n in retrieval_set:
                is_node = False
                if n is node:
                    set_latencies.append(0.0)
                    is_node = True
                for edge in self.edges:
                    if is_node:
                        break
                    if (((edge.nodes[0] is node) and (edge.nodes[1] is n)) or ((edge.nodes[0] is n) and (edge.nodes[1] is node))):
                        set_latencies.append(edge.weight)
            retrieval_latencies.append(set_latencies)
        return retrieval_latencies

    #Private function that calculates the maximum retrieval latency if a node must communicate with a
    #subset of nodes in order to retrieve an object.
    def __calculate_maximum_latencies_from_retrieval_latencies(self, latencies):
        maximum_latency = latencies[0]
        for latency in latencies:
            if latency > maximum_latency:
                maximum_latency = latency
        return maximum_latency

    #Private function that calculates the maximum retrieval latency of every subset of nodes with which a node
    #must communicate to retrieve a given object.
    def __calculate_maximum_latencies_from_retrieval_lists(self, retrieval_latencies):
        maximum_latencies = []
        for latencies in retrieval_latencies:
            maximum_latencies.append(self.__calculate_maximum_latencies_from_retrieval_latencies(latencies))
        return maximum_latencies

    #Private function that calculates the minimum retrieval latency given a list of retrieval latencies
    def __calculate_minimum_latency_from_retrieval_list(self, latencies):
        min_latency = latencies[0]
        for latency in latencies:
            if latency < min_latency:
                min_latency = latency
        return min_latency

    #Private function that calculates the minimum retrieval latencies for each object given a node.
    def __calculate_minimum_retrieval_latencies_by_node(self, node):
        min_latencies = []
        for obj in self.objects:
            min_latencies.append(self.__calculate_minimum_latency_from_retrieval_list(self.__calculate_maximum_latencies_from_retrieval_lists(self.__calculate_coded_retrieval_latencies_by_node_and_object(node, obj))))
        return min_latencies

    #Public function that calculates the local average latency of a given node.
    def calculate_average_latency_by_node(self, node):
        latency_sum = 0.0
        for obj in self.objects:
            latency_sum += self.__calculate_minimum_latency_from_retrieval_list(self.__calculate_maximum_latencies_from_retrieval_lists(self.__calculate_coded_retrieval_latencies_by_node_and_object(node, obj)))
        return (latency_sum / len(self.objects))

    #Public function that calculates the local worst case latency of a given node.
    def calculate_worst_case_coded_latency_by_node(self, node):
        min_latencies = self.__calculate_minimum_retrieval_latencies_by_node(node)
        worst_case_latency = min_latencies[0]
        for f in min_latencies:
            if f > worst_case_latency:
                worst_case_latency = f
        return worst_case_latency

    #Private function which generates a list of local worst case latencies (i.e. for each node)
    def __calculate_worst_case_coded_latencies(self):
        min_latencies = []
        for node in self.nodes:
            min_latencies.append(self.calculate_worst_case_coded_latency_by_node(node))
        return min_latencies

    #Private function which generates a list of average latencies (i.e. for each node)
    def __calculate_average_coded_latencies(self):
        average_latencies = []
        for node in self.nodes:
            average_latencies.append(self.calculate_average_latency_by_node(node))
        return average_latencies

    #Public function which generates the global worst case latency of the distributed data store
    def calculate_worst_case_coded_latency(self):
        worst_case_latencies = self.__calculate_worst_case_coded_latencies()
        worst_case_latency = worst_case_latencies[0]
        for f in worst_case_latencies:
            if f > worst_case_latency:
                worst_case_latency = f
        return worst_case_latency

    #Public function which generates the global average latency of the distributed data store
    def calculate_average_coded_latency(self):
        latency_sum = 0.0
        average_coded_latencies = self.__calculate_average_coded_latencies()
        for f in average_coded_latencies:
            latency_sum += f
        return (latency_sum / len(average_coded_latencies))
