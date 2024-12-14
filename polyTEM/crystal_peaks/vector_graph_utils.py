## utility functions to work with networkx Graph that represents
## 3D nematic connectivity

import sparse
import numpy as np
import networkx as nx
from .. import utilities 


def create_nodes_from_vectorfield(vector_field,node_container=None):
    """
    Args:
        vector_field: numpy array from CrystalStack3D vector field
        node_container: list of node key labels, optional.
    """
        
    ## Create empty graph with nodes
    n_nodes = vector_field.shape[0]
    if node_container is None:
        graph = nx.empty_graph(n=n_nodes)
        ## Convert vector field information to graph nodes
        r_dict = dict(enumerate(vector_field[:,0:2].tolist()))
        n_dict = dict(enumerate(vector_field[:,3:6].tolist()))
        alpha_dict = dict(enumerate(vector_field[:,2].tolist()))
        nx.set_node_attributes(graph,r_dict,name='Position')
        nx.set_node_attributes(graph,n_dict,name='Director')
        nx.set_node_attributes(graph,alpha_dict,name="Tilt")
    else:
        graph = nx.empty_graph(n=node_container)
        ## Convert vector field information to graph nodes
        r_dict = dict(zip(node_container.tolist(),vector_field[:,0:2].tolist()))
        n_dict = dict(zip(node_container.tolist(),vector_field[:,3:6].tolist()))
        alpha_dict = dict(zip(node_container.tolist(),vector_field[:,2].tolist()))
        nx.set_node_attributes(graph,r_dict,name='Position')
        nx.set_node_attributes(graph,n_dict,name='Director')
        nx.set_node_attributes(graph,alpha_dict,name="Tilt")
    
    
    return graph

def create_edges_from_simulation(graph,probs_mat,node_index):
    """
    creates edges with edge attribute probability
    Args:
        graph: networkx Graph to modify
        prob_mat: sparse object, Probability Adjacency Matrix with shape (N, graph.order())
        node_index: the Node index value that corresponds to the first row in prob_mat
    """
    ## generate random numbers to simulate against the probabiltiy matrix
    sim = np.random.random_sample(probs_mat.nnz)
    
    ## form edges where the simulated random numbers are less than the probability level
    connected_indices = (sim < probs_mat.data) ## Boolean Array if there's an edge

    ## edges = container of edges
    # where each edge given must be a 3-tuple (u,v,w), where w is a number
    # weight: str is the attribute name for the edge weights to be added, so I should add these as the probability
    edges_mat = np.vstack([probs_mat.coords[:,connected_indices],
                           probs_mat.data[connected_indices]]
                         ).T
    edges_mat[:,0] += node_index
    # create List of Tuples
    edges_lot = [(int(x[0]),int(x[1]),x[2]) for x in edges_mat.tolist()] 
    
    ## add edges to graph
    graph.add_weighted_edges_from(edges_lot,weight='probability')
    return

def sort_largest_subgraph(graph):
    """
    Return list of subgraphs based on graph size
    """
    S = [graph.subgraph(c).copy() for c in sorted(nx.connected_components(graph),key=len,reverse=True)]
    return S

def save(graph,savefile):
    utilities.saveas_json(nx.node_link_data(graph),savefile)
    
def load(savefile):
    vector_graph_data = utilities.load_json(savefile)
    vector_graph = nx.node_link_graph(vector_graph_data)
    return vector_graph

def nonintersecting_paths(paths):
    """
    Find all nonintersecting paths given paths dictionary,
    where paths is the output of nx.shortest_path(graph,source)
    """
    ## Find unique, non-intersecting paths
    traversed_nodes = []
    unique_paths = []

    # Iterate through paths by longest_path, equivalent to moving through paths backwards, since paths is sorted
    for target,path in reversed(paths.items()):
        if target in traversed_nodes:
            continue
        else:
            ## get path
            # compare used nodes with path
            seen_nodes_bool = list(map(lambda x: x in traversed_nodes, path))
            seen_node_indices = list(itertools.compress(range(len(seen_nodes_bool)), seen_nodes_bool))
            if seen_node_indices:
                last_seen_node_index = seen_node_indices[-1]
            else:
                last_seen_node_index = 0
            new_path = path[last_seen_node_index:-1]
            unique_paths.append(new_path)

            # Add all nodes in path to traversed_nodes
            traversed_nodes.extend(new_path)
    return unique_paths