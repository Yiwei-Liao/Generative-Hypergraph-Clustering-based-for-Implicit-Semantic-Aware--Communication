import os
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional

class Hypergraph:
    def __init__(self, N: range, E: Dict[int, Dict[Tuple[int, ...], int]], D: List[int]):
        self.N = N
        self.E = E
        self.D = D

    def __repr__(self) -> str:
        num_nodes = len(self.N)
        num_edges = sum(len(edges) for edges in self.E.values())
        return f"Hypergraph(nodes={num_nodes}, edges={num_edges})"

def _get_data_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data")

def read_hypergraph_label_names(dataname: str) -> List[str]:
    names = []
    pathname = _get_data_path()
    file_path = os.path.join(pathname, dataname, f"label-names-{dataname}.txt")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                names.append(line.strip())
    except FileNotFoundError:
        print(f"Warning: Label names file not found at {file_path}")
    return names

def read_hypergraph_labels(dataname: str) -> List[int]:
    labels = []
    pathname = _get_data_path()
    file_path = os.path.join(pathname, dataname, f"node-labels-{dataname}.txt")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                labels.append(int(line.strip()))
    except FileNotFoundError:
        print(f"Warning: Node labels file not found at {file_path}")
    return labels

def read_hypergraph_edges(dataname: str, maxsize: int = 25, minsize: int = 2) -> Dict[int, Dict[Tuple[int, ...], int]]:
    E: Dict[int, Dict[Tuple[int, ...], int]] = {}
    pathname = _get_data_path()
    file_path = os.path.join(pathname, dataname, f"hyperedges-{dataname}.txt")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                edge = [int(v) for v in line.strip().split(',')]
                edge.sort()
                if minsize <= len(edge) <= maxsize:
                    sz = len(edge)
                    if sz not in E:
                        E[sz] = {}
                    E[sz][tuple(edge)] = 1
    except FileNotFoundError:
        print(f"Warning: Hyperedges file not found at {file_path}")
    return E

def read_node_features(dataname: str) -> Optional[np.ndarray]:
    pathname = _get_data_path()
    file_path = os.path.join(pathname, dataname, f"node-features-{dataname}.txt")
    try:
        features = np.loadtxt(file_path) 
        return features
    except FileNotFoundError:
        print(f"Warning: Node features file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading node features from {file_path}: {e}")
        return None

def _shuffle_hyperedges(E: Dict[int, Dict[Tuple[int, ...], int]], shuffle_ratio: float) -> Dict[int, Dict[Tuple[int, ...], int]]:
    if shuffle_ratio <= 0:
        return E

    all_edges = [edge for size in E for edge in E[size]]
    edge_set = set(all_edges)
    num_edges = len(all_edges)
    num_swaps = int(num_edges * shuffle_ratio)
    
    print(f"--- Attempting to shuffle edges. Target swaps: {num_swaps} ---")
    
    successful_swaps = 0
    max_attempts = num_swaps * 10 

    for _ in range(max_attempts):
        if successful_swaps >= num_swaps:
            break
        try:
            idx1, idx2 = random.sample(range(num_edges), 2)
        except ValueError:
            break
            
        edge1_orig = all_edges[idx1]
        edge2_orig = all_edges[idx2]

        node1 = random.choice(edge1_orig)
        node2 = random.choice(edge2_orig)

        if node1 == node2 or node1 in edge2_orig or node2 in edge1_orig:
            continue

        new_edge1_set = (set(edge1_orig) - {node1}) | {node2}
        new_edge2_set = (set(edge2_orig) - {node2}) | {node1}

        new_edge1 = tuple(sorted(list(new_edge1_set)))
        new_edge2 = tuple(sorted(list(new_edge2_set)))

        if len(new_edge1) != len(edge1_orig) or \
           len(new_edge2) != len(edge2_orig) or \
           new_edge1 in edge_set or \
           new_edge2 in edge_set:
            continue

        all_edges[idx1] = new_edge1
        all_edges[idx2] = new_edge2
        edge_set.remove(edge1_orig)
        edge_set.remove(edge2_orig)
        edge_set.add(new_edge1)
        edge_set.add(new_edge2)
        
        successful_swaps += 1

    print(f"--- Edge shuffling complete. Performed {successful_swaps} successful swaps. ---")

    new_E: Dict[int, Dict[Tuple[int, ...], int]] = {}
    for edge in all_edges:
        sz = len(edge)
        if sz not in new_E:
            new_E[sz] = {}
        new_E[sz][edge] = 1
        
    return new_E

def read_hypergraph_data(dataname: str, maxsize: int = 25, minsize: int = 2, 
                         shuffle_ratio: float = 0.0,
                         return_labels: bool = True, return_features: bool = False) -> Any:
    E = read_hypergraph_edges(dataname, maxsize, minsize)

    if not E:
        print(f"Warning: No edges found for dataset '{dataname}' with given size constraints.")
        empty_hypergraph = Hypergraph(range(1, 1), {}, [])
        if return_labels and return_features: return empty_hypergraph, [], None
        elif return_labels: return empty_hypergraph, []
        elif return_features: return empty_hypergraph, None
        else: return empty_hypergraph

    if shuffle_ratio > 0:
        E = _shuffle_hyperedges(E, shuffle_ratio)

    if not E:
        n = 0
    else:
        n = max(max(e) for edges in E.values() for e in edges if e) if any(E.values()) else 0
    
    D = [0] * (n + 1)
    for edges in E.values():
        for edge in edges:
            for node in edge:
                D[node] += 1

    for k in range(minsize, maxsize + 1):
        if k not in E:
            E[k] = {}

    N = range(1, n + 1)
    hypergraph_obj = Hypergraph(N, E, D)

    labels = read_hypergraph_labels(dataname) if return_labels else None
    features = read_node_features(dataname) if return_features else None

    if labels is not None:
        labels = labels[:n]

    if return_labels and return_features: return hypergraph_obj, labels, features
    elif return_labels: return hypergraph_obj, labels
    elif return_features: return hypergraph_obj, features
    else: return hypergraph_obj

def hypermodularity_datasets():
    print("HyperModularity Package Datasets \n")
    pathname = _get_data_path()
    if not os.path.isdir(pathname):
        print(f"Data directory not found at: {pathname}")
        return
    for item in sorted(os.listdir(pathname)):
       full_path = os.path.join(pathname, item)
       if os.path.isdir(full_path):
           print(f"\t {item}")

if __name__ == '__main__':
    print("--- Listing available datasets ---")
    hypermodularity_datasets()
    print("\n--- Loading a sample dataset without shuffling ---")
    dataname = "amap"
    H, _, _ = read_hypergraph_data(dataname, return_labels=True, return_features=True)
    print("Loaded Hypergraph:", H)
    print(f"\n--- Statistics for dataset '{dataname}' ---")
    # H.E is a dictionary where keys are the cardinalities (i.e., sizes) of hyperedges,
    # and values are the sets of hyperedges for each cardinality.
    # We can iterate through it to count the number of hyperedges for each cardinality.
    
    total_edges = 0
    # Sort the cardinalities (keys) for a consistent and ordered output
    cardinalities = sorted(H.E.keys())
    
    print("Hyperedge count by cardinality:")
    for cardinality in cardinalities:
        # Get the number of hyperedges for the current cardinality
        num_edges = len(H.E[cardinality])
        # Only print cardinalities with actual hyperedges to keep the output clean
        if num_edges > 0:
            print(f"  - Cardinality {cardinality}: {num_edges} hyperedges")
            total_edges += num_edges
            
    print(f"Total number of hyperedges: {total_edges}")
