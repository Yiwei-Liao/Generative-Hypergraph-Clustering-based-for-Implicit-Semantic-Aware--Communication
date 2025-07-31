import random
import math
import copy
import numpy as np
from collections import defaultdict
from scipy import sparse
from itertools import combinations
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure
from collections import Counter, defaultdict

# ==============================================================================
# Helper Classes and Functions (to make the code runnable)
# ==============================================================================

class Hypergraph:
    """
    A helper class to represent a hypergraph, simulating the hypergraph type in Julia code.
    """
    def __init__(self, N, E, D):
        """
        Initializes a hypergraph object.

        Args:
            N (list or range): A list of nodes (e.g., range(n) for nodes 0 to n-1).
            E (dict): A dictionary of edges, formatted as {size: {edge_tuple: count}}.
            D (list or np.array): A list of degrees for each node.
        """
        self.N = list(N)
        self.E = E
        self.D = np.array(D, dtype=int)

    def __repr__(self):
        return f"Hypergraph(n_nodes={len(self.N)}, n_edge_sizes={len(self.E)})"

    def compute_degrees(self):
        """
        Simulates the computeDegrees! function from Julia.
        Recalculates the degrees of all nodes based on the current edge set E.
        """
        n = len(self.N)
        self.D = np.zeros(n, dtype=int)
        for size, edges in self.E.items():
            for edge, count in edges.items():
                for node in edge:
                    # Assuming nodes are 0-indexed
                    if 0 <= node < n:
                        self.D[node] += count

def clique_expansion(H, arg1, arg2):
    """
    Simulates the CliqueExpansion function from Julia.
    This should return a sparse matrix representing the projected graph.
    """
    n = len(H.D)
    rows, cols, data = [], [], []
    for size, edges in H.E.items():
        if size < 2:
            continue
        for edge, weight in edges.items():
            # Add edges between all pairs of nodes in the hyperedge
            for u, v in combinations(edge, 2):
                rows.append(u)
                cols.append(v)
                data.append(weight)
                # Add symmetric edges since it's an undirected graph
                rows.append(v)
                cols.append(u)
                data.append(weight)
    
    # The COO format will automatically sum the weights of duplicate entries
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

# ==============================================================================
# Translated Python Functions
# ==============================================================================

def down_sample_edges(H, prop):
    """
    Downsamples a hypergraph by retaining hyperedges with a certain probability.
    This function modifies the hypergraph H in-place.

    Args:
        H (Hypergraph): The hypergraph object.
        prop (float): The probability that each hyperedge is kept.
    """
    # Iterate over a copy of the keys, as the dictionary size may change during the loop
    for k in list(H.E.keys()):
        for e in list(H.E[k].keys()):
            original_count = H.E[k][e]
            removed_count = 0
            # Decide independently for each instance of a repeated edge
            for _ in range(original_count):
                if random.random() > prop:
                    removed_count += 1
            
            if removed_count > 0:
                H.E[k][e] -= removed_count

            # If all instances of an edge are removed, delete the entry
            if H.E[k][e] == 0:
                del H.E[k][e]
        
        # If the set of edges for a certain size becomes empty, that size can also be removed
        if not H.E[k]:
            del H.E[k]

    H.compute_degrees()

def subhypergraph(h, in_subhypergraph):
    """
    Creates a sub-hypergraph from a given set of nodes.
    Note: This version corresponds to the first `subhypergraph` function in the Julia code.

    Args:
        h (Hypergraph): The original hypergraph object.
        in_subhypergraph (list[bool]): A boolean list where in_subhypergraph[i] is True
            if node i belongs to the subgraph. Assumes nodes are 0-indexed.

    Returns:
        tuple: (sub_h, node_map)
            - sub_h: The new hypergraph object.
            - node_map: A dictionary mapping old node indices to new node indices.
    """
    # 1. Identify the old nodes to keep and create a mapping from old to new indices
    old_nodes_to_keep = [i for i, keep in enumerate(in_subhypergraph) if keep]
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(old_nodes_to_keep)}

    # 2. Filter edges and preserve their weights
    new_edges_with_multiplicity = []
    for size, edges in h.E.items():
        for edge, multiplicity in edges.items():
            new_edge = [v for v in edge if in_subhypergraph[v]]
            if len(new_edge) > 1:
                # Keep the multiplicity of the edge
                new_edges_with_multiplicity.extend([new_edge] * multiplicity)

    # 3. Renumber the edges using the mapping
    renumbered_new_edges = []
    for e in new_edges_with_multiplicity:
        renumbered_edge = tuple(sorted([node_map[v] for v in e]))
        renumbered_new_edges.append(renumbered_edge)

    # 4. Build the new edge dictionary sub_E and count multiplicities
    sub_E = defaultdict(lambda: defaultdict(int))
    for edge in renumbered_new_edges:
        sz = len(edge)
        sub_E[sz][edge] += 1

    # 5. Create the new hypergraph and compute its degrees
    n = len(node_map)
    sub_h = Hypergraph(range(n), dict(sub_E), [])
    sub_h.compute_degrees()
    
    return sub_h, node_map
    
def mutual_information(Z, Z_hat, normalized=False):
    """
    Calculates the mutual information (MI) between two clusterings, optionally normalized (NMI).

    Args:
        Z (list): The first clustering (a list of cluster labels for each point).
        Z_hat (list): The second clustering.
        normalized (bool): If True, returns the Normalized Mutual Information (NMI).

    Returns:
        float: The value of MI or NMI.
    """
    if len(Z) != len(Z_hat):
        raise ValueError("Input lists must have the same length.")
    
    n = len(Z)
    if n == 0:
        return 0.0

    # Calculate the frequencies for the joint and marginal distributions
    p_XY = defaultdict(int)
    p_X = defaultdict(int)
    p_Y = defaultdict(int)

    for i in range(n):
        p_XY[(Z[i], Z_hat[i])] += 1
        p_X[Z[i]] += 1
        p_Y[Z_hat[i]] += 1

    # Calculate mutual information I(X;Y) = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )
    # This implementation is more efficient than the nested loops in the original Julia code
    I = 0.0
    for (x, y), count_xy in p_XY.items():
        p_xy = count_xy / n
        p_x = p_X[x] / n
        p_y = p_Y[y] / n
        I += p_xy * math.log(p_xy / (p_x * p_y))

    if not normalized:
        return I

    # Calculate entropy H(X) and H(Y) for normalization
    # H(X) = - sum_x p(x) * log(p(x))
    H_X = -sum((c / n) * math.log(c / n) for c in p_X.values())
    H_Y = -sum((c / n) * math.log(c / n) for c in p_Y.values())

    # Calculate Normalized Mutual Information NMI = 2 * I / (H(X) + H(Y))
    denominator = H_X + H_Y
    if denominator == 0:
        # If both clusterings have only one cluster, entropy is 0. 
        # In this case, I is also 0, and NMI should be 1.
        return 1.0 if I == 0 else 0.0
    
    return (2 * I) / denominator

def sub_hypergraph(H, b, Z=None):
    """
    Creates a sub-hypergraph based on a boolean vector indicating node inclusion.
    Note: This version corresponds to the second `subHypergraph` function in the Julia code.

    Args:
        H (Hypergraph): The original hypergraph object.
        b (list[bool] or np.array[bool]): A boolean vector indicating which nodes to include.
        Z (list, optional): An optional clustering assignment for the nodes.

    Returns:
        Returns (sub_H, sub_Z) if Z is provided. Otherwise, returns sub_H.
    """
    # 1. Identify nodes to keep and create a renumbering map
    key = [i for i, val in enumerate(b) if val]
    key_set = set(key)
    nodemap = {old_idx: new_idx for new_idx, old_idx in enumerate(key)}

    # 2. Filter nodes and the optional clustering vector Z
    new_N = list(range(len(key)))
    new_Z = [Z[i] for i in key] if Z is not None else None

    # 3. Filter edges: only keep edges where all nodes are in the subgraph
    new_E = defaultdict(lambda: defaultdict(int))
    for k, edges in H.E.items():
        for edge, count in edges.items():
            if all(node in key_set for node in edge):
                new_edge = tuple(sorted([nodemap[i] for i in edge]))
                # The original Julia code `... + 1` seems to ignore the original multiplicity `count`.
                # Assuming the intent is to preserve weights, we use `+ count`.
                new_E[k][new_edge] += count

    # 4. Create the new hypergraph and compute its degrees
    H_new = Hypergraph(new_N, dict(new_E), [])
    H_new.compute_degrees()

    return (H_new, new_Z) if Z is not None else H_new

def projected_graph(H):
    """
    Creates a projected graph from a hypergraph.
    In the projected graph, an edge exists between two nodes if they share a hyperedge.
    The edge weight is the number of hyperedges they share (or the sum of their weights).

    Args:
        H (Hypergraph): The hypergraph object.

    Returns:
        Hypergraph: A new hypergraph object representing the projected graph (all edges have size 2).
    """
    n = len(H.D)
    A = clique_expansion(H, False, False)
    rows, cols, weights = sparse.find(A)

    # Create the edge dictionary for the new graph
    E = {}
    for i in range(len(rows)):
        # Only take the upper triangle to avoid duplicates, since the graph is undirected
        if rows[i] < cols[i]:
            edge = (rows[i], cols[i])
            E[edge] = weights[i]

    # The new "hypergraph" only has edges of size 2
    E_new = {2: E}

    # Create the new graph object
    H_bar = Hypergraph(range(n), E_new, [])
    H_bar.compute_degrees()

    return H_bar

def k_core(H, Z, core):
    """
    Computes the k-core of a hypergraph.
    Repeatedly removes nodes with a degree less than `core`.

    Args:
        H (Hypergraph): The hypergraph object.
        Z (list): The clustering assignment of the nodes.
        core (int): The minimum degree in the k-core.

    Returns:
        tuple: (H_core, Z_core) containing the core hypergraph and the corresponding clustering assignments.
    """
    H_core = copy.deepcopy(H)
    Z_core = list(Z)

    # The original code iterates 10 times, assuming this is sufficient for convergence.
    # A more robust implementation would loop until no more nodes are removed.
    for _ in range(10):
        if len(H_core.N) == 0:
            break
            
        nodes_to_keep = H_core.D >= core
        
        if np.all(nodes_to_keep):
            break

        H_core, Z_core = sub_hypergraph(H_core, nodes_to_keep, Z_core)

    return H_core, Z_core

def remove_edges(H, remove=None):
    """
    Removes entire size classes of edges from a hypergraph.
    Modifies H in-place.

    Args:
        H (Hypergraph): The hypergraph object.
        remove (list[int], optional): A list of edge sizes to remove.
    """
    if remove is None:
        remove = []

    for k in remove:
        H.E.pop(k, None)
    
    H.compute_degrees()
    
def evaluate_clustering(Z_true: list, Z_pred: list):
    """
    Calculates and prints a series of standard clustering evaluation metrics.

    Args:
        Z_true (list): The ground truth cluster labels.
        Z_pred (list): The predicted cluster labels.
    """
    if len(Z_true) != len(Z_pred):
        raise ValueError("Input lists must have the same length.")
    
    # Convert input lists to numpy arrays for use with scikit-learn
    Z_true = np.array(Z_true)
    Z_pred = np.array(Z_pred)

    # 1. Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(Z_true, Z_pred)
    
    # 2. Normalized Mutual Information (NMI) - should be very close to the result of your own implementation
    nmi = normalized_mutual_info_score(Z_true, Z_pred)
    
    # 3. Homogeneity, Completeness, and V-Measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(Z_true, Z_pred)
    
    # Print results
    print("--- Clustering Evaluation Results ---")
    # print(f"Number of Clusters (True vs. Predicted): {len(np.unique(Z_true))} vs. {len(np.unique(Z_pred))}")
    # print(f"Adjusted Rand Index (ARI):      {ari:.4f}  (1.0 is perfect, 0.0 is random)")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}  (1.0 is perfect)")
    print(f"Homogeneity:                    {homogeneity:.4f}  (each predicted cluster contains only members of a single class)")
    print(f"Completeness:                   {completeness:.4f}  (all members of a given class are assigned to the same cluster)")
    print(f"V-Measure:                      {v_measure:.4f}  (harmonic mean of homogeneity and completeness)")
    print("-----------------------------------")

def merge_clusters_by_majority_vote(Z_pred: np.ndarray, Z_true: np.ndarray) -> np.ndarray:
    """
    Merges predicted clusters based on the majority ground truth label within each predicted cluster.

    This function is very useful, especially when a clustering algorithm (like Louvain) 
    produces many more clusters than the true number of classes. It "corrects" the 
    predicted labels in the following way:
    1. Find all members of each predicted cluster.
    2. Look at the corresponding ground truth labels for these members.
    3. Find the ground truth label that appears most frequently (the "majority vote").
    4. Relabel all members of this predicted cluster with this "majority vote" ground truth label.

    Args:
        Z_pred (np.ndarray): The list of predicted cluster labels (which may have many clusters).
        Z_true (np.ndarray): The list of ground truth cluster labels (used as a reference).

    Returns:
        np.ndarray: A new label array (Z_merged), where the number of clusters is at most 
                    equal to the number of clusters in the ground truth labels.
    """
    if len(Z_pred) != len(Z_true):
        raise ValueError("The lengths of the predicted and true label lists must be the same.")
    
    # Ensure inputs are NumPy arrays
    Z_pred = np.array(Z_pred)
    Z_true = np.array(Z_true)
    
    # Step 1: Create a map from a predicted cluster label to the indices of nodes belonging to that cluster
    # e.g., {pred_cluster_0: [node_idx_1, node_idx_5, ...], ...}
    pred_to_nodes = defaultdict(list)
    for i, label in enumerate(Z_pred):
        pred_to_nodes[label].append(i)
        
    # Create a new array to store the merged results
    Z_merged = np.zeros_like(Z_pred)
    
    # Steps 2 & 3: Iterate through each predicted cluster to find the majority vote ground truth label
    for pred_label, node_indices in pred_to_nodes.items():
        # Get the true labels for all nodes in this predicted cluster
        true_labels_in_cluster = Z_true[node_indices]
        
        # If this cluster is empty, skip it (should not happen in normal cases)
        if len(true_labels_in_cluster) == 0:
            continue
            
        # Use Counter to find the most frequent true label
        # .most_common(1) returns a list, e.g., [(label, count)]
        majority_true_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
        
        # Step 4: Relabel all nodes in this predicted cluster with the majority vote label
        Z_merged[node_indices] = majority_true_label
        
    return Z_merged
