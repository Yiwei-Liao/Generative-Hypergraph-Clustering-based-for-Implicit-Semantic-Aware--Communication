import os
import argparse
import logging
import numpy as np
from collections import defaultdict
from typing import List, Set

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# This function is to remain unchanged as per user request, except for the path
# ==============================================================================
def load_graph_data(dataset_name, show_details=False):
    """
    Loads a graph dataset from .npy files.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'cora', 'citeseer').
                            The function will look for files in the 'data_origin/{dataset_name}/' directory.
        show_details (bool): Whether to print detailed information about the dataset.

    Returns:
        feat (np.ndarray): Node feature matrix.
        label (np.ndarray): Array of node labels (0-based).
        adj (np.ndarray): Adjacency matrix.
    """
    # MODIFIED: Changed input directory from "dataset" to "data_origin"
    load_path_base = os.path.join("data_origin", dataset_name)
    load_path = os.path.join(load_path_base, dataset_name)
    
    feat_path = load_path + "_feat.npy"
    label_path = load_path + "_label.npy"
    adj_path = load_path + "_adj.npy"
    
    if not all(os.path.exists(p) for p in [feat_path, label_path, adj_path]):
        # MODIFIED: Updated error message to reflect the new path
        logging.error(f"Required .npy files not found in the '{load_path_base}/' directory.")
        logging.error(f"Please ensure the following files exist: \n- {os.path.basename(feat_path)}\n- {os.path.basename(label_path)}\n- {os.path.basename(adj_path)}")
        return None, None, None

    feat = np.load(feat_path, allow_pickle=True)
    label = np.load(label_path, allow_pickle=True)
    adj = np.load(adj_path, allow_pickle=True)
    
    if show_details:
        print(f"\n--- Dataset Details: {dataset_name} ---")
        print(f"Node feature matrix shape: {feat.shape}")
        print(f"Node label array shape:   {label.shape}")
        print(f"Adjacency matrix shape:      {adj.shape}")
        print(f"Number of nodes:            {adj.shape[0]}")
        # Handles both sparse and dense matrices
        num_edges = adj.nnz if hasattr(adj, 'nnz') else np.count_nonzero(adj)
        print(f"Number of undirected edges: {int(num_edges / 2)}")
        
        min_lbl, max_lbl = min(label), max(label)
        num_categories = max_lbl - min_lbl + 1
        print(f"Number of classes:            {num_categories}")
        print("Class distribution:")
        for i in range(min_lbl, max_lbl + 1):
            count = np.sum(label == i)
            print(f"  - Class {i}: {count} samples")
        print("------------------------------------\n")
            
    return feat, label, adj
# ==============================================================================

def build_adj_list_from_matrix(adj_matrix: np.ndarray) -> defaultdict[int, set[int]]:
    """Builds an adjacency list (0-based node IDs) from an adjacency matrix."""
    logging.info("Building adjacency list from matrix...")
    adj_list = defaultdict(set)
    # Efficiently find all edge locations (u, v) using np.nonzero
    edges = np.transpose(np.nonzero(adj_matrix))
    for u, v in edges:
        if u != v:  # ignore self-loops
            adj_list[u].add(v)
            adj_list[v].add(u) # ensure the graph is undirected
            
    logging.info("Adjacency list built successfully.")
    return adj_list

def find_maximal_cliques(
    adj: defaultdict[int, set[int]], 
    min_size: int = 3
) -> List[List[int]]:
    """
    Finds all maximal cliques in a graph using the Bron-Kerbosch algorithm with pivoting.
    The returned cliques have 0-based node IDs.
    """
    cliques: List[List[int]] = []
    nodes = list(adj.keys())
    
    def bron_kerbosch_pivot(R: Set[int], P: Set[int], X: Set[int]):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(sorted(list(R)))
            return
        if not P:
            return

        try:
            pivot = next(iter(P | X))
            P_without_neighbors_of_pivot = P - adj[pivot]
        except StopIteration:
            # This can happen if P or X is empty, handles the edge case.
            P_without_neighbors_of_pivot = P

        for v in list(P_without_neighbors_of_pivot):
            bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    logging.info(f"Starting search for maximal cliques of size >= {min_size}...")
    bron_kerbosch_pivot(set(), set(nodes), set())
    
    logging.info(f"Clique search complete. Generated {len(cliques)} hyperedges.")
    return cliques

def write_output_files(
    output_path: str,
    dataset_name: str,
    hyperedges_0based: List[List[int]],
    labels_0based: np.ndarray
) -> None:
    """Writes the generated hypergraph data to files, converting IDs to 1-based."""
    final_output_dir = os.path.join(output_path, dataset_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    output_hyperedges_file = os.path.join(final_output_dir, f'hyperedges-{dataset_name}.txt')
    output_labels_file = os.path.join(final_output_dir, f'node-labels-{dataset_name}.txt')
    output_names_file = os.path.join(final_output_dir, f'label-names-{dataset_name}.txt')

    logging.info(f"Writing hyperedges file to: {output_hyperedges_file}")
    with open(output_hyperedges_file, 'w') as f:
        for h_edge in hyperedges_0based:
            # Convert 0-based node IDs to 1-based
            h_edge_1based = [node + 1 for node in h_edge]
            f.write(",".join(map(str, h_edge_1based)) + "\n")

    logging.info(f"Writing node labels file to: {output_labels_file}")
    with open(output_labels_file, 'w') as f:
        # Convert 0-based labels to 1-based
        for label_id in labels_0based:
            f.write(f"{label_id + 1}\n")

    logging.info(f"Writing class name mapping file to: {output_names_file}")
    min_label, max_label = min(labels_0based), max(labels_0based)
    # Map label IDs starting from 1
    label_name_map = {i + 1: f"Class_{i}" for i in range(min_label, max_label + 1)}
    with open(output_names_file, 'w') as f:
        for label_id, label_name in sorted(label_name_map.items()):
            f.write(f"{label_id},{label_name}\n")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate a hypergraph from a graph in NumPy format using Clique Expansion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help="Name of the dataset, e.g., 'cora'. The program will look for files in the 'data_origin/<name>/' directory."
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        # CHANGED: Default output path is now 'data'
        default='data', 
        help="Root directory to store the output hypergraph files. Output will be saved in '<output_path>/<dataset_name>/'."
    )
    parser.add_argument(
        '--min_clique_size', 
        type=int, 
        default=2, 
        help="Minimum clique size to be considered a hyperedge. Setting to 2 will include all original edges."
    )
    args = parser.parse_args()

    try:
        # 1. Load data
        _, labels, adj = load_graph_data(args.dataset, show_details=True)
        if adj is None:
            raise RuntimeError("Data loading failed. Please check file paths and content.")

        # 2. Build adjacency list
        adj_list = build_adj_list_from_matrix(adj)

        # 3. Find cliques (hyperedges)
        hyperedges = find_maximal_cliques(adj_list, args.min_clique_size)
        
        # 4. Write to files
        write_output_files(args.output_path, args.dataset, hyperedges, labels)
        
        final_dir = os.path.join(args.output_path, args.dataset)
        logging.info("\nAll processing is complete!")
        logging.info(f"Generated files are located at: {os.path.abspath(final_dir)}")

    except (FileNotFoundError, RuntimeError) as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unknown error occurred during processing: {e}", exc_info=True)


if __name__ == '__main__':
    main()
