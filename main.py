import Load_datasets
import os
import Model 
import numpy as np
import Evaluation
from sklearn.metrics import accuracy_score
import Visualize

print("--- Listing available datasets ---")
Load_datasets.hypermodularity_datasets()
print("\n--- Loading a sample dataset ---")
dataname = "cora" 
print(f"--- Loading data for '{dataname}' ---")

H, labels, node_features = Load_datasets.read_hypergraph_data(
    dataname, 
    return_labels=True, 
    return_features=True,
    maxsize=5,
    minsize=2
)

unique_labels = sorted(np.unique(labels))
class_names = [f'Class {i}' for i in unique_labels]

if node_features is None:
    print("无法加载节点特征，将跳过t-SNE可视化。")
    ifvisualize = False
else:
    ifvisualize = True

d, n, kmax, e2n, n2e, weights, edge_lengths = Model.alternate_hypergraph_storage(H)

d_1based = np.array(H.D, dtype=float) 

d_0based = d_1based[1:]

assert len(d_0based) == n, f"d_0based length {len(d_0based)} does not match node count {n}"

randflag = False
verbose = False
maxits = 100
print("--- Step 1: Initializing partition and learning parameters ---")
Z = Model.initialize_partition_with_clique_expansion(H, randflag=False)
print(f"Initial partition found with {len(np.unique(Z))} clusters.")
assert len(Z) == n, f"Z length {len(Z)} does not match node count {n}"
beta, gamma, omega = Model.learn_mle_parameters(e2n, weights, Z, kmax, d_0based, n)
print("Initial parameters (beta, gamma, omega) learned.")
# print("Initial beta:\n", beta)
# print("Initial gamma:\n", gamma)
# print("Initial omega:\n", omega)

print("\n--- Step 2: Iteratively recovering semantics and updating parameters ---")
for i in range(4):
    print(f"\n--- Main Loop Iteration {i+1} ---")
    Z = Model.recover_implicit_semantics(H, beta, gamma, maxits=maxits, verbose=verbose, randflag=False, log_community_changes=True, patience=100)
    obj_val, log_lik = Model.calculate_partition_likelihood(H, Z, omega, likelihood=True)
    beta, gamma, omega = Model.learn_mle_parameters(e2n, weights, Z, kmax, d_0based, n, debug_print=False)   
    nk = len(np.unique(Z))
    print("Updated omega:\n", omega)
    print(f"Objective Value = {obj_val:.4f}, Log-Likelihood = {log_lik:.4f}, Found {nk} clusters.")

# 3. 评估结果
print("\n--- Step 3: Evaluating the final partition ---")
Evaluation.evaluate_clustering(labels, Z)
print("Learned intra/inter-community connection probabilities (omega):\n", omega)
print("Max hyperedge size:", max(H.E.keys()))
print("Min hyperedge size:", min(H.E.keys()))

print("\n--- Evaluating merged clusters ---")
Z_merged = Evaluation.merge_clusters_by_majority_vote(Z, labels)
accuracy = accuracy_score(labels, Z_merged)
print(f"Accuracy after merging clusters: {accuracy:.4f}")
print("---------------------------------")


