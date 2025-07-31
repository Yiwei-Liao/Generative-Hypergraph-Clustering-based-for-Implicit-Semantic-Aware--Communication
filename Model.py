import numpy as np
import random
import copy
import math
import time
import os
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from scipy.sparse import csc_matrix, find, bmat
import math


def _custom_multinomial(p: list) -> float:
    if not p:
        return 1.0
    n = sum(p)
    log_multinomial = math.lgamma(n + 1) - sum(math.lgamma(pi + 1) for pi in p)
    return np.exp(log_multinomial)

class Hypergraph:
    def __init__(self, N: range, E: Dict[int, Dict[Tuple[int, ...], int]], D: List[int]):
        self.N = N
        self.E = E
        self.D = D
    def __repr__(self) -> str:
        num_nodes = len(self.N)
        num_edges = sum(len(edges) for edges in self.E.values())
        return f"Hypergraph(nodes={num_nodes}, edges={num_edges})"

def partition_hypergraph_mle(H: Hypergraph, startclusters: str = "singletons", gamma: float = 1.0,
                             clusterpenalty: float = 0.0, maxits: int = 100,
                             randflag: bool = False, verbose: bool = True,
                             Z0: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(H.D) - 1

    if Z0 is None:
        if startclusters == "singletons":
            Z0 = np.arange(1, n + 1)
        elif startclusters == "cliquelouvain":
            Z0 = initialize_partition_with_clique_expansion(H, gamma)
        elif startclusters == "starlouvain":
            Z0 = initialize_partition_with_star_expansion(H, gamma)
        else:
            raise ValueError(f"Unknown startclusters method: {startclusters}")

    d = np.array(H.D, dtype=float)
    kmax = max(H.E.keys()) if H.E else 0

    He2n, weights = hypergraph2incidence(H)
    e2n_list = incidence2elist(He2n)
    n2e_list = incidence2elist(He2n.T)
    e2n = [[]] + e2n_list
    n2e = [[]] + n2e_list

    elen = np.array([len(edge) for edge in e2n_list], dtype=int)


    Z0_0based = Z0 - 1
    d_0based = d[1:]
    beta, gamma_param, omega = learn_mle_parameters(e2n_list, weights, Z0_0based, kmax, d_0based, n)

    Zwarm_1based = np.concatenate(([0], Z0))
    
    Zset = _core_semantic_partitioning(n2e, e2n, weights, d, elen, beta, gamma_param, kmax,
                                       randflag, maxits, verbose, Zwarm_1based, clusterpenalty)

    return Zset[:, -1]

def recover_implicit_semantics(H: 'Hypergraph',
                               beta: np.ndarray,
                               gamma: np.ndarray,
                               maxits: int = 100,
                               verbose: bool = True,
                               randflag: bool = False,
                               clusterpenalty: float = 0.0,
                               Z0: Optional[np.ndarray] = None,
                               log_community_changes: bool = False,
                               patience: int = 0,
                               return_move_history: bool = False
                              ) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """
    通过最大化似然函数来恢复超图中节点的隐式语义（即社区划分）。
    这是一个核心的、用户可调用的函数。
    """
    if 1 in H.E and H.E[1]:
        print("Warning: This code assumes hyperedges have at least two nodes.")

    if (len(beta) > 1 and beta[1] != 0) or (len(gamma) > 1 and gamma[1] != 0):
        print("Assuming no size-1 hyperedges. Setting beta[1] and gamma[1] to 0.")
        if len(beta) > 1: beta[1] = 0
        if len(gamma) > 1: gamma[1] = 0

    d, n, kmax, e2n_list, n2e_list, weights, elen = alternate_hypergraph_storage(H)

    assert len(beta) >= kmax + 1, f"beta vector length {len(beta)} is insufficient for kmax {kmax}"
    assert len(gamma) >= kmax + 1, f"gamma vector length {len(gamma)} is insufficient for kmax {kmax}"

    if Z0 is None:
        Z0 = np.arange(n)

    logfile_path = None
    if log_community_changes:
        pass 

    history = [] if return_move_history else None

    e2n_1based = [[]] + e2n_list
    n2e_1based = [[]] + n2e_list

    Zwarm_1based = np.concatenate(([0], Z0 + 1))

    Zset = _core_semantic_partitioning(n2e_1based, e2n_1based, weights, d, elen,
                                       beta, gamma, kmax, randflag, maxits, verbose,
                                       Zwarm_1based, clusterpenalty, logfile_path, patience,
                                       history)

    final_partition = Zset[:, -1]

    if return_move_history:
        return final_partition, history
    else:
        return final_partition

def _calculate_move_objective_delta(v: int, Z: np.ndarray, erest: List[int], Ci_ind: int, Cj_ind: int, we: float) -> float:
    if not erest:
        return 0
    p1 = Z[erest[0]]
    is_uniform = all(Z[node] == p1 for node in erest[1:])
    if not is_uniform:
        return 0
    else:
        if Ci_ind == p1: return we
        elif Cj_ind == p1: return -we
        else: return 0

def _refine_partition_pass(node2edges: List[List[int]], edge2nodes: List[List[int]],
                           w: np.ndarray, d: np.ndarray, elen: np.ndarray,
                           beta: np.ndarray, gamma_param: np.ndarray, kmax: int,
                           randflag: bool = False, maxits: int = 100, verbose: bool = True,
                           Zwarm: Optional[np.ndarray] = None, clusterpenalty: float = 0.0,
                           patience: int = 0,
                           history: Optional[List[int]] = None) -> Tuple[np.ndarray, bool]:
    if verbose:
        print("Executing one pass of semantic partition refinement")

    n = len(node2edges) - 1

    if Zwarm is not None and len(Zwarm) > 0:
        Z = renumber(Zwarm)
    else:
        Z = np.arange(n + 1)

    node_order = np.random.permutation(np.arange(1, n + 1)) if randflag else np.arange(1, n + 1)

    K_max_label = np.max(Z) if len(Z) > 1 else 0
    ClusVol = np.zeros(K_max_label + 1)
    ClusSize = np.zeros(K_max_label + 1, dtype=int)
    for i in range(1, n + 1):
        ClusVol[Z[i]] += d[i]
        ClusSize[Z[i]] += 1

    num_communities = np.sum(ClusSize > 0)

    if history is not None and not history:
        history.append(num_communities)

    cutpenalty = np.zeros(len(edge2nodes))
    for e in range(1, len(edge2nodes)):
        k = elen[e-1]
        if k < len(beta): cutpenalty[e] = beta[k] * w[e-1]
    Neighbs = neighbor_list(node2edges, edge2nodes)
    edgelists = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for e_idx in node2edges[i]:
            edge = edge2nodes[e_idx]
            edgelists[i].append([node for node in edge if node != i])

    iter_count = 0
    changemade = False
    consecutive_no_improvement_passes = 0
    toler = 1e-8

    while iter_count < maxits:
        iter_count += 1
        if verbose:
            print(f"  Partition refinement inner iteration {iter_count}")

        pass_improved = False
        for i in node_order:
            Ci_ind = Z[i]
            clus_size = ClusSize[Ci_ind]
            neighbor_nodes_array = np.array(Neighbs[i], dtype=int)
            if len(neighbor_nodes_array) == 0:
                if history is not None:
                    history.append(num_communities)
                continue

            NC = np.unique(Z[neighbor_nodes_array])
            BestC = Ci_ind
            BestImprove = 0
            Cv = node2edges[i]
            Cv_list = edgelists[i]
            vS = ClusVol[Ci_ind]
            dv = d[i]

            for Cj_ind in NC:
                if Cj_ind == Ci_ind: continue
                vJ = ClusVol[Cj_ind]
                delta_vol = 0
                for k in range(2, kmax + 1):
                    if k < len(gamma_param): delta_vol += gamma_param[k] * (pow(vS - dv, k) + pow(vJ + dv, k) - pow(vS, k) - pow(vJ, k))
                delta_cut = 0
                for eid, e_idx in enumerate(Cv):
                    edge_noi = Cv_list[eid]
                    if elen[e_idx-1] > 1: delta_cut += _calculate_move_objective_delta(i, Z, edge_noi, Ci_ind, Cj_ind, cutpenalty[e_idx])
                delta_clus = 0
                if clusterpenalty > 0 and clus_size == 1: delta_clus = clusterpenalty * (math.log(num_communities - 1) - math.log(num_communities)) if num_communities > 1 else 0
                change = delta_cut + delta_vol + delta_clus
                if change < BestImprove - toler:
                    BestImprove = change
                    BestC = Cj_ind

            if BestImprove < -toler:
                ci_old = Z[i]
                Z[i] = BestC
                ClusVol[ci_old] -= dv
                ClusSize[ci_old] -= 1
                ClusVol[BestC] += dv
                ClusSize[BestC] += 1
                changemade = True
                pass_improved = True
                if ClusSize[ci_old] == 0:
                    num_communities -= 1

            if history is not None:
                history.append(num_communities)

        if pass_improved:
            consecutive_no_improvement_passes = 0
        else:
            consecutive_no_improvement_passes += 1

        if consecutive_no_improvement_passes > patience:
            if verbose:
                print(f"  Stopping: No improvement for {patience + 1} consecutive passes.")
            break

    Z = renumber(Z)
    return Z, changemade

def _relabel_edges_by_partition(edge2nodes: List[List[int]], Z: np.ndarray, uniqueflag: bool = True):
    for e in range(1, len(edge2nodes)):
        new_edge = [Z[node_id] for node_id in edge2nodes[e]]
        if uniqueflag:
            edge2nodes[e] = sorted(list(set(new_edge)))
        else:
            edge2nodes[e] = sorted(new_edge)
    return edge2nodes

def _condense_hypergraph_by_partition(edge2nodes: List[List[int]], w: np.ndarray,
                                      label: Optional[np.ndarray] = None) -> Tuple[List[List[int]], np.ndarray, Optional[np.ndarray]]:
    h_dict = {}
    if label is None:
        for i, edge in enumerate(edge2nodes):
            edge.sort()
            key = tuple(edge)
            h_dict[key] = h_dict.get(key, 0.0) + w[i]
        newlist = [list(k) for k in h_dict.keys()]
        wnew = np.array(list(h_dict.values()))
        return newlist, wnew, None
    else:
        for i, edge in enumerate(edge2nodes):
            lab = label[i]
            edge.sort()
            key = (lab, *edge)
            h_dict[key] = h_dict.get(key, 0.0) + w[i]
        newlist, wnew, labelsnew = [], [], []
        for key, weight in h_dict.items():
            edge = list(key[1:])
            if len(edge) > 1:
                newlist.append(edge)
                wnew.append(weight)
                labelsnew.append(key[0])
        return newlist, np.array(wnew), np.array(labelsnew)

def _condense_degrees_by_partition(d: np.ndarray, Z: np.ndarray) -> np.ndarray:
    num_clusters = np.max(Z)
    dnew = np.zeros(num_clusters + 1)
    for i in range(1, num_clusters + 1):
        nodes_in_cluster_i = np.where(Z == i)[0]
        if len(nodes_in_cluster_i) > 0:
            dnew[i] = np.sum(d[nodes_in_cluster_i])
    return dnew

def _core_semantic_partitioning(node2edges: List[List[int]],
                                edge2nodes: List[List[int]], w: np.ndarray,
                                d: np.ndarray, elen: np.ndarray,
                                beta: np.ndarray, gamma_param: np.ndarray,
                                kmax: int, randflag: bool = False, maxits: int = 100, verbose: bool = True,
                                Zwarm: Optional[np.ndarray] = None, clusterpenalty: float = 0.0,
                                logfile_path: Optional[str] = None,
                                patience: int = 0,
                                history: Optional[List[int]] = None) -> np.ndarray:
    d_orig = d.copy()
    w_orig = w.copy()
    elen_orig = elen.copy()
    edge2nodes_orig = copy.deepcopy(edge2nodes)
    n_orig = len(d_orig) - 1

    if verbose: print("--- Semantic Partitioning Pass 1 ---")
    Z, improved = _refine_partition_pass(node2edges, edge2nodes, w_orig, d_orig, elen_orig,
                                         beta, gamma_param, kmax, randflag, maxits, verbose,
                                         Zwarm, clusterpenalty, patience, history)

    if not improved:
        return (Z[1:] - 1).reshape(-1, 1)

    Zs = (Z[1:] - 1).reshape(-1, 1)
    Z_old = Z.copy()

    pass_counter = 1
    while improved:
        pass_counter += 1
        if verbose: print(f"\n--- Semantic Partitioning Pass {pass_counter} ---")

        e2n_current_pass = copy.deepcopy(edge2nodes_orig)
        _relabel_edges_by_partition(e2n_current_pass, Z_old)
        e2n_super, w_super, elen_super = _condense_hypergraph_by_partition(e2n_current_pass[1:], w_orig, elen_orig)
        num_supernodes = np.max(Z_old)
        d_super = _condense_degrees_by_partition(d_orig, Z_old)
        He2n_super = elist2incidence(e2n_super, num_supernodes)
        n2e_super_list = incidence2elist(He2n_super.T)
        edge2nodes_super_1based = [[]] + e2n_super
        node2edges_super_1based = [[]] + n2e_super_list

        Z_super, improved = _refine_partition_pass(node2edges_super_1based, edge2nodes_super_1based,
                                                   w_super, d_super, elen_super,
                                                   beta, gamma_param, kmax, randflag,
                                                   maxits, verbose, None, clusterpenalty, patience,
                                                   history)

        if improved:
            Z_new = np.zeros(n_orig + 1, dtype=int)
            for i in range(1, n_orig + 1):
                Z_new[i] = Z_super[Z_old[i]]
            Zs = np.c_[Zs, Z_new[1:] - 1]
            Z_old = Z_new.copy()

    return Zs

def learn_mle_parameters(e2n: List[List[int]], weights: np.ndarray, Z: np.ndarray,
                         kmax: int, d: np.ndarray, n: int, debug_print: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if Z.size > 0:
        num_clusters = np.max(Z) + 1 if Z.size > 0 else 0
        ClusVol = np.bincount(Z, weights=d, minlength=num_clusters)
    else:
        ClusVol = np.array([], dtype=d.dtype)

    EdgesAndCuts = np.zeros((2, kmax + 1))
    EdgesAndCuts[0, :] = 0.01

    for j, edge in enumerate(e2n):
        we = weights[j]
        k = len(edge)
        if k <= kmax:
            EdgesAndCuts[0, k] += we
            edge_indices = np.array(edge) - 1
            clusters_in_edge = Z[edge_indices]
            is_cut = len(set(clusters_in_edge)) > 1
            if is_cut:
                EdgesAndCuts[1, k] += we

    omega = np.zeros((2, kmax + 1))
    volV = np.sum(ClusVol)

    for k in range(2, kmax + 1):
        volsums = np.sum(np.power(ClusVol, k))
        volVk = pow(volV, k)

        uncut_weight = EdgesAndCuts[0, k] - EdgesAndCuts[1, k]
        if volsums > 1e-12:
            omega[0, k] = uncut_weight / volsums

        cut_weight = EdgesAndCuts[1, k]
        vol_diff = volVk - volsums
        if vol_diff > 1e-12:
            omega[1, k] = cut_weight / vol_diff

    beta = np.zeros(kmax + 1)
    gamma_param = np.zeros(kmax + 1)

    for k in range(2, kmax + 1):
        log_omega0 = np.log(omega[0, k]) if omega[0, k] > 0 else -np.inf
        log_omega1 = np.log(omega[1, k]) if omega[1, k] > 0 else -np.inf

        beta[k] = log_omega0 - log_omega1

        if np.isinf(beta[k]):
            beta[k] = 1e3 # A large penalty if one of the omegas is zero

        gamma_param[k] = omega[0, k] - omega[1, k]

    return beta, gamma_param, omega

def hyperedge_formatting(H: Hypergraph) -> Tuple[List[List[int]], np.ndarray]:
    hyperedges = []
    weights = []
    for key in H.E:
        edges_in_k = list(H.E[key].keys())
        weights_in_k = list(H.E[key].values())
        hyperedges.extend([list(e) for e in edges_in_k])
        weights.extend(weights_in_k)
    return hyperedges, np.array(weights)

def elist2incidence(hyperedges: List[List[int]], n: int) -> csc_matrix:
    rows, cols = [], []
    m = len(hyperedges)
    for e_idx, edge in enumerate(hyperedges):
        for node_id in edge:
            rows.append(e_idx)
            cols.append(node_id - 1)
    data = np.ones(len(rows))
    return csc_matrix((data, (rows, cols)), shape=(m, n))

def hypergraph2incidence(H: Hypergraph) -> Tuple[csc_matrix, np.ndarray]:
    hyperedges, weights = hyperedge_formatting(H)
    n = len(H.D) - 1 if H.D else 0
    He2n = elist2incidence(hyperedges, n)
    return He2n, weights

def incidence2elist(H: csc_matrix, nodelist: bool = False) -> List[List[int]]:
    if not nodelist:
        H = H.T.tocsc()
    num_items = H.shape[1]
    output_list = []
    for i in range(num_items):
        start, end = H.indptr[i], H.indptr[i+1]
        item_indices = H.indices[start:end]
        multiplicities = H.data[start:end]
        current_list = []
        for k in range(len(item_indices)):
            item_id = item_indices[k] + 1
            mult = int(multiplicities[k])
            current_list.extend([item_id] * mult)
        output_list.append(current_list)
    return output_list

def neighbor_list(node2edge: List[List[int]], edge2node: List[List[int]]) -> List[List[int]]:
    n = len(node2edge) - 1
    neighbs = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        edges_of_i = node2edge[i]
        neighbor_set = set()
        for edge_idx in edges_of_i:
            nodes_in_edge = edge2node[edge_idx]
            neighbor_set.update(nodes_in_edge)
        neighbor_set.discard(i)
        neighbs[i] = sorted(list(neighbor_set))
    return neighbs

def renumber(c: np.ndarray) -> np.ndarray:
    _, c_new_0based = np.unique(c, return_inverse=True)
    return c_new_0based + 1

def clique_expansion(H: Hypergraph, weighted: bool = True, binary: bool = False) -> csc_matrix:
    n = len(H.D) - 1
    rows, cols, values = [], [], []
    valid_keys = [k for k in H.E.keys() if k > 1]
    for k in valid_keys:
        for edge, weight in H.E[k].items():
            edge_list = list(edge)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    ei, ej = edge_list[i], edge_list[j]
                    rows.extend([ei - 1, ej - 1])
                    cols.extend([ej - 1, ei - 1])
                    edge_weight = weight / (k - 1) if weighted else weight
                    values.extend([edge_weight, edge_weight])
    A = csc_matrix((values, (rows, cols)), shape=(n, n))
    A.setdiag(0)
    A.eliminate_zeros()
    if binary:
        A.data[:] = 1
    return A

def _construct_adj(A: csc_matrix) -> Tuple[List[List[int]], np.ndarray]:
    n = A.shape[0]
    neighbs = []
    for i in range(n):
        neighbors_of_i = A.indices[A.indptr[i]:A.indptr[i+1]]
        neighbs.append(neighbors_of_i.tolist())
    return neighbs, np.array(A.sum(axis=1)).flatten()

def _condense_graph_by_partition(A: csc_matrix, w: np.ndarray, c: np.ndarray) -> Tuple[csc_matrix, np.ndarray]:
    c_0based = c - 1
    num_clusters = c_0based.max() + 1
    wnew = np.zeros(num_clusters)
    for i in range(A.shape[0]):
        wnew[c_0based[i]] += w[i]
    rows, cols, values = [], [], []
    r_indices, c_indices, v_data = find(A)
    for i in range(len(r_indices)):
        u, v, weight = r_indices[i], c_indices[i], v_data[i]
        c_u, c_v = c_0based[u], c_0based[v]
        if c_u != c_v:
            rows.append(c_u)
            cols.append(c_v)
            values.append(weight)
    Anew = csc_matrix((values, (rows, cols)), shape=(num_clusters, num_clusters))
    return Anew, wnew

def _modularity_maximization_pass(A: csc_matrix, w: np.ndarray, lam: float, randflag: bool = False,
                                  maxits: float = float('inf'), clusterpenalty: float = 0.0) -> Tuple[np.ndarray, bool]:
    n = A.shape[0]
    p = np.random.permutation(n) if randflag else np.arange(n)
    undo_p = np.argsort(p)
    A_perm = A[p, :][:, p]
    w_perm = w[p]
    c = np.arange(n)
    K = n
    neighbs, _ = _construct_adj(A_perm)
    improving = True
    its = 0
    changemade = False
    while improving and its < maxits:
        its += 1
        improving = False
        for i_perm in range(n):
            Ci_ind = c[i_perm]
            nodes_in_Ci = np.where(c == Ci_ind)[0]
            pos_inner = A_perm[i_perm, nodes_in_Ci].sum()
            neg_inner = w_perm[i_perm] * (w_perm[nodes_in_Ci].sum() - w_perm[i_perm])
            total_inner = pos_inner - lam * neg_inner
            BestImprove = 0
            BestC = Ci_ind
            neighbor_nodes = neighbs[i_perm]
            if not neighbor_nodes: continue
            NC = np.unique(c[neighbor_nodes])
            for Cj_ind in NC:
                if Cj_ind == Ci_ind: continue
                nodes_in_Cj = np.where(c == Cj_ind)[0]
                pos_outer = A_perm[i_perm, nodes_in_Cj].sum()
                neg_outer = w_perm[i_perm] * w_perm[nodes_in_Cj].sum()
                total_outer = lam * neg_outer - pos_outer
                change = total_outer + total_inner
                if change < BestImprove:
                    BestImprove = change
                    BestC = Cj_ind
            if BestC != Ci_ind:
                c[i_perm] = BestC
                improving = True
                changemade = True
                if len(np.where(c == Ci_ind)[0]) == 0:
                    K -= 1
    _, final_c_0based = np.unique(c, return_inverse=True)
    return (final_c_0based + 1)[undo_p], changemade

def _iterative_modularity_maximization(A: csc_matrix, w: np.ndarray, lam: float, randflag: bool = False,
                                       maxits: int = 10000, clusterpenalty: float = 0.0) -> np.ndarray:
    n = A.shape[0]
    c, improved = _modularity_maximization_pass(A, w, lam, randflag, maxits, clusterpenalty)
    if not improved:
        return c.reshape(-1, 1)
    all_c = [c]
    c_old = c.copy()
    A_current = A.copy()
    w_current = w.copy()
    while improved:
        Anew, wnew = _condense_graph_by_partition(A_current, w_current, c_old)
        cSuper, improved = _modularity_maximization_pass(Anew, wnew, lam, randflag, maxits, clusterpenalty)
        if improved:
            c_new = np.zeros(n, dtype=int)
            for i in range(n):
                c_new[i] = cSuper[c_old[i] - 1]
            all_c.append(c_new)
            c_old = c_new.copy()
            A_current = Anew
            w_current = wnew
    return np.array(all_c).T

def partition_graph_by_modularity(A: csc_matrix, gamma: float = 1.0, **kwargs) -> np.ndarray:
    d = np.array(A.sum(axis=1)).flatten()
    vol = d.sum()
    if vol == 0: return np.arange(1, A.shape[0] + 1)
    lam = gamma / vol
    randflag = kwargs.get('randflag', False)
    clusterpenalty = kwargs.get('clusterpenalty', 0.0)
    maxits = kwargs.get('maxits', 10000)
    Cs = _iterative_modularity_maximization(A, d, lam, randflag, maxits, clusterpenalty)
    return Cs[:, -1]

def initialize_partition_with_clique_expansion(H: Hypergraph, gamma: float = 1.0, **kwargs) -> np.ndarray:
    A = clique_expansion(H, kwargs.get('weighted', True), kwargs.get('binary', False))
    return partition_graph_by_modularity(A, gamma=gamma, **kwargs)

def initialize_partition_with_star_expansion(H: Hypergraph, gamma: float = 1.0, **kwargs) -> np.ndarray:
    He2n, _ = hypergraph2incidence(H)
    A = bmat([[None, He2n.T], [He2n, None]], format='csc')
    Za = partition_graph_by_modularity(A, gamma=gamma, **kwargs)
    return Za[:He2n.shape[1]]

def alternate_hypergraph_storage(H: 'Hypergraph') -> Tuple[np.ndarray, int, int, List[List[int]], List[List[int]], np.ndarray, np.ndarray]:
    d = np.array(H.D, dtype=float)
    n = len(d) - 1
    kmax = max(H.E.keys()) if H.E else 0
    He2n, weights = hypergraph2incidence(H)
    e2n = incidence2elist(He2n)
    n2e = incidence2elist(He2n.T)
    elen = np.array([len(edge) for edge in e2n], dtype=int)
    return d, n, kmax, e2n, n2e, weights, elen

def _omega_to_betagamma(omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kmax = omega.shape[1] - 1
    safe_omega = np.maximum(omega, 1e-50)
    beta = np.log(safe_omega[0, :]) - np.log(safe_omega[1, :])
    gamma = omega[0, :] - omega[1, :]
    return beta, gamma

def _allsame(items: List) -> bool:
    if not items: return True
    return all(x == items[0] for x in items[1:])

def _partitionize(items: List) -> List[int]:
    counts = Counter(items)
    return list(counts.values())

def calculate_partition_likelihood(H: 'Hypergraph',
                                   Z: np.ndarray,
                                   omega: np.ndarray,
                                   likelihood: bool = False) -> Union[float, Tuple[float, float]]:
    D_np = np.array(H.D)
    beta, gamma = _omega_to_betagamma(omega)
    n = len(Z)
    if not H.E: return (0.0, 0.0) if likelihood else 0.0
    kmax = max(H.E.keys())
    kmin = min(H.E.keys())
    mvec = np.zeros(kmax + 1)
    for k in range(kmin, kmax + 1):
        if k in H.E:
            for weight in H.E[k].values():
                mvec[k] += weight
    num_clusters = np.max(Z) + 1 if n > 0 else 0
    ClusVol = np.bincount(Z, weights=D_np[1:n+1], minlength=num_clusters)
    obj = 0.0
    for k in range(kmin, kmax + 1):
        if k not in H.E: continue
        cut_penalty_k = 0
        for edge, weight in H.E[k].items():
            clusters_of_edge = Z[np.array(edge) - 1]
            if not _allsame(clusters_of_edge.tolist()):
                cut_penalty_k += weight * beta[k]
        obj -= cut_penalty_k
        volume_penalty_k = np.sum(ClusVol ** k) * gamma[k]
        obj -= volume_penalty_k
    if likelihood:
        loglikhood = obj
        for k in range(kmin, kmax + 1):
            if k not in H.E: continue
            obj += mvec[k] * np.log(omega[1, k])
            loglikhood += mvec[k] * np.log(omega[1, k])

            for edge, weight in H.E[k].items():
                p = _partitionize(Z[np.array(edge) - 1].tolist())
                bR = _custom_multinomial(p)
                degrees_of_edge = D_np[np.array(edge)]
                term = np.prod(degrees_of_edge) * omega[1, k] * bR
                obj -= term
                loglikhood -= term

                aR = weight
                loglikhood += aR * np.log(np.prod(degrees_of_edge))
                loglikhood -= (aR * np.log(bR) - math.lgamma(aR + 1))
        return obj, loglikhood
    else:
        return obj
