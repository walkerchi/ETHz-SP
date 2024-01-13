import torch 
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
from enum import Enum

_INF = 1 + 1e10

class SeedingHeuristic(Enum):
    MinAve = 1
    NearCenter = 2


def _BFS_dist(adj_list, n_nodes, seed, mask=None):
    # mask: meaning only search within the subset indicated by it, any outside nodes are not reachable
    #       can be achieved by marking outside nodes as visited, dist to inf
    res = np.full((n_nodes,), fill_value=_INF) 
    vistied = np.zeros((n_nodes,), dtype=np.bool_)
    if isinstance(seed, list):
        for s in seed:
            res[s] = 0
            vistied[s] = True
        frontier = seed
    else:
        res[seed] = 0
        vistied[seed] = True
        frontier = [seed]

    if isinstance(mask, list):
        for i, m in enumerate(mask):
            if m != True:
                res[i] = _INF
                vistied[i] = True

    depth = 0
    track = [frontier]
    while frontier:
        this_level = frontier
        depth += 1
        frontier = []
        while this_level:
            f = this_level.pop(0)
            for n in adj_list[f]:
                if not vistied[n]:
                    vistied[n] = True
                    frontier.append(n)
                    res[n] = depth
        # record each level
        track.append(frontier)

    return res, track

def _BFS_dist_all(adj_list, n_nodes):
    res = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        res[i], _ = _BFS_dist(adj_list, n_nodes, i)
    return res


def _find_clusters(adj_list, mask=None):
    n_nodes = len(adj_list)
    if isinstance(mask, list):
        remaining_nodes = []
        for i, m in enumerate(mask):
            if m == True:
                remaining_nodes.append(i)
    else:
        remaining_nodes = list(range(n_nodes))
    cluster = []
    while remaining_nodes:
        if len(remaining_nodes) > 1:
            seed = remaining_nodes[0]
            dist, _ = _BFS_dist(adj_list, n_nodes, seed, mask)
            tmp = []
            new_remaining = []
            for n in remaining_nodes:
                if dist[n] != _INF:
                    tmp.append(n)
                else:
                    new_remaining.append(n)
            cluster.append(tmp)
            remaining_nodes = new_remaining
        else:
            cluster.append([remaining_nodes[0]])
            break

    return cluster

def _flat_edge_to_adj_list(edge_list, n=None):
    if n == None:
        sender_node = set(edge_list[0])
        receiv_node = set(edge_list[1])
        all_node = list(sender_node.union(receiv_node))
        n = max(all_node) + 1
    adj_list = [[] for _ in range(n)]
    for i in range(len(edge_list[0])):
        adj_list[edge_list[0, i]].append(edge_list[1, i])

    return adj_list

def _flat_edge_to_adj_mat(edge_list, n=None):
    if n is None:
        sender_node = set(edge_list[0])
        receiv_node = set(edge_list[1])
        all_node = list(sender_node.union(receiv_node))
        n = max(all_node) + 1
    adj_mat = scipy.sparse.coo_array((np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n))

    return adj_mat

def _min_ave_seed(adj_list, clusters):
    seeds = []
    dist = _BFS_dist_all(adj_list, len(adj_list))
    for c in clusters:
        d_c = dist[c]
        d_c = d_c[:, c]
        d_sum = np.sum(d_c, axis=1)
        min_ave_depth_node = c[np.argmin(d_sum)]
        seeds.append(min_ave_depth_node)

    return seeds

def _nearest_center_seed(adj_list, clusters, pos_mesh):
    seeds = []
    for c in clusters:
        center = np.mean(pos_mesh[c], axis=0)
        dd = pos_mesh[c] - center[None, :]
        normd = np.linalg.norm(dd, 2, axis=-1)
        thresh_d = np.min(normd) * 1.2
        tmp = np.where(normd < thresh_d)[0].tolist()
        try_node = [c[i] for i in tmp]
        # print(try_node)
        min_node = try_node[0]
        d_min, _ = _BFS_dist(adj_list, len(adj_list), min_node)
        min_d_sum = np.sum(d_min)
        for i in range(1, len(try_node)):
            trial = try_node[i]
            d_trial, _ = _BFS_dist(adj_list, len(adj_list), trial)
            d_trial_sum = np.sum(d_trial)
            if d_trial_sum < min_d_sum:
                min_node = trial
                min_d_sum = d_trial_sum
        seeds.append(min_node)

    return seeds

def _adj_mat_to_flat_edge(adj_mat):
    adj_mat = adj_mat.tocoo()
    s, r = adj_mat.row, adj_mat.col
    dat = adj_mat.data
    valid = np.where(dat.astype(bool))[0]
    s, r = s[valid], r[valid]
    return np.stack([s, r],0)

def pool_edge(g, idx, num_nodes):
    # g in scipy sparse mat
    g = _adj_mat_to_flat_edge(g)  # now flat edge list
    # idx is list
    idx = np.array(idx, dtype=np.longlong)
    idx_new_valid = np.arange(len(idx)).astype(np.longlong)
    idx_new_all = -1 * np.ones(num_nodes).astype(np.longlong)
    idx_new_all[idx] = idx_new_valid
    new_g = -1 * np.ones_like(g).astype(np.longlong)
    new_g[0] = idx_new_all[g[0]]
    new_g[1] = idx_new_all[g[1]]
    both_valid = np.logical_and(new_g[0] >= 0, new_g[1] >= 0)
    e_idx = np.where(both_valid)[0]
    new_g = new_g[:, e_idx]

    return new_g


def bstride_selection(flat_edge, seed_heuristic=SeedingHeuristic.NearCenter, pos_mesh=None, n=None):
    combined_idx_kept = set()
    adj_list = _flat_edge_to_adj_list(flat_edge, n=n)
    adj_mat = _flat_edge_to_adj_mat(flat_edge, n=n)
    # adj mat enhance the diag
    adj_mat.setdiag(1)
    # 0. compute clusters, each of which should be deivded independantly
    clusters = _find_clusters(adj_list)
    # 1. seeding: by BFS_all for small graphs, or by seed_heuristic for larger graphs

    if seed_heuristic == SeedingHeuristic.NearCenter:
        seeds = _nearest_center_seed(adj_list, clusters, pos_mesh)
    else:
        seeds = _min_ave_seed(adj_list, clusters)

    for seed, c in zip(seeds, clusters):
        n_c = len(c)
        odd = set()
        even = set()
        index_kept = set()
        dist_from_cental_node, _ = _BFS_dist(adj_list, len(adj_list), seed)
        for i in range(len(dist_from_cental_node)):
            if dist_from_cental_node[i] % 2 == 0 and dist_from_cental_node[i] != _INF:
                even.add(i)
            elif dist_from_cental_node[i] % 2 == 1 and dist_from_cental_node[i] != _INF:
                odd.add(i)

        # 4. enforce n//2 candidates
        if len(even) <= len(odd) or len(odd) == 0:
            index_kept = even
            index_rmvd = odd
            delta = len(index_rmvd) - len(index_kept)
        else:
            index_kept = odd
            index_rmvd = even
            delta = len(index_rmvd) - len(index_kept)

        if delta > 0:
            # sort the dist of idx rmvd
            # cal stride based on delta nodes to select
            # generate strided idx from rmvd idx
            # union
            index_rmvd = list(index_rmvd)
            dist_id_rmvd = np.array(dist_from_cental_node)[index_rmvd]
            sort_index = np.argsort(dist_id_rmvd)
            stride = len(index_rmvd) // delta + 1
            delta_idx = sort_index[0::stride]
            delta_idx = set([index_rmvd[i] for i in delta_idx])
            index_kept = index_kept.union(delta_idx)

        combined_idx_kept = combined_idx_kept.union(index_kept)

    combined_idx_kept = list(combined_idx_kept)
    adj_mat = adj_mat.tocsr().astype(float)
    adj_mat = adj_mat @ adj_mat
    adj_mat.setdiag(0)
    adj_mat = pool_edge(adj_mat, combined_idx_kept, n)

    return combined_idx_kept, adj_mat
