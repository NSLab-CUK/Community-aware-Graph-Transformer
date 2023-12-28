import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from itertools import chain
import copy, torch, dgl


def GetProbTranMat(Ak, num_node):
    num_node, num_node2 = Ak.shape
    if (num_node != num_node2):
        print('M must be a square matrix!')
    Ak_sum = np.sum(Ak, axis=0).reshape(1, -1)
    Ak_sum = np.repeat(Ak_sum, num_node, axis=0)
    probTranMat = np.log(np.divide(Ak, Ak_sum)) - np.log(1. / num_node)
    probTranMat[probTranMat < 0] = 0;  # set zero for negative and -inf elements
    probTranMat[np.isnan(probTranMat)] = 0;  # set zero for nan elements (the isolated nodes)
    return probTranMat


def getM_logM(nx_g, num_nodes, kstep=3):
    tran_M = []
    tran_logM = []
    Adj = np.zeros((num_nodes, num_nodes))
    for src in nx_g.nodes():
        src_degree = nx_g.degree(src)
        for dst in nx_g.nodes():
            if nx_g.has_edge(src, dst):
                Adj[src][dst] = round(1 / src_degree, 3)

    Ak = np.matrix(np.identity(num_nodes))
    for i in range(kstep):
        Ak = np.dot(Ak, Adj)
        tran_M.append(Ak)
        probTranMat = GetProbTranMat(Ak, num_nodes)
        tran_logM.append(probTranMat)

    return tran_M, tran_logM


def get_distance(deg_A, deg_B):
    damp = 1 / np.sqrt(deg_A * deg_B)  
    return damp   


def get_B_sim_phi(nx_g, tran_M, num_nodes, n_class, X, kstep=5):
    print(f'processing get_B_sim_phi')
    count = 0
    B = np.zeros((num_nodes, num_nodes))
    colour = np.zeros((num_nodes, num_nodes))
    phi = np.zeros((num_nodes, num_nodes, 1))
    sim = np.zeros((num_nodes, num_nodes, kstep))

    trans_check = tran_M[kstep - 1]
    not_adj = tran_M[0]

    kmeans = KMeans(n_clusters=n_class, init='k-means++', max_iter=50, n_init=10, random_state=0)

    y_kmeans = kmeans.fit_predict(X)
    count = 0
    count_1 = 0
    for src in nx_g.nodes():
        if count % 50 == 0:
            print(f' processing node_th {src}/{num_nodes}')
        for dst in nx_g.nodes():

            if src == dst:
                continue

            if not_adj[src, dst] > 0:
                continue

            if colour[src, dst] == 1 or colour[src, dst] == 1:
                continue
            if trans_check[src, dst] > 0.001:

                src_d = nx_g.degree(src)
                dst_d = nx_g.degree(dst)

                if np.abs(src_d - dst_d) > 1:
                    continue

                if y_kmeans[src] != y_kmeans[dst]:
                    continue
                else:
                    count_1 += 1
                    d = get_distance(src_d, dst_d)
                    # B i, j
                    B[src, dst] = d
                    B[dst, src] = d
                    # phi i,j
                    if phi[src, dst] == 0:
                        phi[src, dst] = d
                        phi[dst, src] = d

            colour[src, dst] = 1
            colour[dst, src] = 1
        B[src, src] = 0
        count += 1

     

    sim = compute_sim(tran_M, num_nodes, k_step=kstep)

    return B, sim, phi


def compute_sim(tran_M, num_nodes, k_step=5):
    sim = np.zeros((num_nodes, num_nodes, k_step))
    trans_check = tran_M[k_step - 1]

    for step in range(k_step):
        print(f'compute_sim transition step {step + 1}/{k_step}')
        colour = np.zeros((num_nodes, num_nodes))
        trans_k = copy.deepcopy(tran_M[step])
        trans_k[trans_k >= 0.001] = 1
        trans_k[trans_k < 0.001] = 0
        trans_k = np.array(trans_k)

        row_sums = trans_k.sum(axis=1)
        trans_mul = trans_k @ trans_k.T
        for i in range(num_nodes):

            for j in range(i + 1, num_nodes):
                if trans_check[i, j] < 0.0001:
                    continue
                if colour[i, j] == 1 or colour[j, i] == 1:
                    continue

                score = np.round(trans_mul[i, j] / (row_sums[i] + row_sums[j] - trans_mul[i, j]), 4)
                if score < 0.001:
                    score = 0
                sim[i, j, step] = score
                sim[j, i, step] = score

                colour[i, j] = 1
                colour[j, i] = 1
    return sim


def get_A_D(nx_g, num_nodes):
    num_edges = nx_g.number_of_edges()

    d = np.zeros((num_nodes))

    Adj = np.zeros((num_nodes, num_nodes))

    for src in nx_g.nodes():
        src_degree = nx_g.degree(src)
        d[src] = src_degree
        for dst in nx_g.nodes():
            if nx_g.has_edge(src, dst):
                Adj[src][dst] = 1


    return Adj, d, num_edges


def load_dgl(nx_g, x, sim, phi):
    print('loading dgl...')
    count = 0
    edge_idx1 = []
    edge_idx2 = []
    for e in nx_g.edges:
        edge_idx1.append(e[0])
        edge_idx2.append(e[1])

        edge_idx1.append(e[1])
        edge_idx2.append(e[0])

    s_vals = []
    phi_vals = []
    for i in range(len(edge_idx1)):
        count += 1
        n1 = edge_idx1[i]
        n2 = edge_idx2[i]

        s = np.asarray(sim[n1][n2], dtype=float)
        s_vals.append(s)

        p = np.asarray(phi[n1][n2], dtype=float)
        phi_vals.append(p)

    print(f'networkx: number edges: {count}')
    s_vals = np.array(s_vals)
    phi_vals = np.array(phi_vals)

    s_vals[np.isnan(s_vals)] = 0
    s_vals = normalize(s_vals, axis=0, norm='max')
    phi_vals = normalize(phi_vals, axis=0, norm='max')

    s_vals = torch.tensor(s_vals)
    phi_vals = torch.tensor(phi_vals)

    g = dgl.graph((edge_idx1, edge_idx2))

    s_vals[torch.isnan(s_vals)] = 0
    phi_vals[torch.isnan(phi_vals)] = 0

    g.ndata['x'] = x
    g.edata['sim'] = s_vals
    g.edata['phi'] = phi_vals

    print(f'loading dgl, done, DGL graph edges: {g.number_of_edges()}')
    return g

