import numpy as np


def normalize_adj(a_adj):
    i = 0
    for adj in a_adj:
        row_sum = np.array(adj.sum(1))
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        a_adj[i] = adj_normalized

    return a_adj

def get_adj(text_mask, adj):
    result_adj = np.zeros((len(text_mask), 243, 243), dtype=float)
    map_word = np.zeros((len(text_mask), 100), dtype=int)
    for i in range(len(text_mask)):
        count = 0
        for j in range(len(text_mask[i])):
            if text_mask[i][j] == 1:
                map_word[i][count] = j + 1
                count = count + 1
    for i in range(len(text_mask)):
        for j in range(len(adj[i])):
            for k in range(len(adj[i][j])):
                if adj[i][j][k] != 0:
                    result_adj[i][map_word[i][j] - 1][map_word[i][k] - 1] = adj[i][j][k]
    result_adj = normalize_adj(result_adj)
    return result_adj


