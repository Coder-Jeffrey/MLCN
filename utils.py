import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)

        mat = np.multiply(score_mat, true_mat)

        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)


def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)

    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)


def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)


def get_fre(y, y_pred, num_label):
    result = np.zeros(num_label, dtype=float)
    for i in range(len(y)):
        for j in range(num_label):
            if y[i][j] == 1:
                result[j] = result[j]+1
    return y,y_pred

def get_metrics(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    zeros = np.zeros_like(y_pred)
    ones = np.ones_like(y_pred)
    y_pred = np.where(y_pred >= 0.5, ones, zeros)
    hamming_loss = metrics.hamming_loss(y, y_pred)
    micro_f1 = metrics.f1_score(y, y_pred, average='micro')
    micro_precision = metrics.precision_score(y, y_pred, average='micro')
    micro_recall = metrics.recall_score(y, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y, y_pred, average='macro')
    macro_precision = metrics.precision_score(y, y_pred, average='macro')
    macro_recall = metrics.recall_score(y, y_pred, average='macro')
    return [hamming_loss, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]


def generate_adj(label, num_classes):
    num = np.zeros(num_classes, dtype=float)
    adj = np.zeros((num_classes, num_classes), dtype=float)
    label = label.tolist()
    for i in tqdm(range(len(label))):
        real_label = label[i]
        label_index = [k for k in range(num_classes) if real_label[k] == 1]
        n = len(label_index)
        for j in range(n):
            num[label_index[j]] += 1
            s = j + 1
            while s <= n - 1:
                adj[label_index[j]][label_index[s]] += 1.
                adj[label_index[s]][label_index[j]] += 1.
                s = s + 1

    return {'nums': num, 'adj': adj}


def gen_A(num_classes, result):
    _adj = result['adj']
    _nums = result['nums']
    #_nums[101] = 1.0
    #_nums[102] = 1.0

    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = _adj + np.identity(num_classes, np.float64)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj