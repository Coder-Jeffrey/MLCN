import os
import pickle
import scipy.sparse as sp
from utils import clean_str
from tqdm import tqdm
data_path = 'AAPD'

def load_data_and_labels(data):
    x_text = [doc['text'] for doc in data]

    labels = [doc['catgy'] for doc in data]

    row_idx, col_idx, val_idx, all_label = [], [], [], []
    for i in tqdm(range(len(labels))):
        l_list = list(set(labels[i])) # remove duplicate cateories to avoid double count
        all_label.extend(l_list)
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return [x_text, Y, set(all_label)]


def AAPD_data_got(type):
    """type:'train' or 'test'"""
    text = []
    labels = []
    with open('./Data/AAPD/aapd_' + type + '.tsv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            labels.append([float(k) for k in line[0]])
            text.append(line[1])
    return text, labels


def Other_data_got(data_path):
    with open('./Data/rcv1_raw_text.p', 'rb') as fin:
        [train, test, _, _] = pickle.load(fin, encoding='latin1')
    train_text, train_label, train_all_label = load_data_and_labels(train)
    test_text, test_label, test_all_label = load_data_and_labels(test)

    return train_text, train_label, test_text, test_label

def data_got(data_path):
    if data_path == 'AAPD':
        train_text, train_label = AAPD_data_got('train')
        test_text, test_label = AAPD_data_got('test')
    else:
        train_text, train_label, test_text, test_label = Other_data_got(data_path)

    return train_text, train_label, test_text, test_label
