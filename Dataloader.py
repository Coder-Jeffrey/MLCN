import torch
import torch.utils.data as data_utils
import numpy as np
from utils import generate_adj, gen_A, gen_adj

def load_data(batch_size=32):
    train_input_ids = np.load('./Data/AAPD/train_input_ids.npy')
    train_input_mask = np.load('./Data/AAPD/train_input_mask.npy')
    train_token_type_ids = np.load('./Data/AAPD/train_token_type_ids.npy')
    train_label = np.load('./Data/AAPD/train_label.npy')
    train_fit_doc_position = np.load('./Data/AAPD/train_fit_doc_position.npy')
    train_fit_label_position = np.load('./Data/AAPD/train_fit_label_position.npy')
    test_input_ids = np.load('./Data/AAPD/test_input_ids.npy')
    test_input_mask = np.load('./Data/AAPD/test_input_mask.npy')
    test_token_type_ids = np.load('./Data/AAPD/test_token_type_ids.npy')
    test_label = np.load('./Data/AAPD/test_label.npy')
    test_fit_doc_position = np.load('./Data/AAPD/test_fit_doc_position.npy')
    test_fit_label_position = np.load('./Data/AAPD/test_fit_label_position.npy')

    result = generate_adj(train_label, num_classes=54)
    adj = torch.from_numpy(gen_A(54, result)).float()
    label_adj = gen_adj(adj)

    train_data = data_utils.TensorDataset(torch.from_numpy(train_input_ids), torch.from_numpy(train_input_mask),
                                          torch.from_numpy(train_token_type_ids), torch.from_numpy(train_label),
                                          torch.from_numpy(train_fit_doc_position), torch.from_numpy(train_fit_label_position))
    test_data = data_utils.TensorDataset(torch.from_numpy(test_input_ids), torch.from_numpy(test_input_mask),
                                         torch.from_numpy(test_token_type_ids), torch.from_numpy(test_label),
                                         torch.from_numpy(test_fit_doc_position), torch.from_numpy(test_fit_label_position))
    train_dataloader = data_utils.DataLoader(train_data, batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_data, batch_size, shuffle=False)
    return train_dataloader, test_dataloader, label_adj
