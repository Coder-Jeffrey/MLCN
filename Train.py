import torch
from utils import precision_k, Ndcg_k, get_metrics
import numpy as np
from tqdm import tqdm

def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''

    assert x.size(1) == y.size(1)
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return x @ y.transpose(0, 1)

def better_Lcon(label_vec, text_embedding):
    #label_vec = torch.where(label_vec >= 0.5, torch.ones_like(label_vec), torch.zeros_like(label_vec)).cuda()
    C_ij = cosine_similarity(label_vec, label_vec)
    # Compute B_ij
    C_ij_sum = C_ij.sum(dim=1) - C_ij.diagonal()
    B_ij = torch.where(C_ij_sum.unsqueeze(1) > 0, C_ij / C_ij_sum.unsqueeze(1), torch.zeros_like(C_ij))
    # Compute text_dist, exp_dist, and exp_dist_sum
    text_embedding = torch.nn.functional.normalize(text_embedding, p=1, dim=-2)
    text_dist = torch.cdist(text_embedding, text_embedding, p=2)
    τ = 10
    exp_dist = torch.exp(-text_dist / τ)
    exp_dist_sum = exp_dist.sum(dim=1) - exp_dist.diagonal()

    # Compute L_con
    L_con = -(B_ij) * torch.log(exp_dist / exp_dist_sum.unsqueeze(1))

    # Compute the final result
    result = L_con.mean()

    return result.cuda()
def train_model(train_loader, device, model, optimizer, criterion):
    model.train()
    run_loss = 0.0
    prec_k = []
    ndcg_k = []
    real_labels = []
    preds = []
    iftrain = 1
    for batch_idx, data in enumerate(tqdm(train_loader), 0):
        optimizer.zero_grad()
        input_ids, input_mask, token_type_ids, labels, doc_position, label_position = data[0].long(), data[1].long(), data[2].long(), data[3].float(),\
                                                                                      data[4].type(torch.int64), data[5].type(torch.int64)
        input_ids, input_mask, token_type_ids, labels, doc_position, label_position = input_ids.to(device),input_mask.to(device), \
                                                                                      token_type_ids.to(device), labels.to(device),\
                                                                           doc_position.to(device), label_position.to(device)
        output,H_enc = model(input_ids, input_mask, token_type_ids, doc_position, label_position,labels,iftrain)
        closs = better_Lcon(output,H_enc)
        loss = criterion(output,labels) + 0.1*closs
        loss.backward()
        optimizer.step()
        run_loss += loss.data
        labels_cpu = labels.data.cpu().float()
        real_labels.extend(labels_cpu.tolist())
        preds_cpu = output.data.cpu()
        preds.extend(preds_cpu.tolist())
        prec = precision_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        ndcg_k.append(ndcg)
    metrics = get_metrics(real_labels, preds)
    epoch_prec = np.array(prec_k).mean(axis=0)
    epoch_ndcg = np.array(ndcg_k).mean(axis=0)
    loss = run_loss / (batch_idx+1)

    return loss, epoch_prec, epoch_ndcg, metrics


def test_model(test_loader, device, model, criterion):
    model.eval()
    run_loss = 0.0
    prec_k = []
    ndcg_k = []
    real_labels = []
    preds = []
    iftrain = 0
    for batch_idx, data in enumerate(tqdm(test_loader), 0):
        input_ids, input_mask, token_type_ids, labels, doc_position, label_position = data[0].long(), data[1].long(), data[2].long(), data[3].float(), data[4].type(torch.int64), data[5].type(torch.int64)
        input_ids, input_mask, token_type_ids, labels, doc_position, label_position = input_ids.to(device), input_mask.to(device),  token_type_ids.to(device), labels.to(device),\
                                                                                                 doc_position.to(device), label_position.to(device)
        output,H_enc = model(input_ids, input_mask, token_type_ids, doc_position, label_position,labels,iftrain)
        loss = criterion(output, labels)
        run_loss += loss.data
        labels_cpu = labels.data.cpu().float()
        real_labels.extend(labels_cpu.tolist())
        preds_cpu = output.data.cpu()
        preds.extend(preds_cpu.tolist())
        prec = precision_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        ndcg_k.append(ndcg)
    metrics = get_metrics(real_labels, preds)
    epoch_prec = np.array(prec_k).mean(axis=0)
    epoch_ndcg = np.array(ndcg_k).mean(axis=0)
    loss = run_loss / (batch_idx+1)
    return loss, epoch_prec, epoch_ndcg, metrics

