from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import math
import torch
import ast


class datastore():
    def __init__(self):
        self.datastore_tensor = []
        self.x_label_tensor = []
        self.true_label = []
        self.datastore()

    def datastore(self):
        f = open('./AAPD_datastore.txt', encoding='utf-8')
        result = f.read()
        datastore = result.split("\n")
        datastore_tensor = []
        x_label_tensor = []
        true_label = []
        for item in datastore:
            if item == "":
                continue
            x = ast.literal_eval(item)
            datastore_tensor.append(torch.tensor(x['h']).cuda())
            x_label_tensor.append(torch.tensor(x['p']).cuda())
            true_label.append(torch.tensor(x['t']).cuda())

        self.datastore_tensor = torch.stack(datastore_tensor)
        self.x_label_tensor = torch.stack(x_label_tensor)
        self.true_label = torch.stack(true_label)


    def __iter__(self):
        return zip(self.datastore_tensor, self.x_label_tensor)

    def better_knn(self, labels, text_embedding, k, weight):
        dist_tensor = torch.sqrt(
            torch.sum(torch.square(text_embedding.unsqueeze(1) - self.datastore_tensor.unsqueeze(0)), dim=2))
        cosine_sim_tensor = torch.nn.functional.cosine_similarity(labels.unsqueeze(1),
                                                                  self.x_label_tensor.unsqueeze(0), dim=2)
        new_dis = (1 - weight) * dist_tensor + weight * dist_tensor * (torch.ones_like(cosine_sim_tensor)
                                                                       - cosine_sim_tensor)
        topk_dist, topk_index = torch.topk(new_dis, k, dim=1, largest=False)

        result = topk_index.cuda()
        dist_r = topk_dist.cuda()
        label_r = self.x_label_tensor[topk_index].cuda()
        exp_dist_r = torch.exp(-dist_r)
        l_result = torch.zeros(len(labels), len(labels[0])).cuda()
        for xx in range(len(result)):
            for yy in range(k):
                l = (exp_dist_r[xx][yy] / torch.sum(exp_dist_r[xx])) * label_r[xx][yy]
                l_result[xx] = l + l_result[xx]

        return l_result.cuda()

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EncoderModel(nn.Module):
    def __init__(self, num_labels, label_adj, pretrained_model='./pretrain/bert_base_uncase'):
        super(EncoderModel, self).__init__()
        self.num_labels = num_labels
        self.label_adj = label_adj.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.GCN3 = GraphConvolution(768, 768)
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.GCN4 = GraphConvolution(768, 768)
        self.GCN6 = GraphConvolution(768, 768)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.GCN5 = GraphConvolution(768, 768)
        self.conv = torch.nn.Conv2d(1, num_labels, kernel_size=[9, 9], padding=4)
        self.linear1 = nn.Linear(768*2, 768)
        self.linear = nn.Linear(768, 54)
        self.w_att = torch.nn.LeakyReLU(0.2)
        self.loss_func = torch.nn.BCELoss()

        self.device = 'cuda:0'

    def normalize_adj(self, adj):
        row_sum = torch.tensor(adj.sum(1))
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(adj, d_mat_inv_sqrt).transpose(1,2),d_mat_inv_sqrt)
        return adj_normalized


    def forward(self, input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position,labels,iftrain):
        bert_output = self.bert_model(input_ids, input_mask, token_type_ids)
        output = bert_output[0]

        token_hidden_size = output.shape[-1]
        doc_seq_length = output.shape[-2] - self.num_labels - 3
        doc_embedding = self.gather_indexes(output, fit_docspace_position)
        doc_embedding = torch.reshape(doc_embedding, [-1, doc_seq_length, token_hidden_size])
        # label update
        label_embedding = self.gather_indexes(output, fit_labelspace_position)
        label_embedding = torch.reshape(label_embedding, [-1, self.num_labels, token_hidden_size])

        label_embedding = self.GCN3(label_embedding, self.label_adj)
        label_embedding = self.relu1(label_embedding)
        label_embedding = self.GCN4(label_embedding, self.label_adj)
        # # dual GCN update adj
        A = torch.sigmoid(torch.bmm(self.linear_label1(label_embedding),self.linear_label2(label_embedding).transpose(1,2)))
        A = self.normalize_adj(A)
        d_label_embedding = self.GCN5(label_embedding, A)
        d_label_embedding = self.relu2(d_label_embedding)
        d_label_embedding = self.GCN6(d_label_embedding, A)
        label_embedding = torch.cat((label_embedding,d_label_embedding),-1)
        label_embedding = self.linear1(label_embedding)

        label_embedding = torch.nn.functional.normalize(label_embedding, p=1, dim=-2)
        doc_embedding = torch.nn.functional.normalize(doc_embedding, p=1, dim=-2)
        #
        word_label_att = torch.bmm(doc_embedding, label_embedding.transpose(1, 2))
        word_label_att = word_label_att.unsqueeze(1)

        Att_v = self.conv(word_label_att)

        Att_v = torch.max(Att_v, dim=1)[0]
        Att_v = torch.max(Att_v, keepdim=True, dim=-1)[0]
        Att_v_tanh = torch.tanh(Att_v)
        H_enc = Att_v_tanh * doc_embedding
        H_enc = torch.sum(H_enc, dim=1)
        label_output = torch.sigmoid(self.linear(H_enc))

        return label_output,H_enc

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = sequence_tensor.shape
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = torch.reshape(
            torch.range(0, batch_size-1, dtype=torch.int64) * seq_length, [-1, 1]).type(torch.int64).cuda()
        positions = positions.type(torch.int64)
        flat_positions = torch.reshape(positions + flat_offsets, [-1, 1])
        flat_positions = flat_positions.expand([flat_positions.shape[0], width])
        flat_sequence_tensor = torch.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = torch.gather(flat_sequence_tensor, 0, flat_positions)
        return output_tensor
