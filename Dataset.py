import numpy as np
from transformers import BertTokenizer
from data_got import data_got
from tqdm import tqdm
from global_config import get_aapd_schema_set
import scipy.sparse as sp

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length - 3:
            break
        ##cut sentence frist deal with tokens_a
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _general_token_b_and_seq_label(label_trans_token):
    token_b = []
    token_b_ids = []
    for k, v in label_trans_token.items():
        token_b.append(k)
        token_b_ids.append(v)
    return token_b, token_b_ids


def single_convert_to_token(tokenizer, text, label_list):

    tokens_a = tokenizer._tokenize(text)
    bias=1
    label_trans_token = {}
    for (i, label) in enumerate(label_list):
        if i+bias<100:
            label_trans_token[label] = i + bias
        else:
            label_trans_token[label] = i + bias + 4
    token_b, token_b_ids = _general_token_b_and_seq_label(label_trans_token)
    if token_b_ids:
        _truncate_seq_pair(tokens_a, token_b, 300)
    else:
        if len(tokens_a) > 300 - 2:
            tokens_a = tokens_a[0:(300 - 2)]
    tokens_a = tokenizer.convert_tokens_to_ids(tokens_a)

    fit_docspace_position = [k for k in range(1, len(tokens_a) + 1)]
    fit_labelspace_position = [m for m in range(len(tokens_a)+2, len(tokens_a)+2+len(token_b))]
    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, token_b_ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, token_b_ids)
    input_mask = [1] * len(input_ids)

    doc_idx = len(tokens_a)+2+len(token_b)

    while len(input_ids) < 300:
        input_ids.append(0)
        input_mask.append(0)
        token_type_ids.append(0)
        doc_idx += 1
        fit_docspace_position.append(doc_idx)

    return input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position


def token2ids(data_path):
    train_text, train_label, test_text, test_label = data_got(data_path)
    tokenizer = BertTokenizer.from_pretrained('./pretrain/bert_base_uncase')
    label_list = get_aapd_schema_set().split('#')
    train_input_ids = []
    train_input_mask = []
    train_token_type_ids = []
    train_fit_doc_position = []
    train_fit_label_position = []
    test_input_ids = []
    test_input_mask = []
    test_token_type_ids = []
    test_fit_doc_position = []
    test_fit_label_position = []
    for text in tqdm(test_text):
        input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position = single_convert_to_token(tokenizer, text, label_list)
        test_input_ids.append(input_ids)
        test_input_mask.append(input_mask)
        test_token_type_ids.append(token_type_ids)
        test_fit_doc_position.append(fit_docspace_position)
        test_fit_label_position.append(fit_labelspace_position)
    for text in tqdm(train_text):
        input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position = single_convert_to_token(tokenizer, text, label_list)
        train_input_ids.append(input_ids)
        train_input_mask.append(input_mask)
        train_token_type_ids.append(token_type_ids)
        train_fit_doc_position.append(fit_docspace_position)
        train_fit_label_position.append(fit_labelspace_position)
    np.save('./Data/AAPD/train_input_ids.npy', np.array(train_input_ids))
    np.save('./Data/AAPD/train_input_mask.npy', np.array(train_input_mask))
    np.save('./Data/AAPD/train_token_type_ids.npy', np.array(train_token_type_ids))
    np.save('./Data/AAPD/train_label.npy', np.array(train_label))
    np.save('./Data/AAPD/train_fit_doc_position.npy', np.array(train_fit_doc_position))
    np.save('./Data/AAPD/train_fit_label_position.npy', np.array(train_fit_label_position))
    np.save('./Data/AAPD/test_input_ids.npy', np.array(test_input_ids))
    np.save('./Data/AAPD/test_input_mask.npy', np.array(test_input_mask))
    np.save('./Data/AAPD/test_token_type_ids.npy', np.array(test_token_type_ids))
    np.save('./Data/AAPD/test_label.npy', np.array(test_label))
    np.save('./Data/AAPD/test_fit_doc_position.npy', np.array(test_fit_doc_position))
    np.save('./Data/AAPD/test_fit_label_position.npy', np.array(test_fit_label_position))

if __name__ == '__main__':
    train_text, train_label, test_text, test_label = data_got('AAPD')

