import math
import torch
import numpy as np
import constants
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F

def label_smoothing_loss(logits, gold_words, pad_idx=constants.PAD, epsilon=0.1, cuda=True):

    # one_hot distribution
    one_hot_template = torch.zeros_like(logits).view(-1)
    idxs = torch.LongTensor([i*logits.size(-1) for i in range(logits.size(0))])
    gold_words_flatten = gold_words + Variable(idxs.cuda(), volatile=False, requires_grad=False)
    one_hot_template[gold_words_flatten] = 1.
    one_hot_gold = one_hot_template.view(-1, logits.size(-1))

    # soft distribution
    smoothing_gold = one_hot_gold * (1 - epsilon) + (1 - one_hot_gold) * (epsilon / (logits.size(-1) - 1))
    log_softmax_scores = F.log_softmax(logits, dim=-1)
    smoothing_loss = -smoothing_gold * log_softmax_scores

    # mask the padding
    padding_mask =  gold_words.ne(pad_idx)
    padding_mask_smoothing = padding_mask.unsqueeze(1).float()
    smoothing_loss_padding = smoothing_loss * padding_mask_smoothing

    # statistic correct words
    _, argmax_idxs = torch.max(logits, dim=-1)
    equals_batch_tokens = argmax_idxs.eq(gold_words)
    equals_batch_tokens_padding = equals_batch_tokens.long() * padding_mask.long()
    correct_tokens_sum = torch.sum(equals_batch_tokens_padding)

    return smoothing_loss_padding, correct_tokens_sum

def get_one_hot(K=100):

    one_hot_distribution = torch.eye(K)
    one_hot_distribution = one_hot_distribution.cuda()

    one_hot_distribution = Variable(one_hot_distribution, volatile=False, requires_grad=False)
    # one_hot_gold = one_hot_distribution[gold_tgt_tokens_flatten]
    return one_hot_distribution


def label_smoothing(inputs, num_class, epsilon=0.1, cuda=True):

    one_hot_distribution = Variable(torch.eye(num_class))
    one_hot_gold = one_hot_distribution(inputs)

    smoothing_gold = one_hot_gold * (1 - epsilon) + (1 - one_hot_gold) * epsilon / (num_class - 1)
    if cuda:
        smoothing_gold = smoothing_gold.cuda()

    return smoothing_gold

def zipped(src_sents, tgt_sents):
    zipped_data = []
    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        zipped_data.append((src_sent, tgt_sent))

    return zipped_data

def get_threshold(n_in, n_out=None):
    if n_out:
        return math.sqrt(6. / (n_in + n_out))
    return math.sqrt(3. / n_in)

def read_corpus(file_path, source):
    data = []
    inData = open(file_path, "r", encoding='UTF-8').readlines()
    print("[Corpus] total %d" % len(inData))
    for line in inData:
        sent = line.strip().split(' ')

        # append <EOS> to the source sentence
        if source == 'src':
            sent  += [constants.EOS_WORD]

        # append <BOS> and <EOS> to the target sentence
        if source == 'tgt':
            sent = [constants.BOS_WORD] + sent + [constants.EOS_WORD]
        data.append(sent)

    return data

def word_batch_slice(data, word_batch_size, sort=True):
    """
    word batch
    """
    batch_size = word_batch_size // len(data[0][0])
    batch_num = int(np.ceil(len(data) / float(word_batch_size / len(data[0][0]))))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents

def sentence_batch_slice(data, batch_size, sort=True):
    """
    sentence batch
    """

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents

def data_iter(data, batch_size, batch_type="sentence_batch", shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)

        if batch_type == "sentence_batch":
            batched_data.extend(list(sentence_batch_slice(tuples, batch_size)))
        if batch_type == "word_batch":
            batched_data.extend(list(word_batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def tensor_transform(linear, X):
    # X is a 3D tensor
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)

def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]

def input_format(sents, pad_token):

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    pos = []
    for i in range(batch_size):
        sents_t.append([sents[i][k] if len(sents[i]) > k else pad_token for k in range(max_len)])
        pos.append([ k+1 if len(sents[i]) > k else pad_token for k in range(max_len)])

    return sents_t, pos

def get_variable(idxs, cuda=False, is_test=False):

    sents_t, pos = input_format(idxs, constants.PAD)

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    pos_var = Variable(torch.LongTensor(pos), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        pos_var = pos_var.cuda()

    return sents_var, pos_var

def to_input_variable(sents, vocab, cuda=False, is_test=False):

    """ Return a tensor of shape(batch_size, seq_max_len) """

    word_ids = word2id(sents, vocab)
    sents_t, pos = input_format(word_ids, vocab[constants.PAD_WORD])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    pos_var = Variable(torch.LongTensor(pos), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        pos_var = pos_var.cuda()

    return  sents_var, pos_var
