import math
import time
import torch
import subprocess
import constants
from utils import data_iter, to_input_variable

def evaluate_loss(model, data, criterion, pad_idx=constants.PAD, cuda=True):
    model.eval()

    total_loss = 0.
    total_tgt_words = 0
    total_correct_tokens = 0
    for src_sents, tgt_sents in data_iter(data, batch_size=model.args.valid_batch_size, batch_type=model.args.batch_type, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents)

        src = to_input_variable(src_sents, model.vocab.src, cuda=cuda)
        tgt = to_input_variable([item[:-1] for item in tgt_sents], model.vocab.tgt, cuda=cuda)

        gold_tgt_sents, _ = to_input_variable([item[1:] for item in tgt_sents], model.vocab.tgt, cuda=cuda)
        gold_tgt_tokens_flatten = gold_tgt_sents.view(-1)

        scores = model(src, tgt)
        loss = criterion(scores, gold_tgt_tokens_flatten)

        _, argmax_idxs = torch.max(scores, dim=-1)
        equals_batch_tokens = argmax_idxs.eq(gold_tgt_tokens_flatten)
        padding_mask = gold_tgt_tokens_flatten.ne(pad_idx)
        equals_batch_tokens_padding = equals_batch_tokens.long() * padding_mask.long()
        correct_tokens = torch.sum(equals_batch_tokens_padding)

        total_loss += loss.item()
        total_tgt_words += pred_tgt_word_num
        total_correct_tokens += correct_tokens.item()

    loss = total_loss / total_tgt_words
    ppl = math.exp(loss)
    acc = 1.0 * total_correct_tokens / total_tgt_words * 100
    return loss, ppl, acc

def inference(model, data, cuda=True, verbose=False, tmp_dir="tmp_work/"):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    model.eval()
    if cuda:
        model.cuda()

    hypotheses = []
    best_candidate = []
    begin_time = time.time()

    if len(data[0]) == 2:
        source_example = " ".join(data[0][0])
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)
            best_candidate.append(hyps[0])

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Target: ', ' '.join(tgt_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))
    else:
        source_example = " ".join(data[0])
        for src_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)
            best_candidate.append(hyps[0])

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))

    total_time = time.time() - begin_time
    print("[Decoing] [%d sentences] [time: %.2f]" % (len(list(data)), total_time))
    print("[Source Example] %s" % source_example)
    print("[Decode Example] %s" % " ".join(best_candidate[0]))

    BLEU_score = 0
    if model.args.external_valid_script is not None:
        decode_file = tmp_dir + str(int(time.time())) + ".tmp"
        outfile = open(decode_file, "w")

        for item in best_candidate:
            outfile.write(" ".join(item[1:-1]) + "\n")
        outfile.close()

        status, output = subprocess.getstatusoutput(model.args.external_valid_script + " " + decode_file)
        lines = output.strip().split("\n")