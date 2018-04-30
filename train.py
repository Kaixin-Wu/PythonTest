import os
import codecs
import torch
import time
import math
import argparse
import torch.nn as nn

from vocab import Vocab, VocabEntry
from torch.nn.utils import clip_grad_norm

import constants
import torch.optim
from optim import ScheduledOptim
from model import Transformer
from evaluate import evaluate_loss
from translate import test, decode
from utils import read_corpus, zipped, data_iter, to_input_variable, label_smoothing_loss

def config_initializer():
    """ Config initializer """

    parser = argparse.ArgumentParser()

    parser.add_argument('-option', type=str, default="train", choices=["train", "test"])

    # training phase
    parser.add_argument('-cuda', action='store_true', default=True, help="Use GPU")
    parser.add_argument('-epochs', default=20, type=int, help="Max epochs")
    parser.add_argument('-batch_type', default="sentence_batch", type=str, choices=['sentence_batch', 'word_batch'],
                        help="1.sentence_batch(Sentence number for each batch) 2.word_batch(token number for each batch)")
    parser.add_argument('-batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('-max_seq_len', default=120, type=int, help="Using for Recording positional encoding")

    parser.add_argument('-params_initializer', default='uniform', choices=['uniform', 'normal'], type=str)
    parser.add_argument('-params_scale', default=1.0, type=float)
    parser.add_argument('-optimizer', default="Adam", type=str, choices=['Warmup_Adam', 'Adam', 'SGD'],
                        help="Optimizer methods")
    parser.add_argument('-positional_encoding', default='sinusoid', type=str, choices=['sinusoid', 'learned'],
                        help="Positional encoding methods")
    parser.add_argument('-lr', default=0.0001, type=float, help="Learning rate")

    parser.add_argument('-d_model', default=512, type=int)
    parser.add_argument('-d_inner_hid', default=512, type=int)
    parser.add_argument('-d_k', default=64, type=int)
    parser.add_argument('-d_v', default=64, type=int)

    parser.add_argument('-n_head', default=8, type=int)
    parser.add_argument('-n_layers_enc', default=2, type=int, help="Layers of encoder block")
    parser.add_argument('-n_layers_dec', default=2, type=int, help="Layers of decoder block")

    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-n_warmup_steps', default=4000, type=int)
    parser.add_argument('-embs_share_weight', default=False, type=bool)
    parser.add_argument('-proj_share_weight', default=False, type=bool)

    parser.add_argument('-label_smoothing', default=False, type=bool, help="Label smoothing")
    parser.add_argument('-label_smoothing_rate', default=0.1, type=float)

    parser.add_argument('-finetune', default=False, type=bool)
    parser.add_argument('-finetune_model_path', default="./models/transformer_epoch4", type=str)

    parser.add_argument('-decode_max_steps', default=120, type=int)
    parser.add_argument('-clip_grad', default=0., type=float, help="Clip gradient")
    parser.add_argument('-save_mode', default='best', choices=['all', 'best'], type=str)

    parser.add_argument('-displayFreq', default=50, type=int, help='Evaluate train set frequency(batch)')
    parser.add_argument('-validFreq', default=100, type=int, help='Evaluate valid set frequency(batch)')
    parser.add_argument('-save_to', default='./models/', type=str, help="Path of the saving model")
    parser.add_argument('-checkpoint', default='./models/checkpoint', type=str, help="Record the models")
    parser.add_argument('-external_eval_script', default=None, type=str, help="External scripts for evaluating(BLEU)")

    parser.add_argument('-vocab', default='./2000sents/vocab.bin', type=str, help="Path of the vocabulary binary file")
    parser.add_argument('-train_src', default='./2000sents/train.ch', type=str, help="Path of the training source file")
    parser.add_argument('-train_tgt', default='./2000sents/train.en', type=str, help="Path of the training target file")
    parser.add_argument('-valid_src', default='./2000sents/valid.ch', type=str, help="Path of the valid source file")
    parser.add_argument('-valid_tgt', default='./2000sents/valid.en', type=str, help="Path of the valid target file")

    # inference phase
    parser.add_argument('-decode_model_path', type=str, default="./models/transformer_epoch4", help="Path of the model file")
    parser.add_argument('-decode_from_file', type=str, default="./2000sents/valid.ch", help="Path of the input file to decode")
    parser.add_argument('-decode_output_file', type=str, default="./2000sents/decode.output", help="Output of the decode file")

    parser.add_argument('-decode_alpha', type=float ,default=1.3)
    parser.add_argument('-decode_beam_size', type=int, default=4, help="Beam size")
    parser.add_argument('-decode_batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('-n_best', type=int, default=1, help="If verbose is set, will output the n_best decoded sentences")

    args = parser.parse_args()
    return args

def init_training(args):
    """ Initialize training process """

    # load vocabulary
    vocab = torch.load(args.vocab)

    # build model
    transformer = Transformer(args, vocab)

    # if finetune
    if args.finetune:
        print("[Finetune] %s" % args.finetune_model_path)
        transformer.load_state_dict(torch.load(args.finetune_model_path))

    # vocab_mask for masking padding
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt[constants.PAD_WORD]] = 0

    # loss object
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        transformer = transformer.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    if args.optimizer == "Warmup_Adam":
        optimizer = ScheduledOptim(
            torch.optim.Adam(
                transformer.get_trainable_parameters(),
                betas=(0.9, 0.98), eps=1e-09),
                args.d_model, args.n_warmup_steps)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=transformer.get_trainable_parameters(),
                                     lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=transformer.get_trainable_parameters(), lr=args.lr)

    return vocab, transformer, optimizer, cross_entropy_loss

def main(args):
    """ Main function """

    # training data
    print("****** Train Set *****")
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    print()

    # valid data
    print("****** Valid Set *****")
    valid_data_src = read_corpus(args.valid_src, source='src')
    valid_data_tgt = read_corpus(args.valid_tgt, source='tgt')
    print()

    # merge data for source and target
    train_data = zipped(train_data_src, train_data_tgt)
    valid_data = zipped(valid_data_src, valid_data_tgt)

    vocab, transformer, optimizer, cross_entropy_loss = init_training(args)
    print("[Transformer Config] ",)
    print(transformer)

    epoch = 0
    checkpoint = codecs.open(args.checkpoint, "w", encoding="utf-8")
    transformer.train()
    while epoch < args.epochs:

        total_loss = 0.
        total_tgt_words = 0
        total_correct_tokens = 0
        freq = 0
        start_epoch = start_batch = time.time()

        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size, batch_type=args.batch_type):

            # sum for predicting target words per batch(no padding)
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents)

            optimizer.zero_grad()

            # format data for source and target(add padding)
            src = to_input_variable(src_sents, vocab.src, cuda=args.cuda)
            tgt = to_input_variable([item[:-1] for item in tgt_sents], vocab.tgt, cuda=args.cuda)

            # scores for predicting(before softmax)
            scores = transformer(src, tgt)

            gold_tgt_sents, _ = to_input_variable([item[1:] for item in tgt_sents], vocab.tgt, cuda=args.cuda)
            gold_tgt_tokens_flatten = gold_tgt_sents.view(-1)

            # get loss according cross_entropy(one_hot distribution)
            weight_loss = cross_entropy_loss(scores, gold_tgt_tokens_flatten)
            mean_loss = weight_loss / pred_tgt_word_num

            # get loss according cross_entropy(smoothing distribution)
            if args.label_smoothing:
                smoothing_loss = label_smoothing_loss(scores, gold_tgt_tokens_flatten,
                                                      epsilon=args.label_smoothing_rate)
                smoothing_mean_loss = smoothing_loss / pred_tgt_word_num

            _, pred_idxs = torch.max(scores, dim=-1)
            is_target = gold_tgt_tokens_flatten.ne(constants.PAD)
            correct_tokens = torch.sum(gold_tgt_tokens_flatten.eq(pred_idxs).float() * is_target.float())

            if args.label_smoothing:
                smoothing_mean_loss.backward()
            else:
                mean_loss.backward()

            optimizer.step()

            if args.optimizer == "Warmup_Adam":
                optimizer.update_learning_rate()

            total_loss += mean_loss.data[0]
            total_correct_tokens += correct_tokens.data[0]
            total_tgt_words += pred_tgt_word_num

            freq += 1
            if freq % args.displayFreq == 0:
                end_batch = time.time()
                total_time = end_batch - start_batch
                aver_per_word_loss = total_loss / args.displayFreq
                acc = 1.0 * total_correct_tokens / total_tgt_words * 100

                print("[%d] [loss:%5.2f] [acc:%5.2f%%] [ppl:%5.2f] [speed:%5.2f words/s] [time:%5.2fs]" %
                      (freq, aver_per_word_loss, acc, math.exp(aver_per_word_loss), total_tgt_words / total_time,
                       total_time))

                total_loss = 0.
                total_tgt_words = 0
                total_correct_tokens = 0
                start_batch = end_batch

            if freq % args.validFreq == 0:
                t0 = time.time()
                valid_loss, ppl, acc = evaluate_loss(transformer, valid_data, cross_entropy_loss)
                t1 = time.time()
                print("[Valid] [loss:%5.2f] [acc:%5.2f%%] [ppl:%5.2f] [time:%5.2fs]" %
                      (valid_loss, acc, ppl, t1-t0))

        epoch += 1
        end_epoch = time.time()
        print("[Epoch %d] is ending... [total_time:%.2f min]"% (epoch, (end_epoch - start_epoch) / 60))

        print("Saving model...")
        if not os.path.isdir(args.save_to):
            os.makedirs(args.save_to)

        if args.finetune:
            torch.save(transformer.state_dict(), args.finetune_model_path + "_finetune_epoch%d" % (epoch))
            checkpoint.write(args.finetune_model_path + "_finetune_epoch%d\n" % (epoch))
        else:
            torch.save(transformer.state_dict(), args.save_to + "transformer_epoch%d" % (epoch))
            checkpoint.write(args.save_to + "transformer_epoch%d\n" % (epoch))

        checkpoint.flush()
        print("Saving finish...\n")
    checkpoint.close()

if __name__ == "__main__":

    args = config_initializer()
    print("[Config]")
    print(args)


    if args.option == "train":
        main(args)

    if args.option == "test":
        decode(args)

