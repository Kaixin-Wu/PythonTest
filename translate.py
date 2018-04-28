import torch
import codecs
import constants
import numpy as np
from model import Transformer
import torch.nn.functional as F
from my_beam_search import Beam_Search
from torch.autograd import Variable
from utils import read_corpus, get_variable, id2word, word2id, to_input_variable

def length_penalty(length, alpha):
    """
    length: candidate translation length
    alpha : hyper params
    """
    return np.power(((5.0 + length) / 6.), alpha)

def decode(args):
    """ Decode with beam search """

    # load vocabulary
    vocab = torch.load(args.vocab)

    # build model
    translator = Transformer(args, vocab)
    translator.eval()

    # load parameters
    translator.load_state_dict(torch.load(args.decode_model_path))
    if args.cuda:
        translator = translator.cuda()

    test_data = read_corpus(args.decode_from_file, source="src")
    output_file = codecs.open(args.decode_output_file, "w", encoding="utf-8")
    for test in test_data:
        test_seq, test_pos = to_input_variable([test], vocab.src, cuda=args.cuda, is_test=True)
        test_seq_beam = test_seq.expand(args.decode_beam_size, test_seq.size(1))

        enc_output = translator.encode(test_seq, test_pos)
        enc_output_beam = enc_output.expand(args.decode_beam_size, enc_output.size(1), enc_output.size(2))

        beam = Beam_Search(beam_size=args.decode_beam_size, tgt_vocab=vocab.tgt)
        for i in range(args.decode_max_steps):

            # the first time for beam search
            if i == 0:
                # <BOS>
                pred_var = to_input_variable(beam.candidates[:1], vocab.tgt, cuda=args.cuda, is_test=True)
                scores = translator.translate(enc_output, test_seq, pred_var)
            else:
                pred_var = to_input_variable(beam.candidates, vocab.tgt, cuda=args.cuda, is_test=True)
                scores = translator.translate(enc_output_beam, test_seq_beam, pred_var)

            log_softmax_scores = F.log_softmax(scores, dim=-1)
            log_softmax_scores = log_softmax_scores.view(pred_var[0].size(0), -1, log_softmax_scores.size(-1))
            log_softmax_scores = log_softmax_scores[:, -1, :]

            is_done = beam.advance(log_softmax_scores)
            beam.update_status()

            if is_done:
                break

        print("[Source] %s" % " ".join(test))
        print("[Predict] %s" % beam.get_best_candidate())
        print()

        output_file.write(beam.get_best_candidate() + "\n")
        output_file.flush()

    output_file.close()

def test(args):
    """ Test function """

    # load vocabulary
    vocab = torch.load(args.vocab)

    # build model
    translator = Transformer(args, vocab)
    translator.eval()

    # load parameters
    translator.load_state_dict(torch.load(args.decode_model_path))
    if args.cuda:
        translator = translator.cuda()

    test_data = read_corpus(args.decode_from_file, source="src")
    # ['<BOS>', '<PAD>', 'PAD', '<PAD>', '<PAD>']
    pred_data = len(test_data) * [[constants.PAD_WORD if i else constants.BOS_WORD for i in range(args.decode_max_steps)]]

    output_file = codecs.open(args.decode_output_file, "w", encoding="utf-8")
    for test, pred in zip(test_data, pred_data):
        pred_output = [constants.PAD_WORD] * args.decode_max_steps
        test_var = to_input_variable([test], vocab.src, cuda=args.cuda, is_test=True)

        # only need one time
        enc_output = translator.encode(test_var[0], test_var[1])
        for i in range(args.decode_max_steps):
            pred_var = to_input_variable([pred[:i+1]], vocab.tgt, cuda=args.cuda, is_test=True)

            scores = translator.translate(enc_output, test_var[0], pred_var)

            _, argmax_idxs = torch.max(scores, dim=-1)
            one_step_idx = argmax_idxs[-1].data[0]

            pred_output[i] = vocab.tgt.id2word[one_step_idx]
            if (one_step_idx == constants.EOS) or (i == args.decode_max_steps-1):
                print("[Source] %s" % " ".join(test))
                print("[Predict] %s" % " ".join(pred_output[:i]))
                print()

                output_file.write(" ".join(pred_output[:i])+"\n")
                output_file.flush()
                break
            pred[i+1] = vocab.tgt.id2word[one_step_idx]

    output_file.close()
