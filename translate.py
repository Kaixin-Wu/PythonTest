import torch
import codecs
import constants
import numpy as np
from model import Transformer
from torch.autograd import Variable
from utils import read_corpus, get_variable, id2word, word2id, to_input_variable

def length_penalty(length, alpha):
    """
    length: candidate translation length
    alpha : hyper params
    """
    return np.power(((5.0 + length) / 6.), alpha)


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

        for i in range(args.decode_max_steps):
            pred_var = to_input_variable([pred[:i+1]], vocab.tgt, cuda=args.cuda, is_test=True)
            scores = translator(test_var, pred_var)

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
