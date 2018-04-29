import torch
import constants
from utils import length_penalty

class Beam_Search(object):

    """
        Store necessary information for beam search process.
        The beam-search function same as OpenNMT, with no length penalty.
    """

    def __init__(self, beam_size, tgt_vocab, cuda=True):

        self.beam_size = beam_size
        self.tgt_vocab = tgt_vocab
        self.done = False

        self.func = torch.cuda if cuda else torch

        # the score for each translation candidate on the beam.
        self.scores = self.func.FloatTensor(beam_size).zero_()

        # record all scores in all steps.
        self.all_scores = []

        # the back-pointers at each time-step (beam idxs)
        self.prev_ks = []

        # the outputs at each time-step (word idxs)
        self.next_ys = [self.func.LongTensor(beam_size).fill_(constants.BOS)]

        # translation candidates
        self.candidates = [[constants.BOS_WORD] for i in range(beam_size)]

    def advance(self, log_softmax_scores):
        """
        log_softmax_scores: beam_size * tgt_vocab_size
        """
        tgt_vocab_size = log_softmax_scores.size(-1)

        if len(self.prev_ks) > 0:
            beam_scores = log_softmax_scores + self.scores.unsqueeze(1).expand_as(log_softmax_scores)
        else:
            beam_scores = log_softmax_scores[0]  # the first time

        flat_beam_scores = beam_scores.view(-1)
        topk_scores, topk_idxs = torch.topk(flat_beam_scores, k=self.beam_size, dim=-1)

        self.all_scores.append(self.scores)
        self.scores = topk_scores

        prev_k = topk_idxs / tgt_vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append(topk_idxs - prev_k * tgt_vocab_size)

        # if the best candidate's output word is <EOS>, then beam search finished
        if self.next_ys[-1][0].data[0] == constants.EOS:
            self.done = True

        return self.done

    def update_status(self):

        prev_beam_idxs = self.prev_ks[-1].data.cpu().numpy()
        self.candidates = [self.candidates[idx] for idx in prev_beam_idxs]

        next_idxs = self.next_ys[-1].data.cpu().numpy()
        for i in range(len(next_idxs)):
            self.candidates[i] = self.candidates[i].copy() + [self.tgt_vocab.id2word[next_idxs[i]]]

    def get_best_candidate(self):

        for i in range(len(self.candidates[0])):
            if self.candidates[0][i] == constants.EOS_WORD:
                return " ".join(self.candidates[0][1:i])

        return " ".join(self.candidates[0][1:])

    def sort_scores(self):
        """ Sort the scores """

        return torch.sort(self.scores, dim=0, descending=True)

class Beam_Search_V2():

    """
        The beam search function is same as Google's Transformer, with length penalty.
    """
    def __init__(self, beam_size, tgt_vocab, length_alpha, cuda=True):

        """
        alpha: length penalty rate
        """
        self.beam_size = beam_size
        self.tgt_vocab = tgt_vocab
        self.length_alpha = length_alpha
        self.done = False

        self.func = torch.cuda if cuda else torch

        # the score for each translation candidate on the beam.
        self.scores = self.func.FloatTensor(beam_size).zero_()
        self.penalty_scores = self.func.FloatTensor(beam_size).zero_()

        # the back-pointers at each time-step (beam idxs)
        self.prev_ks = []

        # the outputs at each time-step (word idxs)
        self.next_ys = [self.func.LongTensor(2*beam_size).fill_(constants.BOS)]

        # beam-search candidates
        self.candidates = [[constants.BOS_WORD] for i in range(beam_size)]

        # translation candidates
        self.trans_candidates = []
        self.trans_candidates_scores = []
        self.best_score = float("-inf")

    def advance(self, log_softmax_scores):
        """
        log_softmax_scores: beam_size * tgt_vocab_size
        """
        tgt_vocab_size = log_softmax_scores.size(-1)

        if len(self.prev_ks) > 0:
            beam_scores = log_softmax_scores + self.scores.unsqueeze(1).expand_as(log_softmax_scores)
        else:
            beam_scores = log_softmax_scores[0]  # the first time

        flat_beam_scores = beam_scores.view(-1)

        # k = 2 * beam_size
        topk_scores, topk_idxs = torch.topk(flat_beam_scores, k=self.beam_size*2, dim=-1)

        self.scores = topk_scores

        # add length penalty
        penalty = length_penalty(len(self.candidates[0]), self.length_alpha)
        self.penalty_scores = topk_scores / penalty

        prev_k = topk_idxs / tgt_vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append(topk_idxs - prev_k * tgt_vocab_size)

        # method1: if the best candidate's output word is <EOS>, then beam search finished
        if self.next_ys[-1][0].data[0] == constants.EOS:
             self.done = True

        # method2: if the number of best candidates is beam-size, then finished
        '''
        if len(self.trans_candidates) >= self.beam_size:
            self.done = True
        '''

        return self.done

    def update_status(self):

        prev_beam_idxs = self.prev_ks[-1].data.cpu().numpy()                      # 2*beam_size
        candidates_2beam = [self.candidates[idx] for idx in prev_beam_idxs]       # 2*beam_size

        next_idxs = self.next_ys[-1].data.cpu().numpy()                           # 2*beam_size
        self.candidates = []                                                      # clear
        for i in range(len(next_idxs)):
            token = self.tgt_vocab.id2word[next_idxs[i]]
            if token == constants.EOS_WORD:
                if i < self.beam_size:
                    self.trans_candidates.append(candidates_2beam[i])
                    self.trans_candidates_scores.append(float(self.penalty_scores[i]))
            else:
                self.candidates.append(candidates_2beam[i] + [token])

        self.candidates = self.candidates[:self.beam_size]                        # beam size
        self.scores = self.scores[:self.beam_size]

    def get_best_candidate(self):

        if len(self.trans_candidates) == 0:
            return " ".join(self.candidates[0][1:])

        _, max_idx = torch.max(torch.FloatTensor(self.trans_candidates_scores), dim=-1)
        return " ".join(self.trans_candidates[int(max_idx)][1:])