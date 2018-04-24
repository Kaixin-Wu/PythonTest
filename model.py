import torch
import torch.nn as nn
import numpy as np
import constants
import torch.nn.init as init
from layers import EncoderLayer, DecoderLayer
from module import Linear

def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table. """

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    """ Indicate the padding-related part to mask """
    assert seq_q.dim() == 2 and seq_k.dim() == 2

    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk

    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    """ Get an attention mask to avoid using the subsequent info. """
    assert seq.dim() == 2

    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

class Encoder(nn.Module):
    """ A encoder model with self-attention mechanism. """

    def __init__(self, args, n_src_vocab):

        super(Encoder, self).__init__()

        n_position = args.max_seq_len + 1
        n_head = args.n_head
        d_k = args.d_k
        d_v = args.d_v
        d_inner_hid = args.d_inner_hid
        dropout = args.dropout
        n_layers = args.n_layers_enc

        self.n_max_seq = args.max_seq_len
        self.d_model = args.d_model
        self.d_word_vec = args.d_model

        if args.positional_encoding == "sinusoid":
            self.position_enc = nn.Embedding(n_position, self.d_word_vec, padding_idx=constants.PAD)
            self.position_enc.weight.data = position_encoding_init(n_position, self.d_word_vec)
            self.position_enc.weight.requires_grad = False

        if args.positional_encoding == "learned":
            self.position_enc = nn.Embedding(n_position, self.d_word_vec, padding_idx=constants.PAD)
            init.xavier_normal(self.position_enc.weight.data)

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_word_vec, padding_idx=constants.PAD)
        # self.src_word_emb.weight.data.normal_(-np.power(self.d_model, -0.5), np.power(self.d_model, -0.5))
        init.xavier_normal(self.src_word_emb.weight.data)
        self.src_word_emb.weight.data[constants.PAD].fill_(0)
        # limit = get_threshold(n_src_vocab, self.d_word_vec)
        # self.src_word_emb.weight.data.uniform_(-limit, limit)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, scale=True, return_attns=False):

        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)
        if scale:
            enc_input *= np.power(self.d_model, 0.5)

        # Position Encoding addition
        enc_pos_input = self.position_enc(src_pos)
        enc_input += enc_pos_input
        ## enc_input += self.position_enc(src_pos)

        # Add dropout yo the sums of the embeddings and the positional encodings
        enc_input = self.dropout(enc_input)

        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,

class Decoder(nn.Module):
    """ A decoder model with self-attention mechanism. """

    def __init__(self, args, n_tgt_vocab):

        super(Decoder, self).__init__()

        n_position = args.max_seq_len + 1
        n_head = args.n_head
        d_k = args.d_k
        d_v = args.d_v
        d_inner_hid = args.d_inner_hid
        dropout = args.dropout
        n_layers = args.n_layers_dec

        self.n_max_seq = args.max_seq_len
        self.d_model = args.d_model
        self.d_word_vec = args.d_model

        if args.positional_encoding == "sinusoid":
            self.position_enc = nn.Embedding(n_position, self.d_word_vec, padding_idx=constants.PAD)
            self.position_enc.weight.data = position_encoding_init(n_position, self.d_word_vec)
            self.position_enc.weight.requires_grad = False

        if args.positional_encoding == "learned":
            self.position_enc = nn.Embedding(n_position, self.d_word_vec, padding_idx=constants.PAD)
            init.xavier_normal(self.position_enc.weight.data)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, self.d_word_vec, padding_idx=constants.PAD)
        # self.tgt_word_emb.weight.data.normal_(-np.power(self.d_model, -0.5), np.power(self.d_model, -0.5))
        init.xavier_normal(self.tgt_word_emb.weight.data)
        self.tgt_word_emb.weight.data[constants.PAD].fill_(0)
        # limit = get_threshold(n_tgt_vocab, self.d_word_vec)
        # self.tgt_word_emb.weight.data.uniform_(-limit, limit)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(self.d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, scale=True, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)
        if scale:
            dec_input *= np.power(self.d_model, 0.5)

        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        # Add dropout yo the sums of the embeddings and the positional encodings
        dec_input = self.dropout(dec_input)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,

class Transformer(nn.Module):

    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, args, vocab):

        super(Transformer, self).__init__()
        self.args = args
        self.vocab = vocab

        self.encoder = Encoder(args, len(vocab.src))
        self.decoder = Decoder(args, len(vocab.tgt))

        self.tgt_word_proj = Linear(args.d_model, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """

        return (param for param in self.parameters() if param.requires_grad == True)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))

    def encode(self, src_seq, src_pos):

        enc_output, *_ = self.encoder(src_seq, src_pos)

        return enc_output

    def translate(self, enc_output, src_seq, tgt):
        """ translate """

        tgt_seq, tgt_pos = tgt

        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))

