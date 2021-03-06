import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

__author__ = "Wu Kaixin"

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class LayerNormalization(nn.Module):

    def __init__(self, features, epsilon=1e-6):
        '''Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, input):

        '''
        if input.size(1) == 1:
            return input
        '''

        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)

        return self.gamma * (input - mean) / (std + self.epsilon) + self.beta

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, n_head, attention_dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model // n_head, 0.5)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn