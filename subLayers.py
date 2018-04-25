import torch
import torch.nn as nn
import torch.nn.init as init
from module import Linear, ScaledDotProductAttention, LayerNormalization

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, attention_mechanism="vanilla_attention"):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.attention_mechanism = attention_mechanism

        if attention_mechanism == "self-attention":
            self.w_qkv = nn.Sequential(Linear(d_model, 3*d_model), nn.ReLU())

        if attention_mechanism == "vanilla-attention":
            self.w_q = nn.Sequential(Linear(d_model, d_model), nn.ReLU())
            self.w_kv = nn.Sequential(Linear(d_model, 2*d_model), nn.ReLU())

        self.attention = ScaledDotProductAttention(d_model, n_head)
        self.layer_norm = LayerNormalization(d_model)

        self.proj = Linear(n_head*d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        
        residual = q
        
        # Linear projections
        if self.attention_mechanism == "self-attention":
            qs, ks, vs = torch.split(self.w_qkv(q), split_size=q.size(-1), dim=-1)

        if self.attention_mechanism == "vanilla-attention":
            qs = self.w_q(q)
            ks, vs = torch.split(self.w_kv(k), split_size=k.size(-1), dim=-1)

        # Split and concat
        q_ = torch.cat(torch.chunk(qs, self.n_head, dim=-1), dim=0)  # (h*N, T_q, C/h)
        k_ = torch.cat(torch.chunk(ks, self.n_head, dim=-1), dim=0)  # (h*N, T_q, C/h)
        v_ = torch.cat(torch.chunk(vs, self.n_head, dim=-1), dim=0)  # (h*N, T_q, C/h)

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_, k_, v_, attn_mask=attn_mask.repeat(self.n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, qs.size(0), dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # nn.Linear is faster than nn.Conv1d
        ## need parameters initialization
        self.conv1 = nn.Sequential(Linear(d_hid, d_inner_hid), nn.ReLU())
        self.conv2 = Linear(d_inner_hid, d_hid)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.dropout(output)

        return self.layer_norm(output + residual)

