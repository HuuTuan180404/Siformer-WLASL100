import torch.nn as nn
from siformer.attention import AttentionLayer, ProbAttention, WindowAttention


def prob_attention_factory(d_model, n_heads, dropout=0.):
    return AttentionLayer(ProbAttention(attention_dropout=dropout , output_attention = True), d_model, n_heads, mix = False)


def multi_head_attention_factory(d_model, n_heads, dropout=0.):
    return nn.MultiheadAttention(d_model, n_heads, dropout=dropout)


def window_attention_factory(d_model, n_heads, window_size, dropout=0.):
    return WindowAttention(d_model, n_heads, window_size, dropout)