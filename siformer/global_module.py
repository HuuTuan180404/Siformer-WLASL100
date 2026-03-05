import torch
import torch.nn as nn
from siformer.attention import AttentionLayer, ProbAttention


def prob_attention_factory(d_model, n_heads, dropout=0.):
    return AttentionLayer(ProbAttention(attention_dropout=dropout , output_attention = True), d_model, n_heads, mix = False)

def multi_head_attention_factory(d_model, n_heads, dropout=0.):
    return nn.MultiheadAttention(d_model, n_heads, dropout=dropout)


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn=None, act=None, dropout=0.):
        super().__init__()

        if attn is None: # self-attention
            self.attn = prob_attention_factory(d_model, n_heads, dropout)
        else: # shared attention
            self.attn = attn

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.ReLU() if act == 'relu' else nn.GELU(),
            nn.Conv1d(d_ff, d_model, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, L, D]

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = x.transpose(1, 2)  # [B, D, L]
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = x.transpose(1, 2)  # [B, L, D]

        return self.norm2(x)


class GlobalLayer(nn.Module):
    def __init__(self, d_model_list, n_heads, d_ff, dropout, act, attn=None):
        super().__init__()

        assert act in ['relu', 'gelu'], "act must be 'relu' or 'gelu'"

        self.lh_dim = d_model_list[0]
        self.rh_dim = d_model_list[1]
        self.body_dim = d_model_list[2]
        self.d_global = sum(d_model_list)

        self.layer = Encoder(self.d_global, n_heads, d_ff, attn, act, dropout)
        
    def forward(self, l_hand, r_hand, body):
        # src: [B, L, D]

        x = torch.cat((l_hand, r_hand, body), dim = -1) # [B, L, D_sum]
        x = self.layer(x)

        l_hand, r_hand, body = torch.split(
            x,
            [self.lh_dim, self.rh_dim, self.body_dim],
            dim=-1
        )

        return l_hand, r_hand, body
