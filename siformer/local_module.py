import torch
import torch.nn as nn
import torch.nn.functional as F
from siformer.utils_module import *


class Stream1(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn=None, act=None, dropout=0.):
        super().__init__()

        self.attn = prob_attention_factory(d_model, n_heads, dropout) if attn is None else attn

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if act == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x: [B, L, D]

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return self.norm2(x)

class Stream(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn=None, act=None, dropout=0.):
        super().__init__()

        self.attn = prob_attention_factory(d_model, n_heads, dropout) if attn is None else attn

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if act == "relu" else F.gelu

    def forward(self, x):
        # x: [B, L, D]

        attn_out, _ = self.attn(x, x, x) # (B, L, D)
        y = x = self.norm1(x + self.dropout(attn_out))

        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))

        return self.norm2(x+y)

class LocalLayer(nn.Module):
    def __init__(self, d_model_list, n_heads_list, d_ff, attn_list=None, act=None, dropout=0.):
        super().__init__()

        if attn_list is None: # self-attention
            self.lh = Stream(d_model_list[0], n_heads_list[0], d_ff, None, act, dropout)
            self.rh = Stream(d_model_list[1], n_heads_list[1], d_ff, None, act, dropout)
            self.body = Stream(d_model_list[2], n_heads_list[2], d_ff, None, act, dropout)
        else: # shared attention
            self.lh = Stream(d_model_list[0], n_heads_list[0], d_ff, attn_list[0], act, dropout)
            self.rh = Stream(d_model_list[1], n_heads_list[1], d_ff, attn_list[1], act, dropout)
            self.body = Stream(d_model_list[2], n_heads_list[2], d_ff, attn_list[2], act, dropout)

    def forward(self, lh, rh, body):
        # src: [B, L, D]

        lh = self.lh(lh)
        rh = self.rh(rh)
        body = self.body(body)

        return lh, rh, body


if __name__ == '__main__':
    batch_size = 24
    seq_len = 204
    # input_dim = 
    hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = LocalLayer(
        d_model_list=[42, 42, 24],
        hidden_dim=2048
    )

    y, _, _ = model(lh, rh, bd)

    print(y.shape)