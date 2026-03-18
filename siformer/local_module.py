import torch
import torch.nn as nn
import torch.nn.functional as F
from siformer.attention import WindowAttention



class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim=64, expand_ratio=4.0, d_ff=None, act="relu", dropout=0.1):
        super().__init__()
        hidden_dim = d_ff

        # Pointwise conv (expand)
        self.pw1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Depthwise conv (locality)
        self.dw = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Activation (ONLY here, MedViTV2-style)
        self.act = nn.ReLU() if act == "relu" else nn.GELU()

        # Pointwise conv (project back)
        self.pw2 = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)
        self.norm3 = nn.LayerNorm(in_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = x.transpose(1, 2)  # [B, D, L]

        # pw1
        x = self.pw1(x)
        x = x.transpose(1, 2)  # [B, L, hidden_dim]
        x = self.norm1(x)

        # dw
        x = x.transpose(1, 2)  # [B, hidden_dim, L]
        x = self.dw(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)

        # activation AFTER locality
        x = self.act(x)

        # project back
        x = x.transpose(1, 2)  # [B, hidden_dim, L]
        x = self.pw2(x)
        x = x.transpose(1, 2)  # [B, L, in_dim]
        x = self.norm3(x)

        x = self.dropout(x)

        return x + residual


class LocalLayer(nn.Module):
    def __init__(
        self, d_model_list, nhead_list, d_ff, dropout, act, local_attn_factory=None
    ):
        super().__init__()

        # if callable(local_attn_factory):
        #     self.lh_attn = local_attn_factory(
        #         d_model_list[0], nhead_list[0], window_size=12, dropout=dropout
        #     )
        #     self.rh_attn = local_attn_factory(
        #         d_model_list[1], nhead_list[1], window_size=12, dropout=dropout
        #     )
        #     self.bd_attn = local_attn_factory(
        #         d_model_list[2], nhead_list[2], window_size=12, dropout=dropout
        #     )


        # self.lh_attn, self.rh_attn, self.bd_attn = local_attn_factory

        self.lh_attn = WindowAttention(d_model_list[0], nhead_list[0], window_size=12)
        self.rh_attn = WindowAttention(d_model_list[1], nhead_list[1], window_size=12)
        self.bd_attn = WindowAttention(d_model_list[2], nhead_list[2], window_size=12)

        # self.lh_attn = F.scaled_dot_product_attention()

        self.lh_norm1 = nn.LayerNorm(d_model_list[0])
        self.rh_norm1 = nn.LayerNorm(d_model_list[1])
        self.bd_norm1 = nn.LayerNorm(d_model_list[2])

        self.lh_norm2 = nn.LayerNorm(d_model_list[0])
        self.rh_norm2 = nn.LayerNorm(d_model_list[1])
        self.bd_norm2 = nn.LayerNorm(d_model_list[2])

        self.lh_lffn = LocalityFeedForward(
            in_dim=d_model_list[0],
            expand_ratio=4.0,
            d_ff=d_ff,
            act=act,
            dropout=dropout,
        )
        self.rh_lffn = LocalityFeedForward(
            in_dim=d_model_list[1],
            expand_ratio=4.0,
            d_ff=d_ff,
            act=act,
            dropout=dropout,
        )
        self.bd_lffn = LocalityFeedForward(
            in_dim=d_model_list[2],
            expand_ratio=4.0,
            d_ff=d_ff,
            act=act,
            dropout=dropout,
        )

    def forward(self, l_hand, r_hand, bd):  # post-norm
        # src: [B, L, D]

        # Left hand
        l_hand_out = self.lh_norm1(l_hand + self.lh_attn(l_hand))
        l_hand_out = self.lh_norm2(l_hand_out + self.lh_lffn(l_hand_out))

        # Right hand
        r_hand_out = self.rh_norm1(r_hand + self.rh_attn(r_hand))
        r_hand_out = self.rh_norm2(r_hand_out + self.rh_lffn(r_hand_out))

        # bd
        bd_out = self.bd_norm1(bd + self.bd_attn(bd))
        bd_out = self.bd_norm2(bd_out + self.bd_lffn(bd_out))

        return l_hand_out, r_hand_out, bd_out


if __name__ == "__main__":
    batch_size = 24
    seq_len = 204
    # input_dim =
    hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = LocalLayer(d_model_list=[42, 42, 24], hidden_dim=2048)

    y, _, _ = model(lh, rh, bd)

    print(y.shape)
