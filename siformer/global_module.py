import torch
import torch.nn as nn
from siformer.attention import FlashAttention


class GlobalLayer(nn.Module):
    def __init__(
        self, d_model_list, nhead_list, d_ff, dropout, act, global_attn_factory
    ):
        super().__init__()

        self.self_attn_lh = FlashAttention(d_model_list[0], nhead_list[0])
        self.self_attn_rh = FlashAttention(d_model_list[1], nhead_list[1])
        self.self_attn_bd = FlashAttention(d_model_list[2], nhead_list[2])

        self.norm1_lh = nn.LayerNorm(d_model_list[0])
        self.norm1_rh = nn.LayerNorm(d_model_list[1])
        self.norm1_bd = nn.LayerNorm(d_model_list[2])

        self.lh_from_rh_attn = nn.MultiheadAttention(
            d_model_list[0],
            nhead_list[0],
            kdim=d_model_list[1],
            vdim=d_model_list[1],
            dropout=dropout,
            batch_first=True,
        )
        self.rh_from_lh_attn = nn.MultiheadAttention(
            d_model_list[1],
            nhead_list[1],
            kdim=d_model_list[0],
            vdim=d_model_list[0],
            dropout=dropout,
            batch_first=True,
        )

        self.norm2_lh = nn.LayerNorm(d_model_list[0])
        self.norm2_rh = nn.LayerNorm(d_model_list[1])

        # Giai đoạn 3: Feed-Forward Networks
        self.ffn_lh = nn.Sequential(
            nn.Linear(d_model_list[0], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model_list[0]),
        )
        self.ffn_rh = nn.Sequential(
            nn.Linear(d_model_list[1], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model_list[1]),
        )
        self.ffn_bd = nn.Sequential(
            nn.Linear(d_model_list[2], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model_list[2]),
        )

        self.norm3_lh = nn.LayerNorm(d_model_list[0])
        self.norm3_rh = nn.LayerNorm(d_model_list[1])
        self.norm3_bd = nn.LayerNorm(d_model_list[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, lh_x, rh_x, bd_x):  # post-norm
        # l_hand_x, r_hand_x, bd_x: [B, L, D]

        # --- 1. Self-Attention ---
        lh_self = self.self_attn_lh(lh_x, lh_x, lh_x)
        lh_x = self.norm1_lh(lh_x + self.dropout(lh_self))

        rh_self = self.self_attn_rh(rh_x, rh_x, rh_x)
        rh_x = self.norm1_rh(rh_x + self.dropout(rh_self))

        bd_self = self.self_attn_bd(bd_x, bd_x, bd_x)
        bd_x = self.norm1_bd(bd_x + self.dropout(bd_self))

        # --- 2. Cross-Attention ---
        lh_from_rh, _ = self.lh_from_rh_attn(lh_x, rh_x, rh_x)
        rh_from_lh, _ = self.rh_from_lh_attn(rh_x, lh_x, lh_x)

        lh_x = self.norm2_lh(lh_x + lh_from_rh)
        rh_x = self.norm2_rh(rh_x + rh_from_lh)

        # --- 3. Feed-Forward Network ---
        lh_x = self.norm3_lh(lh_x + self.dropout(self.ffn_lh(lh_x)))
        rh_x = self.norm3_rh(rh_x + self.dropout(self.ffn_rh(rh_x)))
        bd_x = self.norm3_bd(bd_x + self.dropout(self.ffn_bd(bd_x)))

        return lh_x, rh_x, bd_x


if __name__ == "__main__":

    def multi_head_attention_factory(d_model, n_heads, dropout=0.0):
        return nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

    batch_size = 24
    seq_len = 204
    # hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = GlobalLayer(
        d_model_list=[42, 42, 24],
        nhead_list=[3, 3, 2],
        d_ff=2048,
        dropout=0.1,
        act="gelu",
        global_attn_factory=multi_head_attention_factory,
    )

    lh, rh, bd = model(lh, rh, bd)

    print(lh.shape)
    print(rh.shape)
    print(bd.shape)
