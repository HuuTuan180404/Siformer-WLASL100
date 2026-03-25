import torch
import torch.nn as nn
from utils import logger
from siformer.local_module import LocalLayer
from siformer.global_module import GlobalLayer
from torch.nn.modules.normalization import LayerNorm
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.attention import AttentionLayer, ProbAttention
from siformer.utils_module import (
    prob_attention_factory,
    multi_head_attention_factory,
    window_attention_factory,
)


class LGBlock(nn.Module):
    def __init__(
        self,
        d_model_list,
        n_heads_list,
        d_ff,
        dropout,
        act,
        local_attn_type=None,
        global_attn_type=None,
        num_layers=0,
    ):
        super().__init__()

        assert num_layers > 0, "num_layers must be greater than 0"
        assert global_attn_type in [
            "self",
            "shared",
        ], "global_attn_type must be 'self' or 'shared'"

        # window attention
        local_window_attns = window_attention_factory
        if local_attn_type == "shared":
            lh_local_attn = window_attention_factory(
                d_model_list[0], n_heads_list[-1], window_size=12
            )
            rh_local_attn = window_attention_factory(
                d_model_list[1], n_heads_list[-1], window_size=12
            )
            bd_local_attn = window_attention_factory(
                d_model_list[2], n_heads_list[-1], window_size=12
            )
            local_window_attns = [lh_local_attn, rh_local_attn, bd_local_attn]

        global_attns = prob_attention_factory
        if global_attn_type == "shared":
            lh_glocal_attn = prob_attention_factory(d_model_list[0], n_heads_list[0])
            rh_glocal_attn = prob_attention_factory(d_model_list[1], n_heads_list[1])
            bd_glocal_attn = prob_attention_factory(d_model_list[2], n_heads_list[2])
            global_attns = [lh_glocal_attn, rh_glocal_attn, bd_glocal_attn]

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                LocalLayer(
                    d_model_list=d_model_list,
                    nhead_list=n_heads_list,
                    d_ff=d_ff,
                    dropout=dropout,
                    act=act,
                    local_attn_factory=local_window_attns,
                )
            )
            self.layers.append(
                GlobalLayer(
                    d_model_list=d_model_list,
                    nhead_list=n_heads_list,
                    d_ff=d_ff,
                    dropout=dropout,
                    act=act,
                    global_attn_factory=global_attns,
                )
            )

    def forward(self, lh, rh, body):
        # lh, rh, body: (B, L, D)
        for layer in self.layers:
            lh, rh, body = layer(lh, rh, body)
        return lh, rh, body


class MyModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        num_hid=108,
        d_model_list=[42, 42, 24],
        n_heads_list=[3, 3, 2, 10],
        num_enc_layers=4,
        num_dec_layers=3,
        pat_dec=2,
        seq_len=204,
        device=None,
    ):
        super(MyModel, self).__init__()
        logger(f"[Model]: Enc={num_enc_layers}, Dec={num_dec_layers}, Pat={pat_dec}")
        self.embed_dim_list = [128, 128, 64]
        self.embed_n_heads_list = [8, 8, 4]
        self.d_model = sum(self.embed_dim_list)
        self.num_decoder_layers = num_dec_layers
        self.pat_dec = pat_dec
        self.inner_classifiers_config = [num_hid, num_classes]
        self.d_ff = 2048

        self.lh_embedding = nn.Linear(d_model_list[0], self.embed_dim_list[0])
        self.rh_embedding = nn.Linear(d_model_list[1], self.embed_dim_list[1])
        self.bd_embedding = nn.Linear(d_model_list[2], self.embed_dim_list[2])

        # self.feature_extractor = FeatureExtractor(num_hid = 108, kernel_size = 7)
        self.lh_PE = nn.Parameter(
            self.get_encoding_table(d_model=self.embed_dim_list[0])
        )
        self.rh_PE = nn.Parameter(
            self.get_encoding_table(d_model=self.embed_dim_list[1])
        )
        self.bd_PE = nn.Parameter(
            self.get_encoding_table(d_model=self.embed_dim_list[2])
        )

        # self.encoder
        self.encoder = LGBlock(
            d_model_list=self.embed_dim_list,
            n_heads_list=self.embed_n_heads_list,
            d_ff=2048,
            dropout=0.1,
            act="gelu",
            local_attn_type="self",
            global_attn_type="shared",
            num_layers=num_enc_layers,
        )

        self.class_query = nn.Parameter(torch.rand(1, 1, self.d_model))

        self.decoder = self.get_custom_decoder(n_heads_list[-1])

        self.projection = nn.Linear(self.d_model, num_classes)

    def forward(self, lh, rh, bd):
        batch_size = lh.size(0)
        training = self.training

        # (B, L, J, C)
        new_lh = lh.view(lh.size(0), lh.size(1), -1).type(dtype=torch.float32)
        new_rh = rh.view(rh.size(0), rh.size(1), -1).type(dtype=torch.float32)
        new_bd = bd.view(bd.size(0), bd.size(1), -1).type(dtype=torch.float32)
        # -> (B, L, J*C)

        new_lh = self.lh_embedding(new_lh)  # (B, L, D)
        new_rh = self.rh_embedding(new_rh)
        new_bd = self.bd_embedding(new_bd)

        # (B, L, D) -> (L, B, D): (24, 204, 108) -> (204, 24, 108)
        new_lh = new_lh.permute(1, 0, 2)
        new_rh = new_rh.permute(1, 0, 2)
        new_bd = new_bd.permute(1, 0, 2)

        lh_in = new_lh + self.lh_PE  # Shape remains the same
        rh_in = new_rh + self.rh_PE
        bd_in = new_bd + self.bd_PE

        # (L, B, D) -> (B, L, D)
        lh_in = lh_in.permute(1, 0, 2)
        rh_in = rh_in.permute(1, 0, 2)
        bd_in = bd_in.permute(1, 0, 2)

        # encoder: medvitv2 (B, L, D)
        lh_out, rh_out, bd_out = self.encoder(lh_in, rh_in, bd_in)

        # full memory
        full_memory = torch.cat((lh_out, rh_out, bd_out), dim=-1)  # [B, L, D_sum]
        full_memory = full_memory.permute(1, 0, 2)  # [L, B, D_sum]

        # decoder
        decoder_out = self.decoder(
            self.class_query.repeat(1, batch_size, 1), full_memory, training=training
        )

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.projection(decoder_out).squeeze(0)
        return out

    def get_custom_decoder(self, n_heads):
        decoder_layer = DecoderLayer(self.d_model, n_heads, self.d_ff)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.inner_classifiers_config[0] = self.d_model
        return PBEEDecoder(
            decoder_layer,
            self.num_decoder_layers,
            norm=decoder_norm,
            inner_classifiers_config=self.inner_classifiers_config,
            patient=self.pat_dec,
        )

    @staticmethod
    def get_encoding_table(d_model=108, seq_len=204):
        torch.manual_seed(42)
        tensor_shape = (seq_len, d_model)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
        return frame_pos
