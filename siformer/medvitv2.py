import copy
import math
import torch
import torch.nn as nn
from sympy.polys.polyconfig import query
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder

from typing import Optional, Union, Callable, List
from siformer.attention import AttentionLayer, ProbAttention, FullAttention, CrossAttention
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.encoder import  EncoderLayer, PBEEncoder
from siformer.utils import get_sequence_list
from utils import logger

import uuid


class LocalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size=12, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        w = self.window_size

        x = x.view(B, -1, w, D)          # [B, num_win, w, D]
        x = x.reshape(-1, w, D)          # [B*num_win, w, D]

        out, _ = self.attn(x, x, x)

        out = out.reshape(B, -1, w, D).reshape(B, -1, D)
        return out[:, :L]


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim=64, expand_ratio=4., d_ff=None, act='relu', dropout=0.1):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio) if d_ff is None else d_ff

        # Pointwise conv (expand)
        self.pw1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Depthwise conv (locality)
        self.dw = nn.Conv1d(hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Activation (ONLY here, MedViTV2-style)
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()

        # Pointwise conv (project back)
        self.pw2 = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)
        self.norm3 = nn.LayerNorm(in_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = x.transpose(1, 2)          # [B, D, L]

        # pw1
        x = self.pw1(x)
        x = x.transpose(1, 2)          # [B, L, hidden_dim]
        x = self.norm1(x)

        # dw
        x = x.transpose(1, 2)          # [B, hidden_dim, L]
        x = self.dw(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)

        # activation AFTER locality
        x = self.act(x)

        # project back
        x = x.transpose(1, 2)          # [B, hidden_dim, L]
        x = self.pw2(x)
        x = x.transpose(1, 2)          # [B, L, in_dim]
        x = self.norm3(x)

        x = self.dropout(x)

        return x + residual


class LocalLayer(nn.Module):
    def __init__(self, d_model_list, nhead_list, d_ff, dropout, act):
        super().__init__()

        self.lh_norm1 = nn.LayerNorm(d_model_list[0])
        self.rh_norm1 = nn.LayerNorm(d_model_list[1])
        self.body_norm1 = nn.LayerNorm(d_model_list[2])

        self.lh_attn = LocalSelfAttention(d_model_list[0], nhead_list[0],
                                          window_size=12, dropout=dropout)
        self.rh_attn = LocalSelfAttention(d_model_list[1], nhead_list[1],
                                          window_size=12, dropout=dropout)
        self.body_attn = LocalSelfAttention(d_model_list[2], nhead_list[2],
                                            window_size=12, dropout=dropout)

        self.lh_norm2 = nn.LayerNorm(d_model_list[0])
        self.rh_norm2 = nn.LayerNorm(d_model_list[1])
        self.body_norm2 = nn.LayerNorm(d_model_list[2])

        self.lh_lffn = LocalityFeedForward(in_dim=d_model_list[0], expand_ratio=4., d_ff=d_ff, act=act, dropout=dropout)
        self.rh_lffn = LocalityFeedForward(in_dim=d_model_list[1], expand_ratio=4., d_ff=d_ff, act=act, dropout=dropout)
        self.body_lffn = LocalityFeedForward(in_dim=d_model_list[2], expand_ratio=4., d_ff=d_ff, act=act, dropout=dropout)

    def forward(self, l_hand, r_hand, body):
        # src: [B, L, D]
        l_hand_out = l_hand + self.lh_attn(self.lh_norm1(l_hand))
        l_hand_out = l_hand_out + self.lh_lffn(self.lh_norm2(l_hand_out))

        r_hand_out = r_hand + self.rh_attn(self.rh_norm1(r_hand))
        r_hand_out = r_hand_out + self.rh_lffn(self.rh_norm2(r_hand_out))

        body_out = body + self.body_attn(self.body_norm1(body))
        body_out = body_out + self.body_lffn(self.body_norm2(body_out))
        return l_hand_out, r_hand_out, body_out


# class GlobalLayer(nn.Module):
#     def __init__(self, d_model_list, nhead_list, d_ff, dropout, act, self_attn_list: List):
#         super().__init__()

#         # attn
#         self.self_attn_lh = self_attn_list[0]
#         self.self_attn_rh = self_attn_list[1]
#         self.self_attn_body = self_attn_list[2]

#         # ffn
#         self.ffn_lh = nn.Sequential(nn.Linear(d_model_list[0], d_ff), 
#                                     nn.ReLU() if act == 'relu' else nn.GELU(), 
#                                     nn.Linear(d_ff, d_model_list[0]))
#         self.ffn_rh = nn.Sequential(nn.Linear(d_model_list[1], d_ff), 
#                                     nn.ReLU() if act == 'relu' else nn.GELU(), 
#                                     nn.Linear(d_ff, d_model_list[1]))
#         self.ffn_body = nn.Sequential(nn.Linear(d_model_list[2], d_ff), 
#                                       nn.ReLU() if act == 'relu' else nn.GELU(), 
#                                       nn.Linear(d_ff, d_model_list[2]))
        
#         # norm
#         self.norm1_lh = LayerNorm(d_model_list[0])
#         self.norm1_rh = LayerNorm(d_model_list[1])
#         self.norm1_body = LayerNorm(d_model_list[2])

#         self.norm2_lh = LayerNorm(d_model_list[0])
#         self.norm2_rh = LayerNorm(d_model_list[1])
#         self.norm2_body = LayerNorm(d_model_list[2])

#         # dropout
#         self.attn_dropout = nn.Dropout(dropout)
#         self.ffn_dropout = nn.Dropout(dropout)

#     def forward(self, l_hand_x, r_hand_x, body_x):
#         # l_hand_x, r_hand_x, body_x: [B, L, D]

#         # --- 1. Self-Attention ---
#         attn_lh_out, _ = self.self_attn_lh(self.norm1_lh(l_hand_x), 
#             self.norm1_lh(l_hand_x), self.norm1_lh(l_hand_x),
#         )
#         attn_rh_out, _ = self.self_attn_rh(self.norm1_rh(r_hand_x), 
#             self.norm1_rh(r_hand_x), self.norm1_rh(r_hand_x),
#         )
#         attn_body_out, _ = self.self_attn_body(self.norm1_body(body_x), 
#             self.norm1_body(body_x), self.norm1_body(body_x),
#         )

#         l_hand_x = l_hand_x + self.attn_dropout(attn_lh_out)
#         r_hand_x = r_hand_x + self.attn_dropout(attn_rh_out)
#         body_x = body_x + self.attn_dropout(attn_body_out)

#         # --- 3. Feed-Forward Network ---
#         l_hand_x = l_hand_x + self.ffn_dropout(self.ffn_lh(l_hand_x))
#         r_hand_x = r_hand_x + self.ffn_dropout(self.ffn_rh(r_hand_x))
#         body_x = body_x + self.ffn_dropout(self.ffn_body(body_x))

#         return l_hand_x, r_hand_x, body_x


class GlobalLayer(nn.Module):
    def __init__(self, d_model_list, nhead_list, d_ff, dropout, act, self_attn_list: List):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers
        self.self_attn_lh = self_attn_list[0]
        self.self_attn_rh = self_attn_list[1]
        self.self_attn_body = self_attn_list[2]
        self.norm1_lh = LayerNorm(d_model_list[0])
        self.norm1_rh = LayerNorm(d_model_list[1])
        self.norm1_body = LayerNorm(d_model_list[2])

        # rh truyền cho lh
        self.lh_from_rh_attn = nn.MultiheadAttention(d_model_list[0], nhead_list[0], kdim=d_model_list[1],
                                                   vdim=d_model_list[1], dropout=dropout, batch_first=False)
        self.rh_from_lh_attn = nn.MultiheadAttention(d_model_list[1], nhead_list[1], kdim=d_model_list[0],
                                                   vdim=d_model_list[0], dropout=dropout, batch_first=False)

        # Fusion layer chỉ nhận đầu ra từ một chú ý chéo
        self.lh_fusion_layer = nn.Linear(d_model_list[0], d_model_list[0])
        self.rh_fusion_layer = nn.Linear(d_model_list[1], d_model_list[1])
        self.norm2_lh = LayerNorm(d_model_list[0])
        self.norm2_rh = LayerNorm(d_model_list[1])

        # Giai đoạn 3: Feed-Forward Networks
        self.ffn_lh = nn.Sequential(nn.Linear(d_model_list[0], d_ff), 
                                    nn.ReLU() if act == 'relu' else nn.GELU(), 
                                    nn.Linear(d_ff, d_model_list[0]))
        self.ffn_rh = nn.Sequential(nn.Linear(d_model_list[1], d_ff), 
                                    nn.ReLU() if act == 'relu' else nn.GELU(), 
                                    nn.Linear(d_ff, d_model_list[1]))
        self.ffn_body = nn.Sequential(nn.Linear(d_model_list[2], d_ff), 
                                      nn.ReLU() if act == 'relu' else nn.GELU(), 
                                      nn.Linear(d_ff, d_model_list[2]))

        self.norm3_lh = LayerNorm(d_model_list[0])
        self.norm3_rh = LayerNorm(d_model_list[1])
        self.norm3_body = LayerNorm(d_model_list[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, l_hand_x, r_hand_x, body_x):
        # l_hand_x, r_hand_x, body_x: [B, L, D]

        # --- 1. Self-Attention ---
        lh_self, _ = self.self_attn_lh(l_hand_x, l_hand_x, l_hand_x)
        l_hand_x = self.norm1_lh(l_hand_x + self.dropout(lh_self))

        rh_self, _ = self.self_attn_rh(r_hand_x, r_hand_x, r_hand_x)
        r_hand_x = self.norm1_rh(r_hand_x + self.dropout(rh_self))

        body_self, _ = self.self_attn_body(body_x, body_x, body_x)
        body_x = self.norm1_body(body_x + self.dropout(body_self))

        # --- 2. Cross-Attention ---
        lh_from_rh, _ = self.lh_from_rh_attn(l_hand_x, r_hand_x, r_hand_x)
        lh_fused = self.lh_fusion_layer(lh_from_rh)
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout(lh_fused))

        rh_from_lh, _ = self.rh_from_lh_attn(query=r_hand_x,key= l_hand_x, value=l_hand_x)
        rh_fused = self.rh_fusion_layer(rh_from_lh)
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout(rh_fused))

        # --- 3. Feed-Forward Network ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout(self.ffn_body(body_x)))

        return [l_hand_x, r_hand_x, body_x]
    # def forward_prenorm(self, l_hand_x, r_hand_x, body_x):
    #     # l_hand_x, r_hand_x, body_x: [B, L, D]

    #     # --- 1. Self-Attention (Pre-Norm) ---
    #     lh_norm = self.norm1_lh(l_hand_x)
    #     lh_self, _ = self.self_attn_lh(lh_norm, lh_norm, lh_norm)
    #     l_hand_x = l_hand_x + self.dropout(lh_self)

    #     rh_norm = self.norm1_rh(r_hand_x)
    #     rh_self, _ = self.self_attn_rh(rh_norm, rh_norm, rh_norm)
    #     r_hand_x = r_hand_x + self.dropout(rh_self)

    #     body_norm = self.norm1_body(body_x)
    #     body_self, _ = self.self_attn_body(body_norm, body_norm, body_norm)
    #     body_x = body_x + self.dropout(body_self)

    #     # --- 2. Cross-Attention (Pre-Norm) ---
    #     lh_norm = self.norm2_lh(l_hand_x)
    #     rh_norm = self.norm2_rh(r_hand_x)

    #     lh_from_rh, _ = self.lh_from_rh_attn(
    #         query=lh_norm,
    #         key=rh_norm,
    #         value=rh_norm
    #     )
    #     lh_fused = self.lh_fusion_layer(lh_from_rh)
    #     l_hand_x = l_hand_x + self.dropout(lh_fused)

    #     rh_from_lh, _ = self.rh_from_lh_attn(
    #         query=rh_norm,
    #         key=lh_norm,
    #         value=lh_norm
    #     )
    #     rh_fused = self.rh_fusion_layer(rh_from_lh)
    #     r_hand_x = r_hand_x + self.dropout(rh_fused)

    #     # --- 3. Feed-Forward Network (Pre-Norm) ---
    #     l_hand_x = l_hand_x + self.dropout(self.ffn_lh(self.norm3_lh(l_hand_x)))
    #     r_hand_x = r_hand_x + self.dropout(self.ffn_rh(self.norm3_rh(r_hand_x)))
    #     body_x   = body_x   + self.dropout(self.ffn_body(self.norm3_body(body_x)))

    #     return [l_hand_x, r_hand_x, body_x]


class LGBlock(nn.Module):
    def __init__(self, d_model_list, nhead_list, d_ff, dropout, act, 
                 self_attn_list):
        super().__init__()
        self.local_layer = LocalLayer(
            d_model_list=d_model_list,
            nhead_list=nhead_list,
            d_ff=d_ff,
            dropout=dropout,
            act=act
        )
        self.global_layer = GlobalLayer(
            d_model_list=d_model_list,
            nhead_list=nhead_list,
            d_ff=d_ff,
            dropout=dropout,
            act=act,
            self_attn_list=self_attn_list
        )

    def forward(self, lh, rh, body):
        # lh, rh, body: (B, L, D)
        lh, rh, body = self.local_layer(lh, rh, body)
        lh, rh, body = self.global_layer(lh, rh, body)
        return lh, rh, body


class CombinedLayer(nn.Module):
    def __init__(self, d_model_list: List[int], nhead_list: List[int],
                 num_layers, d_ff, dropout, act):
        super().__init__()

        def attn_layer_factory(d_model, n_heads):
            return AttentionLayer(ProbAttention(output_attention = True), d_model, n_heads, mix = False)

        # define for global layer
        self.self_attn_lh = attn_layer_factory(d_model_list[0], nhead_list[0])
        self.self_attn_rh = attn_layer_factory(d_model_list[1], nhead_list[1])
        self.self_attn_body = attn_layer_factory(d_model_list[2], nhead_list[2])         

        self.layers = nn.ModuleList([
            LGBlock(
                d_model_list=d_model_list,
                nhead_list=nhead_list,
                d_ff=d_ff,
                dropout=dropout,
                act=act,
                self_attn_list=[
                    self.self_attn_lh,
                    self.self_attn_rh,
                    self.self_attn_body
                ]
            )
            for _ in range(num_layers)
        ])
        logger(f'LGBlock = {num_layers}')

    def forward(self, l_hand, r_hand, body):
        # l_hand, r_hand, body: [B, L, D]
        for layer in self.layers:
            l_hand, r_hand, body = layer(l_hand, r_hand, body)
        return l_hand, r_hand, body


class SLMedViTV2(nn.Module):
    def __init__(self, num_classes=100, num_hid=108, d_model_list=[42, 42, 24], 
                 nhead_list=[3, 3, 2, 9],
                  num_medvitv2_layers = 4, num_dec_layers = 3, pat_dec = 2,
                 seq_len = 204, device = None):
        super(SLMedViTV2, self).__init__()

        self.d_model = sum(d_model_list)
        self.num_decoder_layers = num_dec_layers
        self.pat_dec = pat_dec
        self.inner_classifiers_config = [num_hid, num_classes]
        self.d_ff = 2048

        # self.feature_extractor = FeatureExtractor(num_hid = 108, kernel_size = 7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model = 42))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model = 42))
        self.body_embedding   = nn.Parameter(self.get_encoding_table(d_model = 24))

        # self.encoder
        self.medvitv2 = CombinedLayer(d_model_list=d_model_list,
                                        num_layers=num_medvitv2_layers,
                                        nhead_list=nhead_list,
                                        d_ff=2048,
                                        dropout=0.1,
                                        act=nn.ReLU())

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))

        self.decoder = self.get_custom_decoder(nhead_list[-1])

        self.projection = nn.Linear(num_hid, num_classes)

    def encoder_block(self, l_hand, r_hand, body):
        # l_hand, r_hand, body: (B, L, D)
        lh_out, rh_out, body_out = self.medvitv2(l_hand, r_hand, body)
        return lh_out, rh_out, body_out

    def decoder_block(self, target, full_memory, training):
        decoder_out = self.decoder(target, full_memory, training)
        return decoder_out

    def forward(self, l_hand, r_hand, body):
        batch_size = l_hand.size(0)
        training = self.training

        # (B, L, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (B, L, D):  (24, 204, 108)
        new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), -1)
        new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), -1)
        new_body = body.view(body.size(0), body.size(1), -1)

        # (B, L, D) -> (L, B, D): (24, 204, 108) -> (204, 24, 108)
        new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype = torch.float32)
        new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype = torch.float32)
        new_body = new_body.permute(1, 0, 2).type(dtype = torch.float32)

        l_hand_in = new_l_hand + self.l_hand_embedding  # Shape remains the same
        r_hand_in = new_r_hand + self.r_hand_embedding
        body_in = new_body + self.body_embedding

        # (L, B, D) -> (B, L, D)
        l_hand_in = l_hand_in.permute(1, 0, 2)
        r_hand_in = r_hand_in.permute(1, 0, 2)
        body_in = body_in.permute(1, 0, 2)

        # encoder: medvitv2 (B, L, D)
        l_hand_out, r_hand_out, body_out = self.encoder_block(l_hand_in, r_hand_in, body_in)

        # 
        full_memory = torch.cat((l_hand_out, r_hand_out, body_out), dim = -1) # [B, L, D_sum]
        full_memory = full_memory.permute(1, 0, 2) # [L, B, D_sum]

        # decoder
        decoder_out = self.decoder_block(self.class_query.repeat(1, batch_size, 1), full_memory,
                                         training=training)
        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.projection(decoder_out).squeeze(0)
        return out

    def get_custom_decoder(self, nhead):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        self.inner_classifiers_config[0] = self.d_model
        return PBEEDecoder(decoder_layer, self.num_decoder_layers, norm = decoder_norm,
                           inner_classifiers_config = self.inner_classifiers_config, 
                           patient = self.pat_dec)

    @staticmethod
    def get_encoding_table(d_model = 108, seq_len = 204):
        torch.manual_seed(42)
        tensor_shape = (seq_len, d_model)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
        return frame_pos

