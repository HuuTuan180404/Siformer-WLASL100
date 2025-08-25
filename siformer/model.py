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

import uuid

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class AdaptiveTemporalPooling(nn.Module):
    """
    Adaptive pooling để capture temporal features ở các resolutions khác nhau
    """
    def __init__(self, in_channels, pool_sizes=[4, 8, 16, 32]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(size) for size in pool_sizes
        ])
        
        # Linear layers để project về cùng dimension
        self.projections = nn.ModuleList([
            nn.Linear(size * in_channels, in_channels) 
            for size in pool_sizes
        ])
        
        # Attention weights cho các scales
        self.scale_attention = nn.Sequential(
            nn.Linear(in_channels * len(pool_sizes), in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, len(pool_sizes)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        
        # Apply adaptive pooling ở các scales khác nhau
        pooled_features = []
        for pool, proj in zip(self.pools, self.projections):
            pooled = pool(x)  # [B, C, pool_size]
            pooled = pooled.reshape(B, -1)  # [B, C * pool_size]
            pooled = proj(pooled)  # [B, C]
            pooled_features.append(pooled)
        
        # Concatenate tất cả scales
        all_scales = torch.stack(pooled_features, dim=1)  # [B, num_scales, C]
        
        # Compute attention weights cho các scales
        scale_weights = self.scale_attention(all_scales.view(B, -1))  # [B, num_scales]
        scale_weights = scale_weights.unsqueeze(-1)  # [B, num_scales, 1]
        
        # Weighted combination
        output = torch.sum(all_scales * scale_weights, dim=1)  # [B, C]
        
        return output, scale_weights.squeeze(-1)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.dilations = dilations
        self.num_scales = len(dilations)

        channels_per_scale = channels // self.num_scales
        remainder_channels = channels % self.num_scales
        
        scale_channels = [channels_per_scale] * self.num_scales
        # Phân bổ phần dư cho các nhánh đầu tiên
        for i in range(remainder_channels):
            scale_channels[i] += 1

        self.multi_conv = nn.ModuleList([
            DepthwiseSeparableConv(channels, scale_channels[i], kernel_size, dilation=d, dropout=dropout) 
            for i, d in enumerate(dilations)])
        
        self.projection = nn.Conv1d(sum(scale_channels), channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(32, channels//4), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        multi_scale_outputs = []
        for conv in self.multi_conv:
            out = conv(x)
            multi_scale_outputs.append(out)
        
        # Concatenate outputs từ các scales
        out = torch.cat(multi_scale_outputs, dim=1)  # [B, C, L]
        
        # Project về original dimension
        out = self.projection(out)
        
        # Residual connection
        out = out + identity
        
        # Normalize
        # out = out.transpose(1, 2)  # [B, L, C]
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out  # [B, C, L]


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Conv1D (nhẹ hơn conv thường)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        # Depthwise
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )

        # Pointwise
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        num_groups = 1
        if out_channels % 7 == 0:
            num_groups = 7
        elif out_channels % 4 == 0: # Một lựa chọn khác
             num_groups = 4
        # Bạn có thể thêm các logic tìm ước số phức tạp hơn nếu cần
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        out = self.depthwise(x)
        out = self.pointwise(out)          # [B, out_channels, L]
        # out = out.transpose(1, 2)          # [B, L, C]
        # out = self.norm(out)
        # out = F.gelu(out)
        # out = self.dropout(out)
        return out    # [B, C, L]


class ModernTCN(nn.Module): # đầu vào: [B, D, L]
    def __init__(self, in_channels, hidden_dim=128, num_layers=6, 
                 kernel_size=3, base_dilation=1, dropout=0.1,
                 use_adaptive_pooling=True):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        
        # Multi-scale residual blocks
        self.residual_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # Progressive dilation expansion
            dilations = [
                base_dilation * (2 ** j) for j in range(3)  # [1, 2, 4] hoặc [2, 4, 8]
            ]
            
            block = ResidualTCNBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilations=dilations,
                dropout=dropout
            )
            self.residual_blocks.append(block)
            
            # Double base dilation mỗi 2 layers để expand receptive field
            if (i + 1) % 2 == 0:
                base_dilation *= 2
        
        # Adaptive temporal pooling (optional)
        self.use_adaptive_pooling = use_adaptive_pooling
        if use_adaptive_pooling:
            self.adaptive_pool = AdaptiveTemporalPooling(
                in_channels=hidden_dim,
                pool_sizes=[4, 8, 16, 32]
            )
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.output_norm = nn.GroupNorm(min(32, hidden_dim//4), hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        
        # Input projection
        out = self.input_proj(x)  # [B, hidden_dim, L]
        
        # Store intermediate outputs cho cross-scale attention
        scale_outputs = []
        
        # Apply multi-scale residual blocks
        for i, block in enumerate(self.residual_blocks):
            out = block(out)
            
            # Store outputs từ mỗi layer cho cross-attention
            if i % 2 == 1:  # Mỗi 2 layers
                scale_outputs.append(out.transpose(1, 2))  # [B, L, C]
        
        # Cross-scale attention giữa các layers
        if len(scale_outputs) > 1:
            # Concatenate outputs từ các scales
            stacked_outputs = torch.stack(scale_outputs, dim=1)  # [B, num_scales, L, C]
            B, S, L, C = stacked_outputs.shape
            
            # Reshape cho attention
            stacked_outputs = stacked_outputs.reshape(B * S, L, C)
            
            # Self-attention across scales
            attended, attention_weights = self.cross_scale_attention(
                stacked_outputs, stacked_outputs, stacked_outputs
            )
            
            # Reshape back và average
            attended = attended.reshape(B, S, L, C)
            out = attended.mean(dim=1).transpose(1, 2)  # [B, C, L]
        
        # Convert back: [B, C, L] -> [B, L, C]
        # out = out.transpose(1, 2)
        
        # Adaptive temporal pooling (optional)
        if self.use_adaptive_pooling:
            # adaptive_pool nhận vào [B, C, L] và trả ra [B, C]
            pooled_output, pool_weights = self.adaptive_pool(out)
            # Broadcast pooled features lại thành sequence length
            pooled_expanded = pooled_output.unsqueeze(-1).expand(-1, -1, out.size(2))
            
            # Combine với original output
            out = out + 0.1 * pooled_expanded # out vẫn là [B, C, L]
        
        # Final normalization và projection
        # out = out.transpose(1, 2)  # [B, C, L]
        out = self.output_norm(out).transpose(1, 2)
        out = self.output_proj(out)
        
        return out


class PerStreamPBE(nn.Module):
    """ PBEEncoder cho từng stream riêng. """
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float,
                 activation: nn.Module, enc_attn, # AttentionLayer(...)
                 patience: int = 1, inner_classifiers_config: List[int] = None,
                 projections_config: List[int] = None):
        super().__init__()
        
        encoder_layer = EncoderLayer(attention = enc_attn, d_model = d_model,
                                     d_ff = dim_feedforward, dropout = dropout,
                                     activation = "relu" if isinstance(activation, nn.ReLU) else "gelu")

        self.encoder = PBEEncoder(encoder_layer = encoder_layer, num_layers = num_layers,
                                  norm = nn.LayerNorm(d_model), patience = patience,
                                  inner_classifiers_config = inner_classifiers_config,
                                  projections_config = projections_config)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                training: bool = True) -> Tensor:
        # x: [L, B, D_stream]
        return self.encoder(x, mask = mask, src_key_padding_mask = key_padding_mask, training = training)


class CommunicatingEncoderLayer(nn.Module):
    """
    Một lớp Encoder tùy chỉnh thực hiện 3 giai đoạn:
    1. Self-Attention trong mỗi luồng.
    2. Cross-Attention đa hướng giữa các luồng.
    3. Feed-Forward Network.
    """

    def __init__(self, d_model_list, nhead_list, d_ff, dropout, activation, self_attn_list: List):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers
        self.self_attn_lh = self_attn_list[0]
        self.self_attn_rh = self_attn_list[1]
        self.self_attn_body = self_attn_list[2]
        self.norm1_lh = LayerNorm(d_model_list[0])
        self.norm1_rh = LayerNorm(d_model_list[1])
        self.norm1_body = LayerNorm(d_model_list[2])

        # Giai đoạn 2: Cross-Attention & Fusion Layers
        self.lh_to_rh_attn = CrossAttention(d_model = d_model_list[0], nhead = nhead_list[0], dropout = dropout)
        self.rh_to_lh_attn = CrossAttention(d_model = d_model_list[1], nhead = nhead_list[1], dropout = dropout)

        if d_model_list[0] !=  d_model_list[1]:
            self.lh_key_proj = nn.Linear(d_model_list[1], d_model_list[0])  # Project right->left dim
            self.rh_key_proj = nn.Linear(d_model_list[0], d_model_list[1])  # Project left->right dim
        else:
            self.lh_key_proj = nn.Identity()
            self.rh_key_proj = nn.Identity()

        # Fusion layer chỉ nhận đầu ra từ một chú ý chéo
        self.lh_fusion_layer = nn.Linear(d_model_list[0], d_model_list[0])
        self.rh_fusion_layer = nn.Linear(d_model_list[1], d_model_list[1])
        self.norm2_lh = LayerNorm(d_model_list[0])
        self.norm2_rh = LayerNorm(d_model_list[1])

        # Giai đoạn 3: Feed-Forward Networks
        self.ffn_lh = nn.Sequential(nn.Linear(d_model_list[0], d_ff), activation, nn.Linear(d_ff, d_model_list[0]))
        self.ffn_rh = nn.Sequential(nn.Linear(d_model_list[1], d_ff), activation, nn.Linear(d_ff, d_model_list[1]))
        self.ffn_body = nn.Sequential(nn.Linear(d_model_list[2], d_ff), activation, nn.Linear(d_ff, d_model_list[2]))

        self.norm3_lh = LayerNorm(d_model_list[0])
        self.norm3_rh = LayerNorm(d_model_list[1])
        self.norm3_body = LayerNorm(d_model_list[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_list, src_mask = None, src_key_padding_mask = None):
        l_hand_x, r_hand_x, body_x = src_list[0], src_list[1], src_list[2]

        # --- 1. Self-Attention ---
        lh_self, _ = self.self_attn_lh(l_hand_x, l_hand_x, l_hand_x, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        l_hand_x = self.norm1_lh(l_hand_x + self.dropout(lh_self))

        rh_self, _ = self.self_attn_rh(r_hand_x, r_hand_x, r_hand_x, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        r_hand_x = self.norm1_rh(r_hand_x + self.dropout(rh_self))

        body_self, _ = self.self_attn_body(body_x, body_x, body_x, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        body_x = self.norm1_body(body_x + self.dropout(body_self))

        # --- 2. Cross-Attention & Fusion ---
        # lh_from_body, _ = self.lh_to_body_attn(l_hand_x, body_x, body_x)
        rh_for_lh = self.lh_key_proj(r_hand_x)
        lh_from_rh, _ = self.lh_to_rh_attn(query = l_hand_x, key_value = rh_for_lh, mask = src_mask)
        lh_fused = self.lh_fusion_layer(lh_from_rh)
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout(lh_fused))

        lh_for_rh = self.rh_key_proj(l_hand_x)
        rh_from_lh, _ = self.rh_to_lh_attn(query = r_hand_x, key_value = lh_for_rh, mask = src_mask)
        rh_fused = self.rh_fusion_layer(rh_from_lh)
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout(rh_fused))

        # --- 3. Feed-Forward Network ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout(self.ffn_body(body_x)))

        return [l_hand_x, r_hand_x, body_x]


class CombinedEncoder(nn.Module):
    """
    1) PBEEncoder cho LH/RH/Body (song song, độc lập)
    2) Một stack CommunicatingEncoderLayer để cross-attend & fuse
    """
    def __init__(self, d_model_list: List[int], nhead_list: List[int],
                 num_pbe_layers: int, num_comm_layers: int,
                 dim_feedforward: int, dropout: float,
                 activation: nn.Module, attn_layer_factory, patience: int = 1,
                 inner_classifiers_config: List[int] = None,
                 projections_config: List[int] = None):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers
        self.self_attn_lh = attn_layer_factory(d_model_list[0], nhead_list[0])
        self.self_attn_rh = attn_layer_factory(d_model_list[1], nhead_list[1])
        self.self_attn_body = attn_layer_factory(d_model_list[2], nhead_list[2])

        inner_classifiers_config_lh=[42, inner_classifiers_config[1]]

        # 1) PBE per-stream
        self.pbe_lh = PerStreamPBE(d_model = d_model_list[0], nhead = nhead_list[0], 
                                   num_layers = num_pbe_layers,
                                   dim_feedforward = dim_feedforward, dropout = dropout, 
                                   activation = activation, enc_attn = self.self_attn_lh,
                                   patience = patience, inner_classifiers_config=[d_model_list[0], inner_classifiers_config[1]], 
                                   projections_config=projections_config)
        
        self.pbe_rh = PerStreamPBE(d_model = d_model_list[1], nhead = nhead_list[1], 
                                   num_layers = num_pbe_layers,
                                   dim_feedforward = dim_feedforward, dropout = dropout, 
                                   activation = activation, enc_attn = self.self_attn_rh,
                                   patience = patience, inner_classifiers_config=[d_model_list[1], inner_classifiers_config[1]], 
                                   projections_config=projections_config)
        
        self.pbe_body = PerStreamPBE(d_model = d_model_list[2], nhead = nhead_list[2], 
                                   num_layers = num_pbe_layers,
                                   dim_feedforward = dim_feedforward, dropout = dropout, 
                                   activation = activation, enc_attn = self.self_attn_body,
                                   patience = patience, inner_classifiers_config=[d_model_list[2], inner_classifiers_config[1]], 
                                   projections_config=projections_config)

        # 2) Communicating stack
        self.comm_layers = nn.ModuleList([
            CommunicatingEncoderLayer(d_model_list=d_model_list, nhead_list=nhead_list, 
                                      d_ff=dim_feedforward, dropout=dropout, activation=activation, 
                                      self_attn_list=[self.self_attn_lh, self.self_attn_rh, self.self_attn_body])
            for _ in range(num_comm_layers)
        ])

        print(f'num_comm_layers = {num_comm_layers}')

        # Norm cuối mỗi stream
        self.norm_lh = LayerNorm(d_model_list[0])
        self.norm_rh = LayerNorm(d_model_list[1])
        self.norm_body = LayerNorm(d_model_list[2])

    def forward(self, src_list: List[Tensor], src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, training: bool = True) -> List[Tensor]:
        l_hand_x, r_hand_x, body_x = src_list  # [L,B,D_i]

        # 1) PBE per-stream
        l_hand_x = self.pbe_lh(l_hand_x, mask = src_mask, key_padding_mask = src_key_padding_mask, training = training)
        r_hand_x = self.pbe_rh(r_hand_x, mask = src_mask, key_padding_mask = src_key_padding_mask, training = training)
        body_x   = self.pbe_body(body_x, mask = src_mask, key_padding_mask = src_key_padding_mask, training = training)

        # 2) Communicating stack
        feats = [l_hand_x, r_hand_x, body_x]
        for layer in self.comm_layers:
            feats = layer(feats, src_mask = src_mask, src_key_padding_mask = src_key_padding_mask)

        # Norm cuối
        feats[0] = self.norm_lh(feats[0])
        feats[1] = self.norm_rh(feats[1])
        feats[2] = self.norm_body(feats[2])
        return feats  # [LH, RH, Body] đã fused


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 num_pbe_layers: int, num_comm_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU(),
                 selected_attn: str = 'prob', output_attention: str = True,
                 inner_classifiers_config: list = None, patience: int = 1, use_pyramid_encoder: bool = False,
                 distil: bool = False, projections_config: list = None,
                 IA_encoder: bool = False, IA_decoder: bool = False, 
                 device = None):  # Dùng **kwargs cho các tham số không dùng đến

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder

        self.d_model = sum(d_model_list)
        self.d_ff =  dim_feedforward
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = device
        self.use_pyramid_encoder = use_pyramid_encoder
        self.use_IA_encoder = IA_encoder
        self.use_IA_decoder = IA_decoder
        self.inner_classifiers_config = inner_classifiers_config
        self.projections_config = projections_config
        self.patience = patience
        self.distil = distil
        self.activation = activation
        self.selected_attn = selected_attn
        self.output_attention = output_attention

        def attn_layer_factory(d_model, n_heads):
            Attn = ProbAttention if selected_attn == 'prob' else FullAttention
            return AttentionLayer(Attn(output_attention = output_attention), d_model, n_heads, mix = False)

        # Encoder kết hợp
        self.encoder = CombinedEncoder(
            d_model_list = d_model_list,
            nhead_list = nhead_list,
            num_pbe_layers = num_pbe_layers,
            num_comm_layers = num_comm_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            attn_layer_factory = attn_layer_factory,
            patience = patience,
            inner_classifiers_config = inner_classifiers_config,
            projections_config = projections_config
        )

        # --- Khởi tạo Decoder ---
        self.decoder = self.get_custom_decoder(nhead_list[-1])

    def get_custom_decoder(self, nhead):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        self.inner_classifiers_config[0] = 108
        return PBEEDecoder(decoder_layer, self.num_decoder_layers, norm = decoder_norm,
                           inner_classifiers_config = self.inner_classifiers_config, patient = self.patience)

    def forward(self, src: list, tgt: Tensor, 
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,                
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None, memory_is_causal: bool = False,
                training:bool = True, ) -> Tensor:
        
        lh, rh, body = self.encoder(src_list=src, src_mask = src_mask,
                                    src_key_padding_mask = src_key_padding_mask,
                                    training = training)

        # Nối lại để tạo bộ nhớ hoàn chỉnh cho decoder
        full_memory = torch.cat((lh, rh, body), dim = -1) # [L, B, D_sum]

        # Gọi Decoder
        output = self.decoder(tgt, full_memory, tgt_mask = tgt_mask,
                              memory_mask = memory_mask,
                              tgt_key_padding_mask = tgt_key_padding_mask,
                              memory_key_padding_mask = memory_key_padding_mask,
                              training=training)

        return output


class SiFormer(nn.Module):
    def __init__(self, num_classes, num_hid = 108, attn_type = 'prob',
                  num_pbe_layers = 3, num_comm_layers = 1, num_enc_layers = 3, 
                  num_dec_layers = 2, patience = 1,
                 seq_len = 204, device = None, IA_encoder = True, IA_decoder = False):
        super(SiFormer, self).__init__()
        print("Feature isolated transformer")

        # Branch TCN cho từng stream
        hidden_dim=128
        self.tcn_lh = ModernTCN(in_channels=42, num_layers=4)
        self.tcn_rh = ModernTCN(in_channels=42, num_layers=4)
        self.tcn_body = ModernTCN(in_channels=24, num_layers=4)
        self.tcn_reshape=nn.Linear(hidden_dim, 108)

        # self.feature_extractor = FeatureExtractor(num_hid = 108, kernel_size = 7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model = 42))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model = 42))
        self.body_embedding   = nn.Parameter(self.get_encoding_table(d_model = 24))

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.transformer = FeatureIsolatedTransformer(
            d_model_list = [42, 42, 24], nhead_list = [3, 3, 2, 9], 
            num_encoder_layers = num_enc_layers, num_decoder_layers = num_dec_layers,
            selected_attn = attn_type, 
            IA_encoder = IA_encoder, IA_decoder = IA_decoder,
            num_pbe_layers = num_pbe_layers, num_comm_layers = num_comm_layers,
            inner_classifiers_config = [num_hid, num_classes], 
            projections_config = [seq_len, 1],  device = device,
            patience = patience, use_pyramid_encoder = False, distil = False
        )

        self.fuse = nn.Linear(num_hid*2, num_hid)
        self.projection = nn.Linear(num_hid, num_classes)

    def forward(self, l_hand, r_hand, body, training):
        batch_size = l_hand.size(0)

        # (batch_size, seq_len, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (batch_size, seq_len, feature_size):  (24, 204, 108)
        new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), -1).type(dtype = torch.float32) # [B, L, 21, 2] -> reshape thành [B, L, 42]
        new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), -1).type(dtype = torch.float32)
        new_body = body.view(body.size(0), body.size(1), -1).type(dtype = torch.float32)

        # ----- Branch TCN -----
        l_hand_tcn = self.tcn_lh(new_l_hand)  
        r_hand_tcn = self.tcn_rh(new_r_hand)  
        body_tcn = self.tcn_body(new_body)    
        x_tcn = (l_hand_tcn + r_hand_tcn + body_tcn) / 3
        x_tcn = x_tcn.mean(dim=1)
        x_tcn = self.tcn_reshape(x_tcn)
        
        # (batch_size, seq_len, feature_size) : (24, 204, 108)
        # -> (seq_len, batch_size, feature_size): (204, 24, 108)
        new_l_hand = new_l_hand.permute(1, 0, 2)
        new_r_hand = new_r_hand.permute(1, 0, 2)
        new_body = new_body.permute(1, 0, 2)

        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        l_hand_in = new_l_hand + self.l_hand_embedding  # Shape remains the same
        r_hand_in = new_r_hand + self.r_hand_embedding
        body_in = new_body + self.body_embedding

        # (seq_len, batch_size, feature_size) -> (batch_size, 1, feature_size): (24, 1, 108)
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training = training
        ).transpose(0, 1)
        x_trans = transformer_output.squeeze(1)

        fusion = torch.cat([x_tcn, x_trans], dim=-1)  # [B,2H]

        fusion = self.fuse(fusion)

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.projection(fusion)
        return out

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


class AbsolutePE(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 1024, scale_factor = 1.0):
        super(AbsolutePE, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        # reshape the matrix shape to met the input shape
        # pe = pe.squeeze(0).unsqueeze(1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class SpoTer(nn.Module):
    def __init__(self, num_classes, num_hid = 108, num_enc_layers = 3, num_dec_layers = 2, seq_len = 204):
        super(SpoTer, self).__init__()
        print("Normal transformer")
        self.embedding = AbsolutePE(d_model = num_hid,max_len = seq_len)
        self.class_query = nn.Parameter(torch.rand(1, num_hid))
        self.transformer = nn.Transformer(num_hid, 9, num_enc_layers, num_dec_layers)
        custom_decoder_layer = DecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)
        self.projection = nn.Linear(num_hid, num_classes)
        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}")

    def forward(self, l_hand, r_hand, body, training):
        batch_size = l_hand.size(0)

        inputs = torch.cat((l_hand, r_hand, body), -2)
        inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))

        new_inputs = self.embedding(inputs)
        new_inputs = new_inputs.permute(1, 0, 2).type(dtype = torch.float32)

        transformer_out = self.transformer(new_inputs, self.class_query.repeat(1, batch_size, 1)).transpose(0, 1)
        out = self.projection(transformer_out).squeeze()
        return out
