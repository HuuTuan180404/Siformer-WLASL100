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
from siformer.attention import AttentionLayer, ProbAttention, FullAttention
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.encoder import  EncoderLayer, PBEEncoder
from siformer.utils import get_sequence_list

import uuid

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(channels, channels, kernel_size, dilation=dilation, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        return x + self.dropout(out)   # residual connection


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
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        out = self.depthwise(x)
        out = self.pointwise(out)          # [B, out_channels, L]
        out = out.transpose(1, 2)          # [B, L, C]
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out.transpose(1, 2)         # [B, C, L]


class ModernTCN(nn.Module):
    def __init__(self, in_channels, hidden_dim=108, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []

        # Project input lên hidden_dim
        layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=1))
        
        # Nhiều residual block
        for i in range(num_layers):
            dilation = 2 ** i   # tăng dần để mở rộng receptive field
            layers.append(ResidualTCNBlock(hidden_dim, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        out = self.network(x)     # [B, H, L]
        out = out.transpose(1, 2) # [B, L, H]
        return out


class PerStreamPBE(nn.Module):
    """ PBEEncoder cho từng stream riêng. """
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float,
                 activation: nn.Module,
                 attn_layer_factory,
                 patience: int = 1,
                 inner_classifiers_config: List[int] = None,
                 projections_config: List[int] = None):
        super().__init__()

        # attention cho EncoderLayer (ghi chú: EncoderLayer của bạn mong self.attention forward -> Tensor)
        enc_attn = attn_layer_factory(d_model, nhead)  # AttentionLayer(...)
        encoder_layer = EncoderLayer(
            attention=enc_attn,
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation="relu" if isinstance(activation, nn.ReLU) else "gelu"
        )
        
        self.encoder = PBEEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            patience=patience,
            inner_classifiers_config=inner_classifiers_config,
            projections_config=projections_config
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                training: bool = True) -> Tensor:
        # x: [L, B, D_stream]
        return self.encoder(x, mask=mask, src_key_padding_mask=key_padding_mask, training=training)


class CombinedEncoder(nn.Module):
    """
    1) PBEEncoder cho LH/RH/Body (song song, độc lập)
    2) Một stack CommunicatingEncoderLayer để cross-attend & fuse
    """
    def __init__(self, d_model_list: List[int], nhead_list: List[int],
                 num_pbe_layers: int, num_comm_layers: int,
                 dim_feedforward: int, dropout: float,
                 activation: nn.Module,
                 attn_layer_factory,
                 patience: int = 1,
                 inner_classifiers_config: List[int] = None,
                 projections_config: List[int] = None):
        super().__init__()
        # 1) PBE per-stream
        self.pbe_lh = PerStreamPBE(d_model_list[0], nhead_list[0], num_pbe_layers,
                                   dim_feedforward, dropout, activation, attn_layer_factory,
                                   patience, inner_classifiers_config, projections_config)
        self.pbe_rh = PerStreamPBE(d_model_list[1], nhead_list[1], num_pbe_layers,
                                   dim_feedforward, dropout, activation, attn_layer_factory,
                                   patience, inner_classifiers_config, projections_config)
        self.pbe_body = PerStreamPBE(d_model_list[2], nhead_list[2], num_pbe_layers,
                                     dim_feedforward, dropout, activation, attn_layer_factory,
                                     patience, inner_classifiers_config, projections_config)

        # 2) Communicating stack
        self.comm_layers = nn.ModuleList([
            CommunicatingEncoderLayer(d_model_list, nhead_list, dim_feedforward, dropout, activation, attn_layer_factory)
            for _ in range(num_comm_layers)
        ])

        # Norm cuối mỗi stream
        self.norm_lh = LayerNorm(d_model_list[0])
        self.norm_rh = LayerNorm(d_model_list[1])
        self.norm_body = LayerNorm(d_model_list[2])

    def forward(self, src_list: List[Tensor],
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                training: bool = True) -> List[Tensor]:
        l_hand_x, r_hand_x, body_x = src_list  # [L,B,D_i]

        # 1) PBE per-stream
        l_hand_x = self.pbe_lh(l_hand_x, mask=src_mask, key_padding_mask=src_key_padding_mask, training=training)
        r_hand_x = self.pbe_rh(r_hand_x, mask=src_mask, key_padding_mask=src_key_padding_mask, training=training)
        body_x   = self.pbe_body(body_x, mask=src_mask, key_padding_mask=src_key_padding_mask, training=training)

        # 2) Communicating stack
        feats = [l_hand_x, r_hand_x, body_x]
        for layer in self.comm_layers:
            feats = layer(feats, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Norm cuối
        feats[0] = self.norm_lh(feats[0])
        feats[1] = self.norm_rh(feats[1])
        feats[2] = self.norm_body(feats[2])
        return feats  # [LH, RH, Body] đã fused


class CommunicatingEncoderLayer(nn.Module):
    """
    Một lớp Encoder tùy chỉnh thực hiện 3 giai đoạn:
    1. Self-Attention trong mỗi luồng.
    2. Cross-Attention đa hướng giữa các luồng (bây giờ full: tay trái ↔ tay phải, tay trái ↔ body, tay phải ↔ body).
    3. Feed-Forward Network.
    """

    def __init__(self, d_model_list, nhead_list, d_ff, dropout, activation, attn_layer_factory):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers (giữ nguyên)
        self.self_attn_lh = attn_layer_factory(d_model_list[0], nhead_list[0])
        self.self_attn_rh = attn_layer_factory(d_model_list[1], nhead_list[1])
        self.self_attn_body = attn_layer_factory(d_model_list[2], nhead_list[2])
        self.norm1_lh = nn.LayerNorm(d_model_list[0])
        self.norm1_rh = nn.LayerNorm(d_model_list[1])
        self.norm1_body = nn.LayerNorm(d_model_list[2])

        # Giai đoạn 2: Cross-Attention & Fusion Layers (mở rộng full)
        # Cross-Attention giữa tay trái và tay phải (giữ nguyên)
        self.lh_to_rh_attn = nn.MultiheadAttention(d_model_list[0], nhead_list[0], 
                                                   kdim=d_model_list[1], vdim=d_model_list[1], 
                                                   dropout=dropout, batch_first=False)
        self.rh_to_lh_attn = nn.MultiheadAttention(d_model_list[1], nhead_list[1], 
                                                   kdim=d_model_list[0], vdim=d_model_list[0], 
                                                   dropout=dropout, batch_first=False)
        
        # Thêm Cross-Attention giữa tay và body (hai chiều)
        self.lh_to_body_attn = nn.MultiheadAttention(d_model_list[0], nhead_list[0], 
                                                     kdim=d_model_list[2], vdim=d_model_list[2], 
                                                     dropout=dropout, batch_first=False)
        self.rh_to_body_attn = nn.MultiheadAttention(d_model_list[1], nhead_list[1], 
                                                     kdim=d_model_list[2], vdim=d_model_list[2], 
                                                     dropout=dropout, batch_first=False)
        self.body_to_lh_attn = nn.MultiheadAttention(d_model_list[2], nhead_list[2], 
                                                     kdim=d_model_list[0], vdim=d_model_list[0], 
                                                     dropout=dropout, batch_first=False)
        self.body_to_rh_attn = nn.MultiheadAttention(d_model_list[2], nhead_list[2], 
                                                     kdim=d_model_list[1], vdim=d_model_list[1], 
                                                     dropout=dropout, batch_first=False)
        
        # Fusion layers (mở rộng để xử lý nhiều nguồn cross-attn)
        self.lh_fusion_from_rh = nn.Linear(d_model_list[0], d_model_list[0])
        self.lh_fusion_from_body = nn.Linear(d_model_list[0], d_model_list[0])
        self.rh_fusion_from_lh = nn.Linear(d_model_list[1], d_model_list[1])
        self.rh_fusion_from_body = nn.Linear(d_model_list[1], d_model_list[1])
        self.body_fusion_from_lh = nn.Linear(d_model_list[2], d_model_list[2])
        self.body_fusion_from_rh = nn.Linear(d_model_list[2], d_model_list[2])
                
        self.norm2_lh = LayerNorm(d_model_list[0])
        self.norm2_rh = LayerNorm(d_model_list[1])
        self.norm2_body = nn.LayerNorm(d_model_list[2])

        # Giai đoạn 3: Feed-Forward Networks
        self.ffn_lh = nn.Sequential(nn.Linear(d_model_list[0], d_ff), activation, nn.Linear(d_ff, d_model_list[0]))
        self.ffn_rh = nn.Sequential(nn.Linear(d_model_list[1], d_ff), activation, nn.Linear(d_ff, d_model_list[1]))
        self.ffn_body = nn.Sequential(nn.Linear(d_model_list[2], d_ff), activation, nn.Linear(d_ff, d_model_list[2]))

        self.norm3_lh = LayerNorm(d_model_list[0])
        self.norm3_rh = LayerNorm(d_model_list[1])
        self.norm3_body = LayerNorm(d_model_list[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_list, src_mask=None, src_key_padding_mask=None):
        l_hand_x, r_hand_x, body_x = src_list[0], src_list[1], src_list[2]

        # --- 1. Self-Attention ---
        lh_self, _ = self.self_attn_lh(l_hand_x, l_hand_x, l_hand_x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        l_hand_x = self.norm1_lh(l_hand_x + self.dropout(lh_self))

        rh_self, _ = self.self_attn_rh(r_hand_x, r_hand_x, r_hand_x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        r_hand_x = self.norm1_rh(r_hand_x + self.dropout(rh_self))

        body_self, _ = self.self_attn_body(body_x, body_x, body_x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        body_x = self.norm1_body(body_x + self.dropout(body_self))

        # --- 2. Cross-Attention & Fusion ---
        # Cross-Attention
        lh_from_rh, _ = self.lh_to_rh_attn(query=l_hand_x, key=r_hand_x, value=r_hand_x)
        lh_from_body, _ = self.lh_to_body_attn(query=l_hand_x, key=body_x, value=body_x)

        rh_from_lh, _ = self.rh_to_lh_attn(query=r_hand_x, key= l_hand_x, value=l_hand_x)
        rh_from_body, _ = self.rh_to_body_attn(query=r_hand_x, key= body_x, value=body_x)

        body_from_lh, _ = self.body_to_lh_attn(query=body_x, key= l_hand_x, value=l_hand_x)
        body_from_rh, _ = self.body_to_rh_attn(query=body_x, key= r_hand_x, value=r_hand_x)

        # Fusion
        lh_fused_rh = self.lh_fusion_from_rh(lh_from_rh)
        lh_fused_body = self.lh_fusion_from_body(lh_from_body)
        lh_fused = lh_fused_rh + lh_fused_body  # Cộng để kết hợp (có thể thay bằng concat + linear nếu cần)
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout(lh_fused))

        rh_fused_lh = self.rh_fusion_from_lh(rh_from_lh)
        rh_fused_body = self.rh_fusion_from_body(rh_from_body)
        rh_fused = rh_fused_lh + rh_fused_body  # Cộng để kết hợp
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout(rh_fused))
        
        body_fused_lh = self.body_fusion_from_lh(body_from_lh)
        body_fused_rh = self.body_fusion_from_rh(body_from_rh)
        body_fused = body_fused_lh + body_fused_rh  # Cộng để kết hợp
        body_x = self.norm2_body(body_x + self.dropout(body_fused))

        # --- 3. Feed-Forward Network ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout(self.ffn_body(body_x)))

        return [l_hand_x, r_hand_x, body_x]


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 num_pbe_layers: int, num_comm_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU(),
                 selected_attn: str = 'prob', output_attention: str = True,
                 IA_decoder: bool = False, inner_classifiers_config: list = None, patience: int = 1,projections_config: list = None,
                 **kwargs):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder

        self.d_model = sum(d_model_list)
        self.d_ff= dim_feedforward
        self.num_decoder_layers = num_decoder_layers
        self.use_IA_decoder = IA_decoder
        self.inner_classifiers_config = inner_classifiers_config
        self.patience = patience

        # --- Khởi tạo Encoder ---
        # Hàm factory để tạo các lớp AttentionLayer một cách nhất quán
        def attn_layer_factory(d_model, n_heads):
            Attn = ProbAttention if selected_attn == 'prob' else FullAttention
            return AttentionLayer(Attn(output_attention=output_attention), d_model, n_heads, mix=False)

        # Encoder kết hợp
        self.encoder = CombinedEncoder(
            d_model_list=d_model_list,
            nhead_list=nhead_list,
            num_pbe_layers=num_pbe_layers,
            num_comm_layers=num_comm_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_layer_factory=attn_layer_factory,
            patience=patience,
            inner_classifiers_config=inner_classifiers_config,
            projections_config=projections_config
        )

        # Lớp Norm cuối cùng cho mỗi luồng, sẽ được áp dụng sau khi qua tất cả các lớp
        # self.norm_lh = LayerNorm(d_model_list[0])
        # self.norm_rh = LayerNorm(d_model_list[1])
        # self.norm_body = LayerNorm(d_model_list[2])

        # --- Khởi tạo Decoder ---
        self.decoder = self.get_custom_decoder(nhead_list[-1])
        # self._reset_parameters()

    def get_custom_decoder(self, nhead):
        # Hàm này không có gì thay đổi
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        if self.use_IA_decoder:
            return PBEEDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                               inner_classifiers_config=self.inner_classifiers_config, patient=self.patience)
        else:
            print("TransformerDecoder")
            return TransformerDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, training:bool=True, 
                **kwargs) -> Tensor:
        
        lh, rh, body = self.encoder(src, src_mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask,
                                    training=kwargs.get('training', True))

        # Nối lại để tạo bộ nhớ hoàn chỉnh cho decoder
        full_memory = torch.cat((lh, rh, body), dim=-1) # [L, B, D_sum]

        # Gọi Decoder
        # Truyền các kwargs vào decoder một cách linh hoạt
        output = self.decoder(tgt, full_memory,
                              tgt_mask=kwargs.get('tgt_mask'),
                              memory_mask=kwargs.get('memory_mask'),
                              tgt_key_padding_mask=kwargs.get('tgt_key_padding_mask'),
                              memory_key_padding_mask=kwargs.get('memory_key_padding_mask'))

        return output # [1, B, 108]


class SiFormer(nn.Module):
    def __init__(self, num_classes, num_hid=108, attn_type='prob',
                  num_pbe_layers=2, num_comm_layers=1, num_enc_layers=3, 
                  num_dec_layers=2, patience=1,
                 seq_len=204, device=None, IA_encoder = True, IA_decoder = False):
        super(SiFormer, self).__init__()
        print("Feature isolated transformer")

        # Branch TCN cho từng stream
        self.tcn_lh = ModernTCN(in_channels=42, hidden_dim=num_hid, num_layers=4)
        self.tcn_rh = ModernTCN(in_channels=42, hidden_dim=num_hid, num_layers=4)
        self.tcn_body = ModernTCN(in_channels=24, hidden_dim=num_hid, num_layers=4)

        # Branch Transformer
        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.body_embedding   = nn.Parameter(self.get_encoding_table(d_model=24))

        self.projection = nn.Linear(num_hid, num_classes)

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))

        self.transformer = FeatureIsolatedTransformer(
            [42, 42, 24], [3, 3, 2, 9], 
            num_encoder_layers=num_enc_layers, 
            num_decoder_layers=num_dec_layers,
            selected_attn=attn_type, 
            IA_encoder = IA_encoder, IA_decoder = IA_decoder,
            num_pbe_layers = num_pbe_layers, num_comm_layers = num_comm_layers,
            inner_classifiers_config=[num_hid, num_classes], 
            projections_config=[seq_len, 1],  
            device=device,
            patience=patience, 
            use_pyramid_encoder=False, 
            istil=False
        )

        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patience {patience}")

        # Fusion
        self.fuse = nn.Linear(num_hid*2, num_hid)
        self.fc_out = nn.Linear(num_hid, num_classes)

    def forward(self, l_hand, r_hand, body, training):
         # tương đường với l_hand.shape[0  ] | số lượng record đầu vào
        '''
            # Giả sử l_hand có shape như này:
            l_hand = torch.randn(2, 204, 21, 2)
            print(l_hand.shape)  # torch.Size([2, 204, 21, 2])

            # Các dimensions:
            # Dimension 0: batch_size = 2
            # Dimension 1: seq_len = 204  
            # Dimension 2: keypoints = 21
            # Dimension 3: coordinates = 2 (x, y)
            print(l_hand.size())    # torch.Size([2, 204, 21, 2]) - tất cả dimensions
            print(l_hand.size(0))   # 2 - chỉ dimension 0 (batch_size)
            print(l_hand.size(1))   # 204 - chỉ dimension 1 (seq_len)
            print(l_hand.size(2))   # 21 - chỉ dimension 2 (keypoints)
            print(l_hand.size(3))   # 2 - chỉ dimension 3 (coordinates)

            # Tương đương với:
            print(l_hand.shape[0])  # 2
            print(l_hand.shape[1])  # 204
        '''
        # (batch_size, seq_len, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (batch_size, seq_len, feature_size):  (24, 204, 108)
        batch_size = l_hand.size(0)

        new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3)).type(dtype=torch.float32) # [B, L, 21, 2] -> reshape thành [B, L, 42]
        new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3)).type(dtype=torch.float32)
        new_body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3)).type(dtype=torch.float32)

        # ----- Branch TCN -----
        l_hand_tcn = self.tcn_lh(new_l_hand)  
        r_hand_tcn = self.tcn_rh(new_r_hand)  
        body_tcn = self.tcn_body(new_body)    
        x_tcn = (l_hand_tcn + r_hand_tcn + body_tcn) / 3 # [B,L,D]
        x_tcn = x_tcn.mean(dim=1)
        
        # ----- Branch Transformer -----
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
            [l_hand_in, r_hand_in, body_in], 
            self.class_query.repeat(1, batch_size, 1), 
            training=training
        ).transpose(0, 1)

        x_trans = transformer_output.squeeze(1)

        fusion = torch.cat([x_tcn, x_trans], dim=-1)  # [B,2H]
        fusion = self.fuse(fusion)                         # [B,H]
        out = self.fc_out(fusion)         

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        # out = self.projection(transformer_output).squeeze(1)
        return out

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


class AbsolutePE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
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
    def __init__(self, num_classes, num_hid=108, num_enc_layers=3, num_dec_layers=2, seq_len=204):
        super(SpoTer, self).__init__()
        print("Normal transformer")
        self.embedding = AbsolutePE(d_model=num_hid,max_len=seq_len)
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
        new_inputs = new_inputs.permute(1, 0, 2).type(dtype=torch.float32)

        transformer_out = self.transformer(new_inputs, self.class_query.repeat(1, batch_size, 1)).transpose(0, 1)
        out = self.projection(transformer_out).squeeze()
        return out

