import copy
import math
import torch
import torch.nn as nn
from sympy.polys.polyconfig import query
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder

from typing import Optional, Union, Callable

from torch_geometric.nn import GCNConv

# from torch_geometric.graphgym import GCNConv

from siformer.attention import AttentionLayer, ProbAttention, FullAttention
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, PBEEncoder
from siformer.utils import get_sequence_list

import uuid

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])

class SpatialGCNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dims= [16, 32, 64], output_dim=32, dropout=0.1, use_batch_norm=True):
        super(SpatialGCNEncoder,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        dims = [input_dim] + hidden_dims + [output_dim] # [2, 16, 32, 64, 32]

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            # Sử dụng GCNConv với improved aggregation
            self.gcn_layers.append(
                pyg_nn.GCNConv(dims[i], dims[i + 1], improved=True, cached=False, add_self_loops=True)
            )

            if use_batch_norm and i < len(dims) - 2:  # Không batch norm ở layer cuối
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            encoded_features: [num_nodes, output_dim]
        """
        identity = x

        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)

            # Apply batch norm (except last layer)
            if self.use_batch_norm and i < len(self.gcn_layers) - 1:
                x = self.batch_norms[i](x.transpose(1, 2)).transpose(1, 2)

            # Apply activation (except last layer)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Residual connection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        if x.shape == identity.shape:
            x = x + identity

        return x


class HandGraphTopology:
    """Định nghĩa cấu trúc graph cho bàn tay theo MediaPipe Hand landmarks"""

    def __init__(self):
        # MediaPipe Hand có 21 landmarks (0-20)
        self.hand_connections = self._create_hand_edges()
        self.body_connections = self._create_body_edges()

    def _create_hand_edges(self):
        """
        Tạo edges theo cấu trúc anatomical thực tế
        MediaPipe Hand landmarks:
        0: WRIST
        1-4: THUMB (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
        5-8: INDEX (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
        9-12: MIDDLE (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
        13-16: RING (RING_MCP, RING_PIP, RING_DIP, RING_TIP)
        17-20: PINKY (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)
        """

        edges = [
            # Thumb chain: wrist -> thumb_cmc -> thumb_mcp -> thumb_ip -> thumb_tip
            [0, 20], [20, 19], [19, 18], [18, 17],
            # Index chain: wrist -> index_mcp -> index_pip -> index_dip -> index_tip
            [0, 4], [4, 3], [3, 2], [2, 1],
            # Middle chain
            [0, 8], [8, 7], [7, 6], [6, 5],
            # Ring chain
            [0, 12], [12, 11], [11, 10], [10, 9],
            # Pinky chain
            [0, 16], [16, 15], [15, 14], [14, 13]
        ]

        # Cross-finger connections (knuckles)
        knuckle_connections = [
            [4, 8],  # INDEX_MCP -> MIDDLE_MCP
            [8, 12],  # MIDDLE_MCP -> RING_MCP
            [12, 16],  # RING_MCP -> PINKY_MCP
        ]
        edges.extend(knuckle_connections)

        # Convert to tensor và add reverse edges (undirected graph)
        edges_tensor = torch.tensor(edges, dtype=torch.long)
        reverse_edges = edges_tensor.flip(dims=[1])
        all_edges = torch.cat([edges_tensor, reverse_edges], dim=0)

        return all_edges.t().contiguous()  # Shape: [2, num_edges]

    def _create_body_edges(self):
        """
        Tạo edges cho body keypoints (12 points theo WLASL dataset)
        Giả định body points: shoulders, elbows, wrists, hips, torso points
        """
        # Định nghĩa connections dựa trên skeleton structure
        # body_edges = [
        #     [0, 1],  # left_shoulder -> right_shoulder
        #     [0, 2],  # left_shoulder -> left_elbow
        #     [1, 3],  # right_shoulder -> right_elbow
        #     [2, 4],  # left_elbow -> left_wrist
        #     [3, 5],  # right_elbow -> right_wrist
        #     [6, 7],  # left_hip -> right_hip
        #     [0, 8],  # left_shoulder -> neck/spine
        #     [1, 8],  # right_shoulder -> neck/spine
        #     [8, 9],  # neck -> torso_center
        #     [9, 6],  # torso_center -> left_hip
        #     [9, 7],  # torso_center -> right_hip
        # ]

        body_edges =[
            [1, 0], [0, 2], [2, 4],
            [1, 0], [0, 3], [3, 5],
            [1, 6], [6, 8], [8, 10],
            [1, 7], [7, 9], [9, 11]
        ]

        edges_tensor = torch.tensor(body_edges, dtype=torch.long)
        reverse_edges = edges_tensor.flip(dims=[1])
        all_edges = torch.cat([edges_tensor, reverse_edges], dim=0)

        return all_edges.t().contiguous()


class MultiPartGCNEncoder(nn.Module):
    """
    GCN Encoder cho multiple body parts (left_hand, right_hand, body)
    """

    def __init__(self, gcn_config=None):
        super(MultiPartGCNEncoder, self).__init__()

        if gcn_config is None:
            gcn_config = {
                'input_dim': 2,
                'hidden_dims': [16, 32],
                'output_dim': 32,
                'dropout': 0.1
            }

        # Tạo topology
        self.topology = HandGraphTopology()

        # GCN encoders cho từng body part
        self.left_hand_gcn = SpatialGCNEncoder(**gcn_config)
        self.right_hand_gcn = SpatialGCNEncoder(**gcn_config)

        # Body GCN có thể khác config vì số joints khác
        body_config = gcn_config.copy()
        body_config['output_dim'] = 24  # Match với body dimension trong code gốc
        self.body_gcn = SpatialGCNEncoder(**body_config)

        # Cache edge indices
        self.register_buffer('hand_edge_index', self.topology.hand_connections)
        self.register_buffer('body_edge_index', self.topology.body_connections)

    def forward(self, left_hand, right_hand, body):
        """
        Args:
            left_hand: [batch, seq_len, 21, 2]
            right_hand: [batch, seq_len, 21, 2]
            body: [batch, seq_len, 12, 2]

        Returns:
            encoded_left_hand: [batch, seq_len, 21 * output_dim]
            encoded_right_hand: [batch, seq_len, 21 * output_dim]
            encoded_body: [batch, seq_len, 12 * body_output_dim]
        """
        batch_size, seq_len = left_hand.shape[:2]
        num_lh_nodes = left_hand.shape[2]
        num_rh_nodes = right_hand.shape[2]
        num_body_nodes = body.shape[2]

        # Reshape để process frame by frame
        lh_reshaped = left_hand.view(batch_size * seq_len, num_lh_nodes, -1)
        rh_reshaped = right_hand.view(batch_size * seq_len, num_rh_nodes, -1)
        body_reshaped = body.view(batch_size * seq_len, num_body_nodes, -1)

        encoded_lh_all = self.left_hand_gcn(lh_reshaped, self.hand_edge_index)
        encoded_rh_all = self.right_hand_gcn(rh_reshaped, self.hand_edge_index)
        encoded_body_all = self.body_gcn(body_reshaped, self.body_edge_index)

        encoded_lh = encoded_lh_all.view(batch_size, seq_len, -1)
        encoded_rh = encoded_rh_all.view(batch_size, seq_len, -1)
        encoded_body = encoded_body_all.view(batch_size, seq_len, -1)

        return encoded_lh, encoded_rh, encoded_body


class CommunicatingEncoderLayer(nn.Module):
    """
    Một lớp Encoder tùy chỉnh thực hiện 3 giai đoạn:
    1. Self-Attention trong mỗi luồng.
    2. Cross-Attention đa hướng giữa các luồng.
    3. Feed-Forward Network.
    """

    def __init__(self, d_model_list, nhead_list, d_ff, dropout, activation, attn_layer_factory):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers
        self.self_attn_lh = attn_layer_factory(d_model_list[0], nhead_list[0])
        self.self_attn_rh = attn_layer_factory(d_model_list[1], nhead_list[1])
        self.self_attn_body = attn_layer_factory(d_model_list[2], nhead_list[2])
        self.norm1_lh = LayerNorm(d_model_list[0])
        self.norm1_rh = LayerNorm(d_model_list[1])
        self.norm1_body = LayerNorm(d_model_list[2])

        # Giai đoạn 2: Cross-Attention & Fusion Layers
        self.lh_to_rh_attn = nn.MultiheadAttention(d_model_list[0], nhead_list[0], kdim=d_model_list[1],
                                                   vdim=d_model_list[1], dropout=dropout, batch_first=False)

        self.rh_to_lh_attn = nn.MultiheadAttention(d_model_list[1], nhead_list[1], kdim=d_model_list[0],
                                                   vdim=d_model_list[0], dropout=dropout, batch_first=False)

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
        # lh_from_body, _ = self.lh_to_body_attn(l_hand_x, body_x, body_x)
        lh_from_rh, _ = self.lh_to_rh_attn(l_hand_x, r_hand_x, r_hand_x)
        lh_fused = self.lh_fusion_layer(lh_from_rh)
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout(lh_fused))

        rh_from_lh, _ = self.rh_to_lh_attn(query=r_hand_x,key= l_hand_x, value=l_hand_x)
        rh_fused = self.rh_fusion_layer(rh_from_lh)
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout(rh_fused))

        # --- 3. Feed-Forward Network ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout(self.ffn_body(body_x)))

        return [l_hand_x, r_hand_x, body_x]


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU(),
                 selected_attn: str = 'prob', output_attention: str = True,
                 IA_decoder: bool = False, inner_classifiers_config: list = None, patience: int = 1,
                 **kwargs):  # Dùng **kwargs cho các tham số không dùng đến

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder

        self.d_model = sum(d_model_list)

        # --- Khởi tạo Encoder ---
        # Hàm factory để tạo các lớp AttentionLayer một cách nhất quán
        def attn_layer_factory(d_model, n_heads):
            Attn = ProbAttention if selected_attn == 'prob' else FullAttention
            return AttentionLayer(Attn(output_attention=output_attention), d_model, n_heads, mix=False)

        # Tạo một danh sách các lớp Encoder giao tiếp. Đây là bộ Encoder DUY NHẤT.
        self.encoder_layers = nn.ModuleList([
            CommunicatingEncoderLayer(d_model_list, nhead_list, dim_feedforward, dropout, activation,
                                      attn_layer_factory)
            for _ in range(num_encoder_layers)
        ])

        # Lớp Norm cuối cùng cho mỗi luồng, sẽ được áp dụng sau khi qua tất cả các lớp
        self.norm_lh = LayerNorm(d_model_list[0])
        self.norm_rh = LayerNorm(d_model_list[1])
        self.norm_body = LayerNorm(d_model_list[2])

        # --- Khởi tạo Decoder ---
        # (Giả định get_custom_decoder không cần thay đổi)
        self.use_IA_decoder = IA_decoder
        self.inner_classifiers_config = inner_classifiers_config
        self.patience = patience
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = dim_feedforward

        self.decoder = self.get_custom_decoder(nhead_list[-1])
        self._reset_parameters()

    def get_custom_decoder(self, nhead):
        # Hàm này không có gì thay đổi
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        if self.use_IA_decoder:
            return PBEEDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                               inner_classifiers_config=self.inner_classifiers_config, patient=self.patience)
        else:
            return TransformerDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                # Các tham số còn lại được gom vào kwargs
                **kwargs) -> Tensor:

        # Vòng lặp Encoder, truyền trực tiếp list 'src'
        for layer in self.encoder_layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Áp dụng lớp chuẩn hóa cuối cùng cho mỗi luồng
        l_hand_memory = self.norm_lh(src[0])
        r_hand_memory = self.norm_rh(src[1])
        body_memory = self.norm_body(src[2])

        # Nối lại để tạo bộ nhớ hoàn chỉnh cho decoder
        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        # Gọi Decoder
        # Truyền các kwargs vào decoder một cách linh hoạt
        output = self.decoder(tgt, full_memory,
                              tgt_mask=kwargs.get('tgt_mask'),
                              memory_mask=kwargs.get('memory_mask'),
                              tgt_key_padding_mask=kwargs.get('tgt_key_padding_mask'),
                              memory_key_padding_mask=kwargs.get('memory_key_padding_mask'))

        return output



class SiFormer(nn.Module):
    def __init__(self, num_classes, num_hid=108, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
                 seq_len=204, device=None, IA_encoder = True, IA_decoder = False, use_gcn=True):
        super(SiFormer, self).__init__()
        print("Feature isolated transformer")

        self.use_gcn = use_gcn

        if use_gcn:
            # GCN Spatial Encoder
            gcn_config = {
                'input_dim': 2,
                'hidden_dims': [16, 32],
                'output_dim': 32,  # Sẽ tạo ra 21*32=672 dim cho mỗi hand
                'dropout': 0.1
            }
            self.spatial_gcn = MultiPartGCNEncoder(gcn_config)

            # Projection layers để match với original dimensions
            self.lh_projection = nn.Linear(21 * 32, 42)  # 672 -> 42
            self.rh_projection = nn.Linear(21 * 32, 42)  # 672 -> 42
            self.body_projection = nn.Linear(12 * 24, 24)  # 288 -> 24
        else:
            # Original flattening approach
            self.lh_projection = None
            self.rh_projection = None
            self.body_projection = None

        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=24))

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.transformer = FeatureIsolatedTransformer(
            [42, 42, 24], [3, 3, 2, 9], num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
            selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
            inner_classifiers_config=[num_hid, num_classes], projections_config=[seq_len, 1],  device=device,
            patience=patience, use_pyramid_encoder=False, distil=False
        )
        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")
        self.projection = nn.Linear(num_hid, num_classes)

    def forward(self, l_hand, r_hand, body, training):

        l_hand = l_hand.float()
        r_hand = r_hand.float()
        body = body.float()

        batch_size = l_hand.size(0) # tương đường với l_hand.shape[0  ] | số lượng record đầu vào
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

        if self.use_gcn:
            # === GCN SPATIAL ENCODING ===
            # Input: [batch, seq_len, joints, coordinates]
            # GCN xử lý spatial relationships giữa các joints
            encoded_lh, encoded_rh, encoded_body = self.spatial_gcn(l_hand, r_hand, body)

            # Project về original dimensions
            new_l_hand = self.lh_projection(encoded_lh)  # [batch, seq_len, 42]
            new_r_hand = self.rh_projection(encoded_rh)  # [batch, seq_len, 42]
            new_body = self.body_projection(encoded_body)  # [batch, seq_len, 24]

        else:
            # === ORIGINAL FLATTENING (for comparison) ===
            new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
            new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
            new_body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))

        # (batch_size, seq_len, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (batch_size, seq_len, feature_size):  (24, 204, 108)
        # new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
        # new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
        # body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))

        
        # (batch_size, seq_len, feature_size) : (24, 204, 108)
        # -> (seq_len, batch_size, feature_size): (204, 24, 108)
        new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_body = new_body.permute(1, 0, 2).type(dtype=torch.float32)

        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        l_hand_in = new_l_hand + self.l_hand_embedding.float()  # Shape remains the same
        r_hand_in = new_r_hand + self.r_hand_embedding.float()  # Shape remains the same
        body_in = new_body + self.body_embedding.float()  # Shape remains the same

        # print('#########################')

        # (seq_len, batch_size, feature_size) -> (batch_size, 1, feature_size): (24, 1, 108)
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
        ).transpose(0, 1)

        '''
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
        ).transpose(0, 1)        
        '''

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.projection(transformer_output).squeeze()
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

