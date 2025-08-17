import copy
import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder

from typing import Optional, Union, Callable
from siformer.attention import AttentionLayer, ProbAttention, FullAttention
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, PBEEncoder
from siformer.utils import get_sequence_list

import uuid


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])

class SepTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            self.layers.append(
                nn.Conv1d(input_channels, out_channels, kernel_size, padding=kernel_size//2)
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch_size, features, seq_len]
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, out_channels]
        x = self.norm(x)
        return x


class AnatomicalGCN(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[16, 32, 64]):

        super(AnatomicalGCN, self).__init__()

        dims = [input_dim] + hidden_dims  # Ví dụ: [2, 16, 32, 64]
        self.gcn_layers = nn.ModuleList([
            GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

        self.hand_edges = self._create_hand_topology()
        self.body_edges = self._create_body_topology()

        dims = [input_dim] + hidden_dims  # Ví dụ: [2, 16, 32, 64]

        # self.hand_gcn_layers = nn.ModuleList([
        #     GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)
        # ])

        # self.body_gcn_layers = nn.ModuleList([
        #     GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)
        # ])


    def _create_hand_topology(self):
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]
        return torch.tensor(edges).t().contiguous()

    def _create_body_topology(self):
        edges = [
            [0, 1], [1, 2], [2, 3], 
            [0, 4], [4, 5], [5, 6], 
            [0, 7],
            [7, 8], [8, 9],
            [7, 10], [10, 11]
        ]
        return torch.tensor(edges).t().contiguous()

    def forward_hand(self, x, edge_index):
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def forward_body(self, x, edge_index):
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 selected_attn: str = 'prob', output_attention: str = True,
                 inner_classifiers_config: list = None, patience: int = 1, use_pyramid_encoder: bool = False,
                 distil: bool = False, projections_config: list = None,
                 IA_encoder: bool = False, IA_decoder: bool = False, device=None):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder
        self.d_model = sum(d_model_list)
        self.d_ff = dim_feedforward
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
        self.l_hand_encoder = self.get_custom_encoder(d_model_list[0], nhead_list[0])
        self.r_hand_encoder = self.get_custom_encoder(d_model_list[1], nhead_list[1])
        self.body_encoder = self.get_custom_encoder(d_model_list[2], nhead_list[2])
        self.decoder = self.get_custom_decoder(nhead_list[-1])
        self._reset_parameters()

    def get_custom_encoder(self, f_d_model: int, nhead: int):
        Attn = ProbAttention if self.selected_attn == 'prob' else FullAttention
        print(f'self.selected_attn {self.selected_attn}')

        if self.use_pyramid_encoder:
            print("Pyramid encoder")
            print(f'self.distl {self.distil}')
            e_layers = get_sequence_list(self.num_encoder_layers)
            inp_lens = list(range(len(e_layers)))
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(output_attention=self.output_attention),
                                f_d_model, nhead, mix=False),
                            f_d_model,
                            self.d_ff,
                            dropout=self.dropout,
                            activation=self.activation
                        ) for _ in range(el)
                    ],
                    [
                        ConvLayer(
                            f_d_model, self.device
                        ) for _ in range(self.num_encoder_layers - 1)
                    ] if self.distil else None,
                    norm_layer=torch.nn.LayerNorm(f_d_model)
                ) for el in e_layers]

            encoder = EncoderStack(encoders, inp_lens)
        else:
            encoder_layer = TransformerEncoderLayer(f_d_model, nhead, self.d_ff, self.dropout, self.activation)
            encoder_layer.self_attn = AttentionLayer(
                Attn(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            encoder_norm = LayerNorm(f_d_model)

            if self.use_IA_encoder:
                print("Encoder with input adaptive")
                self.inner_classifiers_config[0] = f_d_model
                encoder = PBEEncoder(
                    encoder_layer, self.num_encoder_layers, norm=encoder_norm,
                    inner_classifiers_config=self.inner_classifiers_config,
                    projections_config=self.projections_config,
                    patience=self.patience
                )
            else:
                print("Normal encoder")
                encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, norm=encoder_norm)

        return encoder

    def get_custom_decoder(self, nhead):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        if self.use_IA_decoder:
            print("Decoder with with input adaptive")
            return PBEEDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                inner_classifiers_config=self.inner_classifiers_config, patient=self.patience
            )
        else:
            print("Normal decoder")
            return TransformerDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm)

    def checker(self, full_src, tgt, is_batched):
        if not self.batch_first and full_src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and full_src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if full_src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, training: bool = True) -> Tensor:

        full_src = torch.cat(src, dim=-1)
        self.checker(full_src, tgt, full_src.dim() == 3)

        id = uuid.uuid1()
        # code for concurrency is removed...
        if self.use_IA_encoder:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
        else:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        if self.use_IA_decoder:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask, training=training)
        else:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return output


class SiFormer(nn.Module):
    def __init__(self, num_classes, num_hid=108, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
                 seq_len=204, device=None, IA_encoder = True, IA_decoder = False):
        super(SiFormer, self).__init__()
        print("Feature isolated transformer with Sep-TCN + GCN")

        self.seq_len = seq_len
        self.device = device

        self.anatomical_gcn = AnatomicalGCN(input_dim=2, hidden_dims=[16, 32, 64])

        # === Sep-TCN Components ===
        # Sau GCN: 21 keypoints * 64 features = 1344 channels cho hand
        self.l_hand_tcn = SepTCN(in_channels=21*64, out_channels=128, kernel_size=3, num_layers=3)
        self.r_hand_tcn = SepTCN(in_channels=21*64, out_channels=128, kernel_size=3, num_layers=3)
        # Sau GCN: 12 keypoints * 64 features = 768 channels cho body
        self.body_tcn = SepTCN(in_channels=12*64, out_channels=64, kernel_size=3, num_layers=3)

        # === Positional Embeddings ===
        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=128))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=128))
        self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=64))

        # === Transformer ===
        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.transformer = FeatureIsolatedTransformer(
            [128,128,64], [4, 4, 2, 8], num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
            selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
            inner_classifiers_config=[num_hid, num_classes], projections_config=[seq_len, 1],  device=device,
            patience=patience, use_pyramid_encoder=False, distil=False
        )

        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")
        self.projection = nn.Linear(num_hid, num_classes)

        if device:
            self.hand_edges = self.anatomical_gcn.hand_edges.to(device)
            self.body_edges = self.anatomical_gcn.body_edges.to(device)
        else:
            self.hand_edges = self.anatomical_gcn.hand_edges
            self.body_edges = self.anatomical_gcn.body_edges

    def process_with_gcn_tcn(self, keypoints, keypoint_type='hand'):
        """
        Process keypoints through GCN -> reshape -> TCN pipeline
        
        Args:
            keypoints: [batch_size, seq_len, num_keypoints, 2]
            keypoint_type: 'hand' or 'body'
        Returns:
            features: [seq_len, batch_size, feature_dim]
        """
        batch_size, seq_len, num_keypoints, coord_dim = keypoints.shape

        keypoints = keypoints.to(dtype=torch.float32)
        
        # Reshape for GCN processing: [batch_size * seq_len, num_keypoints, 2]
        keypoints_reshaped = keypoints.view(batch_size * seq_len, num_keypoints, coord_dim)
        
        # Apply GCN
        if keypoint_type == 'hand':
            # Repeat edges for batch processing
            edge_index = self.hand_edges
            batch_edges = []
            for i in range(batch_size * seq_len):
                batch_edges.append(edge_index + i * num_keypoints)
            batch_edge_index = torch.cat(batch_edges, dim=1)
            
            # Flatten keypoints for GCN: [batch_size * seq_len * num_keypoints, 2]
            x_flat = keypoints_reshaped.view(-1, coord_dim)
            gcn_features = self.anatomical_gcn.forward_hand(x_flat, batch_edge_index)
            
        elif keypoint_type == 'body':
            edge_index = self.body_edges
            batch_edges = []
            for i in range(batch_size * seq_len):
                batch_edges.append(edge_index + i * num_keypoints)
            batch_edge_index = torch.cat(batch_edges, dim=1)
            
            x_flat = keypoints_reshaped.view(-1, coord_dim)
            gcn_features = self.anatomical_gcn.forward_body(x_flat, batch_edge_index)
        
        # Reshape GCN output: [batch_size, seq_len, num_keypoints * feature_dim]
        gcn_output_dim = gcn_features.shape[-1]  # 64
        gcn_features = gcn_features.view(batch_size, seq_len, num_keypoints * gcn_output_dim)
        
        # Apply Sep-TCN
        if keypoint_type == 'hand':
            tcn_features = self.l_hand_tcn(gcn_features)  # [batch_size, seq_len, 128]
        elif keypoint_type == 'body':
            tcn_features = self.body_tcn(gcn_features)   # [batch_size, seq_len, 64]
        
        # Convert to transformer format: [seq_len, batch_size, feature_dim]
        tcn_features = tcn_features.permute(1, 0, 2).type(torch.float32)
        
        return tcn_features
    
    def forward(self, l_hand, r_hand, body, training):
        """
        Forward pass with GCN + Sep-TCN integration
        
        Args:
            l_hand: [batch_size, seq_len, 21, 2] - Left hand keypoints
            r_hand: [batch_size, seq_len, 21, 2] - Right hand keypoints  
            body: [batch_size, seq_len, 12, 2] - Body keypoints
            training: bool
        """

        batch_size = l_hand.size(0)

        l_hand_features = self.process_with_gcn_tcn(l_hand, 'hand')  # [seq_len, batch_size, 128]
        r_hand_features = self.process_with_gcn_tcn(r_hand, 'hand')  # [seq_len, batch_size, 128] 
        body_features = self.process_with_gcn_tcn(body, 'body')      # [seq_len, batch_size, 64]

        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        l_hand_in = l_hand_features + self.l_hand_embedding  # Shape remains the same
        r_hand_in = r_hand_features   + self.r_hand_embedding  # Shape remains the same
        body_in = body_features + self.body_embedding  # Shape remains the same

        # (seq_len, batch_size, feature_size) -> (batch_size, 1, feature_size): (24, 1, 108)
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
        ) # [batch_size, 1, num_hid]

        # final_output = transformer_output[-1]  # [seq_len, batch_size, num_classes]
        # out = final_output.transpose(0, 1).squeeze(1)  # [batch_size, num_classes]

        # Permute to (batch_size, target_seq_len, d_model) -> (batch_size, 1, 320)
        output = transformer_output.permute(1, 0, 2)
        out = self.projection(output).squeeze(1)
    
        # === Final Classification ===
        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        # out = self.projection(transformer_output).squeeze() # [batch_size, num_classes]
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

