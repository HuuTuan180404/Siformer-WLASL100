import torch
import torch.nn as nn
from utils import logger
from siformer.local_module import LocalLayer
from siformer.global_module import GlobalLayer
from torch.nn.modules.normalization import LayerNorm
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.attention import AttentionLayer, ProbAttention


def prob_attention_factory(d_model, n_heads, dropout=0.):
    return AttentionLayer(ProbAttention(attention_dropout=dropout , output_attention = True), d_model, n_heads, mix = False)


def multi_head_attention_factory(d_model, n_heads, dropout=0.):
    return nn.MultiheadAttention(d_model, n_heads, dropout=dropout)


class LGBlock(nn.Module):
    def __init__(self, d_model_list, n_heads_list, d_ff, dropout, act, local_attn_type=None, global_attn_type=None, num_layers=0):
        super().__init__()

        assert num_layers > 0, "num_layers must be greater than 0"
        assert global_attn_type in ['self', 'shared'], "global_attn_type must be 'self' or 'shared'"

        local_attn = None
        if local_attn_type == 'shared':
            lh_local_attn = prob_attention_factory(d_model_list[0], n_heads_list[0], dropout)
            rh_local_attn = prob_attention_factory(d_model_list[1], n_heads_list[1], dropout)
            body_local_attn = prob_attention_factory(d_model_list[2], n_heads_list[2], dropout)
            local_attn = [lh_local_attn, rh_local_attn, body_local_attn]

        # global_attn = None
        # if global_attn_type == 'shared':
        #     global_attn = prob_attention_factory(sum(d_model_list), n_heads_list[-1], dropout)
        
        layers = []
        print(f'{num_layers} local layers')
        for i in range(num_layers):
            layers.append(
                LocalLayer(
                    d_model_list=d_model_list,
                    n_heads_list=n_heads_list,
                    d_ff=d_ff,
                    attn_list=local_attn,
                    act=act,
                    dropout=dropout
                )
            )

        layers.append(
            GlobalLayer(
                d_model_list=d_model_list,
                hidden_dim=512,
            )
        )
        
        self.layers = nn.ModuleList(layers)

    def forward(self, lh, rh, body):
        # lh, rh, body: (B, L, D)
        for layer in self.layers:
            lh, rh, body = layer(lh, rh, body)
        return lh, rh, body


class MyModel(nn.Module):
    def __init__(self, num_classes=100, num_hid=108, d_model_list=[42, 42, 24], 
                 n_heads_list=[3, 3, 2, 9], 
                  num_enc_layers = 4, num_dec_layers = 3, pat_dec = 2,
                 seq_len = 204, device = None):
        super(MyModel, self).__init__()

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
        self.encoder = LGBlock(
            d_model_list=d_model_list,
            n_heads_list=n_heads_list,
            d_ff=2048,
            dropout=0.1,
            act='gelu',
            local_attn_type = 'shared',
            global_attn_type='self',
            num_layers=num_enc_layers,
        )

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))

        self.decoder = self.get_custom_decoder(n_heads_list[-1])

        self.projection = nn.Linear(num_hid, num_classes)

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
        l_hand_out, r_hand_out, body_out = self.encoder(l_hand_in, r_hand_in, body_in)

        # full memory
        full_memory = torch.cat((l_hand_out, r_hand_out, body_out), dim = -1) # [B, L, D_sum]
        full_memory = full_memory.permute(1, 0, 2) # [L, B, D_sum]

        # decoder
        decoder_out = self.decoder(self.class_query.repeat(1, batch_size, 1), full_memory, training=training)

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