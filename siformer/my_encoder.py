import torch

from torch import nn, Tensor
from siformer.attention import AttentionLayer, ProbAttention
# from attention import AttentionLayer, ProbAttention



class EncoderLayer(nn.Module):
    def __init__(self,
                 self_attn_lh, self_attn_rh, self_attn_body,
                 joints_list=[21, 21, 12], nhead_list=[3, 3, 2],
                 d_ff=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        self.dim_lh = joints_list[0]*2
        self.dim_rh = joints_list[1]*2
        self.dim_body = joints_list[2]*2

        # --- Norm ---
        self.norm1_lh = nn.LayerNorm(self.dim_lh)
        self.norm1_rh = nn.LayerNorm(self.dim_rh)
        self.norm1_body = nn.LayerNorm(self.dim_body)
        
        self.norm2_lh = nn.LayerNorm(self.dim_lh)
        self.norm2_rh = nn.LayerNorm(self.dim_rh)
        self.norm2_body = nn.LayerNorm(self.dim_body)

        self.norm3_lh = nn.LayerNorm(self.dim_lh)
        self.norm3_rh = nn.LayerNorm(self.dim_rh)
        self.norm3_body = nn.LayerNorm(self.dim_body)

        # --- dropout ---
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # --- Activation ---
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

        # --- Self-Attention ---
        self.attention_lh = self_attn_lh
        self.attention_rh = self_attn_rh
        self.attention_body = self_attn_body

        # --- Communication ---
        self.lh_from_rh_attn = nn.MultiheadAttention(
            embed_dim=self.dim_lh,
            num_heads=nhead_list[0],
            kdim=self.dim_rh,
            vdim=self.dim_rh,
            dropout=dropout,
            batch_first=False
            )
        self.rh_from_lh_attn = nn.MultiheadAttention(
            embed_dim=self.dim_rh,
            num_heads=nhead_list[1],
            kdim=self.dim_lh,
            vdim=self.dim_lh,
            dropout=dropout,
            batch_first=False
            )

        # --- Fusion ---
        self.lh_fusion_layer = nn.Linear(self.dim_lh, self.dim_lh)
        self.rh_fusion_layer = nn.Linear(self.dim_rh, self.dim_rh)

        # --- FFN ---
        self.ffn_lh = nn.Sequential(
            nn.Linear(self.dim_lh, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.dim_lh)
        )
        self.ffn_rh = nn.Sequential(
            nn.Linear(self.dim_rh, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.dim_rh)
        )
        self.ffn_body = nn.Sequential(
            nn.Linear(self.dim_body, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.dim_body)
        )

    def forward(self, l_hand_x, r_hand_x, body_x):
        # INPUT: (L, B, N*C)
        L, B, D = l_hand_x.shape

        # --- Self-Attention => Dropout => Residual => Norm ---
        lh_self, _ = self.attention_lh(l_hand_x, l_hand_x, l_hand_x)
        l_hand_x = self.norm1_lh(l_hand_x + self.dropout_attn(lh_self))

        rh_self, _ = self.attention_rh(r_hand_x, r_hand_x, r_hand_x)
        r_hand_x = self.norm1_rh(r_hand_x + self.dropout_attn(rh_self))

        body_self, _ = self.attention_body(body_x, body_x, body_x)
        body_x = self.norm1_body(body_x + self.dropout_attn(body_self))

        # --- Communication => Linear => Dropout => Residual => Norm ---
        lh_from_rh, _ = self.lh_from_rh_attn(l_hand_x, r_hand_x, r_hand_x)
        lh_fused = self.lh_fusion_layer(lh_from_rh)
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout_fusion(lh_fused))

        rh_from_lh, _ = self.rh_from_lh_attn(r_hand_x, l_hand_x, l_hand_x)
        rh_fused = self.rh_fusion_layer(rh_from_lh)
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout_fusion(rh_fused))

        # --- FFN ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout_ffn(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout_ffn(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout_ffn(self.ffn_body(body_x)))

        # OUT: (L, B, D)
        return l_hand_x, r_hand_x, body_x

class PBEEncoder(nn.TransformerEncoder):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, 
                 norm=None, enable_nested_tensor=False):
        super(PBEEncoder, self).__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
        print(f'Using custom PBEEncoder: num_layers= {num_layers}')

    def forward(self, lh, rh, body):
        # lh, rh, body: (L, B, D)
        for i, mod in enumerate(self.layers):
            lh, rh, body = mod(lh, rh, body)
            # output_lh, output_rh, output_body: (L, B, D)

        return lh, rh, body # (L, B, D)


if __name__ == '__main__':
    attn_lh = AttentionLayer(ProbAttention(), 42, 3, mix = False)
    attn_rh = AttentionLayer(ProbAttention(), 42, 3, mix = False)
    attn_body = AttentionLayer(ProbAttention(), 24, 2, mix = False)

    norm_lh = nn.LayerNorm(42)
    norm_rh = nn.LayerNorm(42)
    norm_body = nn.LayerNorm(24)

    encoder_layer = EncoderLayer(
        self_attn_lh=attn_lh,
        self_attn_rh=attn_rh,
        self_attn_body=attn_body,
        joints_list=[21, 21, 12],
        nhead_list=[3, 3, 2],
        d_ff=2048,
        dropout=0.1,
        activation="relu"
    )

    encoder = PBEEncoder(
        encoder_layer=encoder_layer,
        num_layers=3,
        enable_nested_tensor=False,
    )

    l_hand = torch.rand(204, 24, 42)
    r_hand = torch.rand(204, 24, 42)
    body = torch.rand(204, 24, 24)

    output_lh, output_rh, output_body = encoder(l_hand, r_hand, body, training=True)

    print(output_lh.shape)
    print(output_rh.shape)
    print(output_body.shape)


# class PBEEncoder(nn.TransformerEncoder):
#     __constants__ = ['norm']
#     def __init__(self, encoder_layer, num_layers, 
#                  norm_lh, norm_rh, norm_body, 
#                  inner_classifiers_config_lh, 
#                  inner_classifiers_config_rh, 
#                  inner_classifiers_config_body,
#                  projections_config_lh, 
#                  projections_config_rh,
#                  projections_config_body,
#                  pat_enc=1, 
#                  norm=None, enable_nested_tensor=False):
#         super(PBEEncoder, self).__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
#         print(f'Using custom PBEEncoder: num_layers= {num_layers} | patience= {pat_enc}', )
#         self.norm_lh = norm_lh
#         self.norm_rh=norm_rh
#         self.norm_body=norm_body
#         self.patience = pat_enc

#         # --- inner classifierss ---
#         self.inner_classifiers_lh = nn.ModuleList(
#             [nn.Linear(inner_classifiers_config_lh[0], inner_classifiers_config_lh[1])
#              for _ in range(num_layers)]
#         )
#         self.inner_classifiers_rh = nn.ModuleList(
#             [nn.Linear(inner_classifiers_config_rh[0], inner_classifiers_config_rh[1])
#              for _ in range(num_layers)]
#         )
#         self.inner_classifiers_body = nn.ModuleList(
#             [nn.Linear(inner_classifiers_config_body[0], inner_classifiers_config_body[1])
#              for _ in range(num_layers)]
#         )

#         # --- projection ---
#         self.projections_lh = nn.ModuleList(
#             [nn.Linear(projections_config_lh[0], projections_config_lh[1])
#              for _ in range(num_layers)]
#         )
#         self.projections_rh = nn.ModuleList(
#             [nn.Linear(projections_config_rh[0], projections_config_rh[1])
#              for _ in range(num_layers)]
#         )
#         self.projections_body = nn.ModuleList(
#             [nn.Linear(projections_config_body[0], projections_config_body[1])
#              for _ in range(num_layers)]
#         )

#     def forward(self, lh, rh, body, training):
#         # lh, rh, body: (L, B, D)
#         output_lh, output_rh, output_body = lh, rh, body

#         # if training or self.patience == 0:
#         for i, mod in enumerate(self.layers):
#             output_lh, output_rh, output_body = mod(output_lh, output_rh, output_body)
#             # output_lh, output_rh, output_body: (L, B, D)
#         else:
#             stop_lh = stop_rh = stop_body = False
#             patient_counter_lh, patient_counter_rh, patient_counter_body = 0, 0, 0
#             patient_result_lh, patient_result_rh, patient_result_body = None, None, None
#             for i, mod in enumerate(self.layers):
#                 if all([stop_lh, stop_rh, stop_body]):
#                     break

#                 lh_in = output_lh if not stop_lh else output_lh.detach()  # hoặc giữ nguyên, không update
#                 rh_in = output_rh if not stop_rh else output_rh.detach()
#                 body_in = output_body if not stop_body else output_body.detach()

#                 output_lh, output_rh, output_body = mod(lh_in, rh_in, body_in)

#                 # (L, B, D)
#                 mod_output_lh, mod_output_rh, mod_output_body = output_lh, output_rh, output_body

#                 # (L, B, D)
#                 mod_output_lh = self.norm_lh(mod_output_lh) if self.norm_lh is not None else mod_output_lh
#                 mod_output_rh = self.norm_rh(mod_output_rh) if self.norm_rh is not None else mod_output_rh
#                 mod_output_body = self.norm_body(mod_output_body) if self.norm_body is not None else mod_output_body

#                 # classifier_out: [L, B, D] => [L, B, C]
#                 classifier_out_lh = self.inner_classifiers_lh[i](mod_output_lh)
#                 classifier_out_rh = self.inner_classifiers_rh[i](mod_output_rh)
#                 classifier_out_body = self.inner_classifiers_body[i](mod_output_body)

#                 # projection_out: (L, B, C) => (B, C, L) => (B, C, 1) => (B, C)=(B, 100)
#                 projection_out_lh = self.projections_lh[i](classifier_out_lh.permute(1, 2, 0)).squeeze(-1) # (B, 100)
#                 projection_out_rh = self.projections_rh[i](classifier_out_rh.permute(1, 2, 0)).squeeze(-1) # (B, 100)
#                 projection_out_body = self.projections_body[i](classifier_out_body.permute(1, 2, 0)).squeeze(-1) # (B, 100)

#                 # (B)
#                 labels_lh = projection_out_lh.detach().argmax(dim=1)
#                 labels_rh = projection_out_rh.detach().argmax(dim=1)
#                 labels_body = projection_out_body.detach().argmax(dim=1)

#                 if not stop_lh:
#                     if patient_result_lh is not None:
#                         patient_labels_lh = patient_result_lh.detach().argmax(dim=1) # (B)
#                     if (patient_result_lh is not None) and torch.all(labels_lh.eq(patient_labels_lh)):
#                         patient_counter_lh += 1
#                     else:
#                         patient_counter_lh = 0
#                     patient_result_lh = projection_out_lh
#                     if patient_counter_lh == self.patience:
#                         stop_lh = True
#                 if not stop_rh:
#                     if patient_result_rh is not None:
#                         patient_labels_rh = patient_result_rh.detach().argmax(dim=1) # (B)
#                     if (patient_result_rh is not None) and torch.all(labels_rh.eq(patient_labels_rh)):
#                         patient_counter_rh += 1
#                     else:
#                         patient_counter_rh = 0
#                     patient_result_rh = projection_out_rh
#                     if patient_counter_rh == self.patience:
#                         stop_rh=True
#                 if not stop_body:
#                     if patient_result_body is not None:
#                         patient_labels_body = patient_result_body.detach().argmax(dim=1) # (B)
#                     if (patient_result_body is not None) and torch.all(labels_body.eq(patient_labels_body)):
#                         patient_counter_body += 1
#                     else:
#                         patient_counter_body = 0
#                     patient_result_body = projection_out_body
#                     if patient_counter_body == self.patience:
#                         stop_body=True

#         return output_lh, output_rh, output_body # (L, B, D)


