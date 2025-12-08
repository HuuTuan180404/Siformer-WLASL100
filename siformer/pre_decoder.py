import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing import Optional, Union, Callable


isChecked = False


class DecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, **kwargs):
        super(DecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # Change self.multihead_attn to use Pro-sparse attention
        print('Using custom DecoderLayer')

    def forward(self, tgt: Tensor, memory: Tensor, 
                memory_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        global isChecked
        if not isChecked:
            isChecked = True

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# class PBEEDecoder(nn.TransformerDecoder):
#     __constants__ = ['norm']

#     def __init__(self, decoder_layer, num_layers, norm=None, patient=1, 
#                  inner_classifiers_config=None):
#         super(PBEEDecoder, self).__init__(decoder_layer, num_layers, norm)
#         print(f'Using custom PBEEDecoder: num_layers= {num_layers} | patient= {patient}', )
#         self.patience = patient

#         in_dim, out_dim = inner_classifiers_config
#         self.inner_classifiers = nn.ModuleList([
#             nn.Linear(in_dim, out_dim) for _ in range(num_layers)
#         ])

#     def forward(self, tgt: Tensor, memory: Tensor, training: bool) -> Tensor:
#         output = tgt

#         if training or self.patience == 0:
#             print('1')
#             for i, mod in enumerate(self.layers):
#                 output = mod(output, memory)
#         else:
#             print('2')
#             patient_counter = 0
#             prev_pred = None
#             for i, mod in enumerate(self.layers):
#                 output = mod(output, memory)

#                 mod_output = self.norm(output) if self.norm is not None else output

#                 # classifier_out: [L, B, D] => [L, B, num_classes]
#                 classifier_out = self.inner_classifiers[i](mod_output)

#                 # [L, B, num_classes] => (1, B, num_classes)
#                 pooled = classifier_out.mean(dim=0, keepdim=True)

#                 # pred_label: (B)
#                 pred_label = int(torch.argmax(F.softmax(pooled, dim=2)))
#                 print((pred_label))

#                 if (prev_pred is not None) and (pred_label == prev_pred):
#                     patient_counter += 1
#                 else: patient_counter = 0

#                 prev_pred = pred_label

#                 if patient_counter >= self.patience:
#                     break

#         output = self.norm(output) if self.norm is not None else output

#         return output



class PBEEDecoder(nn.TransformerDecoder):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, patient=1, 
                 inner_classifiers_config=None, projections_config=None):
        super(PBEEDecoder, self).__init__(decoder_layer, num_layers, norm)
        print(f'Using custom PBEEDecoder: num_layers= {num_layers} | patient= {patient}', )
        self.patience = patient

        in_dim, out_dim = inner_classifiers_config
        self.inner_classifiers = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_layers)
        ])

        in_dim, out_dim = projections_config
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_layers)
        ])
    
    def forward(self, tgt: Tensor, memory: Tensor, training: bool) -> Tensor:
        output = tgt
        L, B, D = tgt.shape

        if training or self.patience == 0 or B>1:
            for i, mod in enumerate(self.layers):
                output = mod(output, memory)
        else:
            # áp dụng cho eval (B=1)
            patient_counter = 0
            prev_pred = None
            for i, mod in enumerate(self.layers):
                output = mod(output, memory)

                mod_output = self.norm(output) if self.norm is not None else output

                # classifier_out: [L, B, D] => [L, B, num_classes]
                classifier_out = self.inner_classifiers[i](mod_output)

                # [L, B, num_classes] => (B, num_classes, L) => (B, num_classes, 1) => (B, num_classes)
                pooled = self.projections[i](classifier_out.permute(1, 2, 0)).squeeze(-1)

                # pred_label: class | type: int
                pred_label = int(torch.argmax(F.softmax(pooled, dim=1), dim=1))

                if (prev_pred is not None) and (pred_label == prev_pred): 
                    patient_counter += 1
                else: patient_counter = 0

                prev_pred = pred_label

                if patient_counter >= self.patience:
                    break

        output = self.norm(output) if self.norm is not None else output

        return output


if __name__ == '__main__':
    decoder_layer = DecoderLayer(d_model=108, nhead=9)
    decoder = PBEEDecoder(decoder_layer=decoder_layer, num_layers=3, 
                          patient=2, inner_classifiers_config=[108, 100],
                          projections_config=[204, 1])

    tgt = torch.rand(204, 11, 108)
    memory = torch.rand(204, 11, 108)
    output = decoder(tgt, memory, training=False)
    print(output.shape)


#    The reference for the code is the PBEEDecoder class is the following
#    Title: BERT Loses Patience: Fast and Robust Inference with Early Exit
#    Author: Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu1, Furu Wei
#    Availability: https://github.com/JetRunner/PABEE
