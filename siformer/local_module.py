import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # input -> hidden
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_g = nn.Linear(input_dim, hidden_dim)

        # hidden -> hidden
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t, h_prev, c_prev):

        i = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        f = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        o = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        g = torch.tanh(self.W_g(x_t) + self.U_g(h_prev))

        c_t = f * c_prev + i * g # (batch, hidden_dim)
        h_t = o * torch.tanh(c_t) # (batch, hidden_dim)

        return h_t, c_t


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c = self.cell(x_t, h, c)
            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        return outputs


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super().__init__()

        output_dim = input_dim if output_dim is None else output_dim

        self.forward_lstm = LSTMLayer(input_dim, hidden_dim)
        self.backward_lstm = LSTMLayer(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_ dim)

        # Forward direction
        out_forward = self.forward_lstm(x)

        # Backward direction (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        out_backward = self.backward_lstm(x_reversed)
        out_backward = torch.flip(out_backward, dims=[1])

        # Concat forward + backward
        out = torch.cat([out_forward, out_backward], dim=2)
        # (batch, seq_len, hidden_dim*2)

        output = self.fc(out)
        # (batch, seq_len, output_dim)

        return output


class LocalLayer(nn.Module):
    def __init__(self, d_model_list, hidden_dim):
        super().__init__()
        self.l_hand = BiLSTM(d_model_list[0], hidden_dim)
        self.r_hand = BiLSTM(d_model_list[1], hidden_dim)
        self.body = BiLSTM(d_model_list[2], hidden_dim)

    def forward(self, lh, rh, bd):
        # src: [B, L, D]

        lh = self.l_hand(lh)
        rh = self.r_hand(rh)
        bd = self.body(bd)

        return lh, rh, bd

if __name__ == '__main__':
    batch_size = 24
    seq_len = 204
    # input_dim = 
    hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = LocalLayer(
        d_model_list=[42, 42, 24],
        hidden_dim=2048
    )

    y, _, _ = model(lh, rh, bd)

    print(y.shape)