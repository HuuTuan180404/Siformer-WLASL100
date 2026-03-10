import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W = nn.Linear(input_dim, 4*hidden_dim)
        self.U = nn.Linear(hidden_dim, 4*hidden_dim, bias=False)

    def forward(self, x_t, h_prev, c_prev):

        gates = self.W(x_t) + self.U(h_prev) 
        # shape: (batch, 4*hidden_dim)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

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
            x_t = x[:, t, :] # (batch, hidden_dim)
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


class GlobalLayer(nn.Module):
    def __init__(self, d_model_list, hidden_dim):
        super().__init__()

        self.lh_dim, self.rh_dim, self.body_dim = d_model_list
        self.bilstm = BiLSTM(sum(d_model_list), hidden_dim)

    def forward(self, lh, rh, bd):
        # src: [B, L, D]

        x = torch.cat((lh, rh, bd), dim = -1)
        x = self.bilstm(x)

        l_out, r_out, b_out = torch.split(
            x,
            [self.lh_dim, self.rh_dim, self.body_dim],
            dim=-1
        )

        return l_out, r_out, b_out

if __name__ == '__main__':
    batch_size = 24
    seq_len = 204
    # input_dim = 
    hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = GlobalLayer(
        d_model_list=[42, 42, 24],
        hidden_dim=2048
    )

    y, _, _ = model(lh, rh, bd)

    print(y.shape)