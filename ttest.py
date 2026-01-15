import torch

from siformer.medvitv2 import SLMedViTV2

if __name__ == '__main__':
    l_hand = torch.randn(24, 204, 21, 2)
    r_hand = torch.randn(24, 204, 21, 2)
    body = torch.randn(24, 204, 12, 2)

    model = SLMedViTV2()

    out = model(l_hand, r_hand, body)
    print(out.shape)

    # L = 204
    # w = 7
    # pad_len = (w - L % w) % w
    # print(L % w)