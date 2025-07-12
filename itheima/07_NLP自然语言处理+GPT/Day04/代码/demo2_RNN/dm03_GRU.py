import torch
import torch.nn  as nn

def dm_gru():
    gru = nn.GRU(5, 6, 1,)
    input = torch.randn(2, 3, 5)
    h0 = torch.randn(1, 2, 6)
    output, hn = gru(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')

if __name__ == '__main__':
    dm_gru()