import PositionalEncoding
import torch
if __name__ == '__main__':
    max_pos = 10
    embed_dim = 4

    model = PositionalEncoding.PositionalEncoding.__int__(max_pos, embed_dim)

    x = torch.zeros(2,5,embed_dim)
    print(x)
    out = model(x)
    print(model.PE)
    print(out)

