import torch
from PositionalEncodingGPT import makPositionalEcondin

if __name__ == '__main__':
    max_pos = 10
    embed_dim = 4

    x = torch.zeros(2, 5, embed_dim)

    model = makPositionalEcondin(max_pos,embed_dim)

    print(model)

    x_pe = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_pe.shape}")

    # 查看前两个位置的编码（以验证sin/cos模式）
    print(f"First position encoding (first 10 elements):\n{x_pe[0, 0, :10]}")
    print(f"Second position encoding (first 10 elements):\n{x_pe[1, 0, :10]}")
