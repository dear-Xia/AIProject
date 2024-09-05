import  torch
import PositionalEconding.PositionalEncodingGPT

if __name__ == '__main__':
    d_model = 512  # 位置编码的维度，通常与嵌入维度一致
    seq_len = 60  # 输入序列的长度
    batch_size = 32  # 批次大小

    # 创建一个随机输入
    x = torch.randn(seq_len, batch_size, d_model)

    # 初始化位置编码模块
    pe = PositionalEconding.PositionalEncodingGPT(d_model=d_model, max_len=5000)

    # 通过位置编码模块运行输入
    x_pe = pe(x)

    # 输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_pe.shape}")

    # 查看前两个位置的编码（以验证sin/cos模式）
    print(f"First position encoding (first 10 elements):\n{x_pe[0, 0, :10]}")
    print(f"Second position encoding (first 10 elements):\n{x_pe[1, 0, :10]}")