import torch
import torch.nn as nn
import math


class PositionalEncodingGPT(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingGPT, self).__init__()

        # 创建一个足够大的矩阵，保存位置编码
        pe = torch.zeros(max_len, d_model)

        # 位置编码公式中的 "pos" 部分
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 位置编码公式中的 "i" 部分
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 应用 sin 到偶数索引（2i）
        pe[:, 0::2] = torch.sin(position * div_term)

        # 应用 cos 到奇数索引（2i+1）
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 添加为模型的 buffer，而不是参数（不会被训练）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将输入 x 与相应位置的 pe 相加
        return x + self.pe[:x.size(0), :]
def makPositionalEcondin(d_model, max_len):
    positional = PositionalEncodingGPT(d_model,max_len)
    return positional