import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    # 传入最大位置长度max_pos和词向量的维度embed_dim
    def __int__(self, max_pos, embed_dim):
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵，全部为0
        PE = torch.zeros(max_pos, embed_dim)
        # 从0到max_pos-1的位置数组pos
        pos = torch.arange(0, max_pos, ).unsqueeze(1).float()
        # 从0开始，生成间隔为2的序列，对应公式2i
        multi_term = torch.arange(0, embed_dim, 2).float()
        # 计算公式中pos对应的系数部分
        # 这里计算的是e^(-log(10000/d))
        # 从数学上计算等价于 1/10000^(2i/d)
        multi_term = torch.exp(multi_term * (-math.log(10000.0) / embed_dim))
        # 使用正弦函数sin和余弦函数生成位置编码矩阵PE

        PE[:, 0::2] = torch.sin(pos * multi_term)
        PE[:, 1::2] = torch.cos(pos * multi_term)
        # 将数组PE注册为一个不需要梯度更新的缓存数组
        # 相当于将位置信息保存在了一个常量数组中
        self.register_buffer('PE', PE.unsqueeze(0))

    # 向前传播函数，函数传入输入数据x
    def forward(self, x):
        # 将x加上位置信息PE
        return x + self
