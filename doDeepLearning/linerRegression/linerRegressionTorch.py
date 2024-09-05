import numpy as np
import torch
from  torch.utils import data
from  d2l import torch as d2l

# 通过深度学习框架来简洁的实现线性回归模型，生成数据

#调用框架中现有的api来读取数据
def load_array(data_arrays,batch_size,is_train=True):
    '''构造一个pytorch的数据迭代器'''
    dataset = data.TensorDataset(*data_arrays) # *表示可以接受任意多个元素并将其放入元组
    return data.DataLoader(dataset,batch_size,shuffle=is_train) # shuffle表示是否打乱顺序


if __name__ == '__main__':
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    #生成训练数据
    features, labels = d2l.synthetic_data(true_w,true_b,1000)

    batch_size = 10
    data_iter = load_array((features,labels),batch_size)
    print(next(iter(data_iter)))

    # 模型定义
    from torch import nn
    net = nn.Sequential(nn.Linear(2,1)) #需要明确输入和输出的维度  liner表示的一个全连接层

    net[0].weight.data.normal_(0,0.01)  # 就是设置权重w矩阵  net[0]表示net容器中的线性层（全连接层），因为只有一层
    net[0].bias.data.fill_(0)#这里设置的是偏差b
    # 整个这部分就是给模型的单层全连接层初始化w和b
    loss = nn.MSELoss()  # 均方误差，也是L2范数
    #实例化SGD
    trainer = torch.optim.SGD(net.parameters(),lr=0.03)  #需要传入两个参数，一个是network中的所有参数，包括w和b  另一个是学习率

    # 训练部分
    num_epoch = 3
    for epoch in range(num_epoch):
        for x,y in data_iter:
            l = loss(net(x),y)
            trainer.zero_grad() # 优化器梯度清零
            l.backward()   # 反向传播
            trainer.step() # 进行一次模型更新，即对w和b更新
        l = loss(net(features),labels)
        print(f'epoch{epoch+1},loss{l:f}')