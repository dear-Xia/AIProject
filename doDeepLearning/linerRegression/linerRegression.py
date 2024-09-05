import random
import torch
from d2l import torch as d2l

# 从0开始实现整个方法，包括数据流水线，模型，损失函数和小批量随机梯度下降优化器

# 根据带有噪声的线性模型构造一个人造数据集。使用线性模型参数W=[2,-3.4]T,b=4.2和噪声项生成数据集和标签
def synthetic_data(w,b,num_examples):
    ''''生成 y=Xw+b+噪声'''
    x = torch.normal(0,1,(num_examples,len(w)))  # 均值为0，方差为1，形状为第三个参数的随机矩阵
    #print(x)
    y = torch.matmul(x,w)+b   # y = Xw +b
    #print(y)
    y += torch.normal(0,0.01,y.shape)  # 增加了一个均值为0，方差为0.01形状和y相同的随机噪音
    #print(y)
    # y是一个行向量，这里通过reshape转成列向量
    return x,y.reshape((-1,1))# -1表示根据输入来判定

# 定义一个函数，来每次读取小批量的数据
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本是随机读取的，没有特定顺序
    random.shuffle(indices)  # 这个操作就是打算数据
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

# 定义线性模型
def linreg(x,w,b):
    ''''线性回归模型'''
    return torch.matmul(x,w)+b

#定义损失函数
def squared_loss(y_hat,y):
    '''均方误差'''
    return (y_hat-y.reshape(y_hat.shape))**2 /2

#定义优化算法
def sgd(params,lr,batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():
        for param in params:
            param-= lr*param.grad /batch_size
            param.grad.zero_()
if __name__ == '__main__':
    ture_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(ture_w,true_b,1000)
    # features中的每一行都包含一个二维数据样本，labels中的每一行都包含一维标签值（一个标量 ）
    print('feature:',features[0],'\nlabel:',labels[0])
    d2l.set_figsize()
    d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
    #d2l.plt.show()

    batch_size = 10

    for x,y in data_iter(batch_size,features,labels):
        print(x,'\n',y)
        break


    w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)


    lr = 0.03  # 学习率  # 不能太小，也不能太大，太小计算梯度的次数太多，太大会造成震荡
    num_epochs = 10
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for x,y in data_iter(batch_size,features,labels):  # 获取小批量的数据
            l = loss(net(x,w,b),y)  #计算的是x和y的小批量损失
            # 因为l的形状是（batch_size,1）,而不是一个标量。l中的所有元素都被加到
            l.sum().backward()
            sgd([w,b],lr,batch_size) # 使用参数的梯度更惨参数
        with torch.no_grad():  # 表示下面的代码块不需要计算梯度
            train_l = loss(net(features,w,b),labels)
            print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')

    '''
    训练过程
        首先就是初始化w，b
        然后定义训练次数
        然后就是获取小批量的训练数据，即x,y
        然后计算这一小批量数据的loss
        然后loss.sum去反向传播
        然后调用sgd优化算法，在优化算法里面更新w和b，即params -= params.grad/size
        最后执行完一次训练后，打印一次均方误差
        
    '''
