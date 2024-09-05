import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() >0:
        c = b
    else:
        c = 100 * b
    return c
if __name__ == '__main__':
    # 自动求导
    x = torch.arange(4.0,requires_grad=True)
    print(x)
    # 在计算y对x的梯度之前，需要一个地方存储梯度
    #x.requires_grad(True) # 等价于x=torch.arange(4.0.requires_grad=True)
    print(x.grad) # 这里表示grad可以访问x的梯度，默认是none

    # 计算y
    y = 2 * torch.dot(x,x)
    print(y) 

    #通过调用反向传播函数来自动计算y关于x每个分量的梯度
    y.backward()
    print(x.grad)
    print(x.grad == 4 * x)

    # 在默认情况下，pytorch会累积梯度，我们需要清除之前的值
    x.grad.zero_() # 这里就表示梯度清零
    y = x.sum()
    y.backward()
    print(x.grad)


    # 在深度学习中，我们目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数之和
    # 对非标量调用backforward需要传入一个gradient参数，该参数指定微分函数
    x.grad.zero_()
    y = x*x
    y.sum().backward()   # 等价于 y.backward(torch.ones(len(x)))
    print(x.grad)


    # 将某些计算移动到记录的计算图之外
    x.grad.zero_()
    y = x* x
    u = y.detach()  # 这里表示设置u为一个常数的意思
    z = u * x
    z.sum().backward()
    print(x.grad == u)

    # 同样 y还是x的函数，这里y也可以backforward
    x.grad.zero_()
    y.sum().backward()
    print(x.grad == 2*x)

    #即使构建函数的计算图，需要通过python控制流，如if，循环等。我们仍然可以计算得到变量的梯度
    a = torch.randn(size=(),requires_grad=True)
    d = f(a)
    d.backward()
    print(a.grad == d / a)
