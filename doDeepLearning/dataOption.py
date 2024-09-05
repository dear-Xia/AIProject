import torch


if __name__ == '__main__':
    # 张量是一个数值组成的数组，数组可以是多维的
    x = torch.arange(12)
    print(x)
    # 通过shape访问形状
    print(x.shape)
    # numel() 表示元素的个数
    print(x.numel())
    # 使用reshape()函数改变张量形状
    print(x.reshape(3,4))
    #可以创建一个初始的全0的张量,也可以使用ones函数创建全1的张量
    print(torch.zeros((2,3,4)))
    #可以通过使用包含数值的python列表来为张量中的每个元素赋值
    print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]))
    #张量做算术运算,案例是按元素进行计算
    x = torch.tensor([1.0,2,4,8])
    y = torch.tensor([2,2,2,2])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)
    print(torch.exp(x))# 指数运算

    #把多个张量连接在一起 cat
    x = torch.arange(12,dtype=torch.float32).reshape((3,4))
    y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    print(torch.cat((x,y),dim=0))# 按照行合并
    print(torch.cat((x,y),dim=1))#按照列合并

    # 通过逻辑运算符构建二元张量
    print(x == y)#按元素比较
    #张量求和,返回的是一个元素的张量
    print(x.sum())

    #广播机制，将两个形状不一样的张量通过复制自身来达到两个张量形状一致，然后再进行运算的方法，这里容易出错
    a = torch.arange(3).reshape((3,1))
    b = torch.arange(2).reshape((1,2))
    print(a)
    print(b)
    print(a + b)

    #元素访问
    # [-1]可以访问最后一个元素，可以使用[1:3]选择第二个和第三个元素--左闭右开
    print(x[-1])
    print(x[1:3])
    # 指定位置的元素值的写入
    x[1,2] = 9
    print(x)
    x[0:2,:] = 12  # 用：表示选中所有的列
    print(x)

    # 简单的内存操作，要防止出现过大的变量被多次复制
    before = id(x) #id表示该变量在内存中的唯一标识
    y = y + x
    print(id(y) == before) # 这里表示结果运算之后，变量就变化了，id就不一致

    z = torch.zeros_like(y)
    print('id(z):',id(z))
    z[:] = x + y  # 表示z中的每一个元素都等于x和y对应位置的元素相加
    print('id(z):', id(z))

    # 如果后续计算中没有重复使用x，可以使用x[:] = x+y或者x+=y来减少操作的内存开销
    x+=y
    print(before == id(x))
    # 转换numpy张量
    A = x.numpy()
    B = torch.tensor(A)
    print(type(A))
    print(type(B))

    # 将大小为1的张量转换为python标量
    a = torch.tensor([3.5])
    print(a)
    print(a.item())
    print(float(a))
    print(int(a))


    # reshape,并没有改变原张量，指向的还是原来的张量地址，新变量更新值，会改变原来数据
    a = torch.arange(12)
    b = a.reshape((3,4))
    b[:] = 2
    print(a)