import torch

if __name__ == '__main__':
    # 指定两个分量m,n来创建一个矩阵
    A = torch.arange(20).reshape((4,5))
    print(A)
    # 转置
    print(A.T)
    #对称矩阵A等于其转置
    B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
    print(B)
    print(B == B.T)
    # 给定具有相同形状的任何两个张量，任何按元素二元运算的结果都是相同形状的张量
    A = torch.arange(20,dtype=torch.float32).reshape(5,4)
    B = A.clone() # 通过分配新的内存，将A的一个副本分配给B
    print(A)
    print(A+B)

    # 矩两个矩阵按元素乘法称为哈达玛积
    print(A*B)

    a = 2
    X = torch.arange(24).reshape(2,3,4)
    print(a+X)
    print((a*X).shape)


    A = torch.arange(20*2).reshape(2,5,4)
    print(A.shape)
    print(A.sum())
    print(A)

    A_sum_axis = A.sum(axis=0) # 合第一维度，剩下的后面两个维度的数据 即5*4
    print(A_sum_axis)
    print(A_sum_axis.shape)

    A_sum_axis = A.sum(axis=1)
    print(A_sum_axis)
    print(A_sum_axis.shape)

    A_sumaxis = A.sum(axis=2)
    print(A_sum_axis)
    print(A_sum_axis.shape)

    # 两个维度求和
    A_sum_axis = A.sum(axis=[0,1])
    print(A_sum_axis)
    print(A_sum_axis.shape)

    #平均值  mean 或average
    print(A.mean(dtype=float))
    print(A.sum()/A.numel())

    print(A.mean(axis=0,dtype=float))
    print(A.sum(axis=0)/A.shape[0])

    # 计算总和或均值时保持维度不变。因为按照维度计算和的时候通常会丢掉一个维度
    sum_A = A.sum(axis=1,keepdim=True)
    print(A)
    print(A.sum(axis=1).shape)
    print(A.sum(axis=1))
    print(sum_A)
    print(sum_A.shape)

    # 通过广播将A除以sum_A
    print(A/sum_A)

    # 按某个轴累加求和
    A = torch.arange(20,dtype=torch.float32).reshape(5,4)
    print(A.cumsum(axis=0))

    # 向量点积，按位置相乘的和
    x = torch.arange(4,dtype=torch.float32)
    y = torch.ones(4,dtype=torch.float32)
    print(x)
    print(y)
    print(torch.dot(x,y))
    #可以通过执行按元素乘法，然后求和
    print(torch.sum(x*y))

    #矩阵向量积 Ax是一个长度为m的列向量，其中i(th)元素是点积（(ai)T x）
    print(A.shape)
    print(x.shape)
    print(torch.mv(A,x))

    #矩阵乘法 AB看做是简单的执行m次矩阵向量积，并将结果拼接在一起，形成一个n*m的矩阵
    B = torch.ones(4,3,dtype=torch.float32)
    print(torch.mm(A,B))

    # L2范数，积算术平方根
    u = torch.tensor([3.0,-4.0])
    print(torch.norm(u))
    #L1范数，表示向量元素绝对值和
    print(torch.abs(u).sum())

    #矩阵的F（弗罗贝尼乌斯范数）范数，是矩阵元素的平方和的平方根
    print(torch.norm(torch.ones((4,9))))


