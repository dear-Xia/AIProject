import numpy as np

if __name__ == '__main__':
    # 创建一个3*3的单位矩阵
    print(np.eye(3,3))
    # 创建一个3*3*3的随机矩阵
    print(np.random.random((3,3,3)))
    # 创建一个10*10的矩阵并找出最大值和最小值
    a = np.random.random((10,10))
    print(a.max())
    print(a.min())
    #创建一个长度为30的随机向量并找到它的平均值
    a = np.random.random(30)
    print(a.mean())
    # 创建一个二维数组，其中边界为1，其余值为0
    b = np.ones((10,10))
    b[1:-1,1:-1] = 0
    print(b)
    # 对于一个存在的数组，如何添加一个用0填充的边界
    c = np.ones((5,5))
    c = np.pad(c,pad_width=1,mode='constant',constant_values=0)
    print(c)
    # 以下表达式运行结果分别是什么
    print(0*np.nan)
    print(np.nan == np.nan)
    print(np.inf > np.nan)
    print(np.nan - np.nan)
    aa = 0.3
    bb = 3 * 0.1
    print(aa == bb)
    print(type(0.3))
    print(type(3*0.1))
    # 创建一个5*5的矩阵，并设置值1，2，3，4落在其对角线下方位置
    d = np.diag(1+np.arange(4),k=-1)
    print(d)
    # 创建一个8*8的矩阵，并且设置成棋盘的样子

    # 考虑一个（6，7，8）形状的数组，其第100个元素的索引（x，y，z）是什么
    print(np.unravel_index(100,(6,7,8)))