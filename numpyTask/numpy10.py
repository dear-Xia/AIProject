import numpy as np

if __name__ == '__main__':
    # 创建一个长度为10的空向量
    a = np.zeros(10)
    print(a)
    # 查看数组内存大小
    print(a.size * a.itemsize)
    # 创建一个值域10-49的向量
    b = np.arange(10, 49)
    print(b)
    # 反转一个向量
    print(b[:: -1])
    # 创建一个矩阵
    print(np.arange(9).reshape(3,3))

    # 找到数组中非0元素的索引
    print(np.nonzero([1,2,3,0,0,5]))





