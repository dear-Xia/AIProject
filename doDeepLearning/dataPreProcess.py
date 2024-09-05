import os
import pandas as pd
import torch


def init_data():
    # 创建一个人数据集，并存储在csv（逗号分隔值）文件
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建一个文件夹，存储数据
    data_file = os.path.join('..', 'data', 'house_tiny.csv')  # 在数据文件夹里面创建一个csv文件用来存储数据
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 数据样例
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


if __name__ == '__main__':

    # init_data()
    # 数据读取
    data = pd.read_csv('..\\data\\house_tiny.csv')
    print(data)

    # 处理缺失的数据，典型的方法如，插值和删除，这里考虑插值
    inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]  # 第一个参数是行，第二个参数是列，如例，第一列和第二列的所有行取出来作为inputs
    print(inputs)
    print(outputs)
    #print(type(inputs))
    #print(inputs.mean())
    inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())  # 这里是因为这里的值读取的时候alley列默认是str，无法填充，所以指定具体的列填充
    # 这里 mean是表示插入的是NumRooms 列的均值
    print(inputs)

    # 对于inputs中出现的str，对应成类别，NAN也视为类别。使用类别之或者离散值来替换
    inputs = pd.get_dummies(inputs,dummy_na=True,dtype=float)  # torch2中默认值给了true和false，所以可以设定填充的值为数值
    print(inputs)

    # inputs和outputs中的所有的数据都是数值类型，可以转换为张量格式
    x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
    print(x)
    print(y)


