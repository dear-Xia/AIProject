import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import  display

def create_dataset():
    # 通过使用totensor实例将图像数据从PIL类型转成32位浮点数格式
    # 并除以255使得所有像素的数值均在0-1之间
    trans = transforms.ToTensor
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    print(len(mnist_test))
    print(len(mnist_train))
    print(mnist_train)
    print(mnist_train[0][0].shape)


## 两个可视化数据集的函数
def get_fashion_mnist_lables(lables):
    '''返回Fashion-MNIST数据集'''
    text_lables = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_lables[int(i)]for i in lables]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    '''Polt a list of images.'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
def getdataloader_workers():
    '''使用4个进程来读取数据'''
    return 8
# 数据加载函数
def loda_data_fashion_mnist(batch_size,resize=None):
    '''下载数据集，然后加载到内存'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=getdataloader_workers()),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=getdataloader_workers()))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)  # 按行求和
    return X_exp / partition  #这里是应用广播机制，将partition复制扩展位和X_exp形状相同

# 定义softmax回归模型
def net(X,W,b):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

# 实现交叉熵函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

# 评估任意模型net在数据迭代器中的准确度
def  evaluate_accuracy(net,data_iter):
    '''计算在特定数据集上模型的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模型  训练的时候用net.test()  因为eval() 开了参数就不再更新
    metric = Accumulator(2)    # 正确预测数，预测总数
    for x,y in data_iter:
        metric.add(accuracy(net(x),y),y.numel())  # 两个参数，第一个是预测中的数量，第二个是全部的数量
    return metric[0] / metric[1]

#预测类别和真是y元素进行比较
def accuracy(y_hat,y):
    '''计算正确的数量'''
    if len(y_hat.shape)>1 and  y_hat.shape[1] > 1 :
        y_hat = y_hat.argmax(axis=1)      # 返回數組中最大元組索引  1表示的是消灭列1*2，保留行，0表示消灭行保留列  1*3   这里按行降维的结果就是标签id
    cmp = y_hat.type(y.dtype) == y   # 这里是根据标签id和预测标签id进行比较
    return float(cmp.type(y.dtype).sum())

def example():
    y = torch.tensor([0,2])
    y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
    print(accuracy(y_hat,y))
    print(accuracy(y_hat,y) / len(y))
    print(y_hat[[0,1],y])   # 这里是用的花式索引，操作顺序是，第一位的参数用来筛选行，这里筛选的是y_hat的第一第二行的数据，第二个参数是用来指定对应行的对应列，这里y是0，和2，对应的就是（0，0）和（1，2）取值


# 使用Fashion-MNIST数据集来验证
if __name__ == '__main__':
    '''
    d2l.use_svg_display()

    # create_dataset()
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    print(mnist_train[0][0].shape)

    #x,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
    #show_images(x.reshape(18,28,28),2,9,titles=get_fashion_mnist_lables(y))
    #d2l.plt.show()

    batch_size = 256
    # 批量读取大小，每次256个
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=getdataloader_workers())

    timer = d2l.Timer()
    for x,y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
    '''

    batch_size = 256
    train_iter, test_iter = loda_data_fashion_mnist(batch_size)

    # softmax需要的是一个向量，这里输入的是一个1*28*28的三维数据，因此需要将数据拉平到一个一维空间中去，因为输出是10个类别，因此网络输出的维度位10
    num_input = 784
    num_output = 10

    w = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
    b = torch.zeros(num_output,requires_grad=True)
    example()

    print(evaluate_accuracy(net, test_iter))


class Accumulator:
    '''在n个变量上累加'''
    def __init__(self,n):
        self.data = [0.0] * n
    def add(self,*args):
        self.data == [a + float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0] * len(self.data)

    def  __getitem__(self,idx):
        return self.data[idx]




