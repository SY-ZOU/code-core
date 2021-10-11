





























### 2. 自动梯度（autograd）













### 1. 数据集加载

#### （1）自定义数据集

```python
import numpy as np;
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetsDataset(Dataset):
    def __init__(self,filePath):
        txt = np.loadtxt(filePath,delimiter=',',dtype=np.float32)
        self.len = txt.shape[0]
        self.x_data = torch.from_numpy(txt[:,:-1]);
        self.y_data = torch.from_numpy(txt[:,[-1]]);

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

data = DiabetsDataset("/Users/zoushiyu/Desktop/diabetes.csv")
train_data = DataLoader(dataset=data,batch_size=32,shuffle=True,num_workers=3)


for epoch in range(100):
    for index,(x,y) in enumerate(train_data,0):
      ##
```

#### （2）torchvision

```python
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
```



### 6. CUDA

将模型和数据从内存复制到GPU的显存中，就可以使用GPU进行训练，其操作方法是将数据和模型都使用`.cuda()`方法。通过`torch.cuda.is_available()`来返回值判断GPU是否可以使用。通过`torch.cuda.device_count()`获取能够使用的GPU数量。

```python
if torch.cuda.is_available():
	ten1 = ten1.cuda()
	MyModel = MyModel.cuda()
```

直接使用`.cuda()`方法进行数据迁移，默认使用的是0号显卡，可以使用`.cuda(<显卡号数>)`将数据存储在指定的显卡中。Tensor类型，直接使用`.cuda()`方法。Variable类型（使用Variable容器装载数据，可以进行反向传播来实现自动求导），将`Tensor.cuda()`后装载在Variable中和将Tensor装载在Variable中后再使用`.cuda()`是同样的。

通常情况下，多GPU运算分为单机多卡和多机多卡，两者在pytorch上面的实现并不相同，因为多机时，需要多个机器之间的通信协议等设置。

Pytorch 的多 GPU 处理接口是 `torch.nn.DataParallel(module, device_ids)`，其中 `module` 参数是所要执行的模型，而 `device_ids`则是指定并行的 GPU id 列表。

并行处理机制是，首先将模型加载到主 GPU 上，然后再将模型复制到各个指定的从 GPU 中，然后将输入数据按 batch 维度进行划分，具体来说就是每个 GPU 分配到的数据 batch 数量是总输入数据的 batch 除以指定 GPU 个数。每个 GPU 将针对各自的输入数据独立进行 forward 计算，**得到的输出在主GPU上进行汇总，计算loss并反向传播，更新主GPU上的权值，再将更新后的模型参数复制到剩余指定的 GPU 中**，这样就完成了一次迭代计算。所以该接口还要求输入数据的 batch 数量要不小于所指定的 GPU 数量。

<img src="https://upload-images.jianshu.io/upload_images/5214592-5ac8d3400461d0b5.png" width=600/>

pytorch实现单机多卡十分容易，其基本原理就是：加入我们一次性读入一个batch的数据, 其大小为[16, 10, 5]，我们有四张卡可以使用。那么计算过程遵循以下步骤：

1. 假设我们有4个GPU可以用，pytorch先把模型同步放到4个GPU中。
2. 那么首先将数据分为4份，按照次序放置到四个GPU的模型中，每一份大小为[4, 10, 5]；
3. 每个GPU分别进行前项计算过程；
4. 前向过程计算完后，pytorch再从四个GPU中收集计算后的结果假设[4, 10, 5]，然后再按照次序将其拼接起来[16, 10, 5]，计算loss。
   整个过程其实就是 同步模型参数→分别前向计算→计算损失→梯度反传

```python
# 假设就一个数据
data = torch.rand([16, 10, 5])

# 前向计算要求数据都放进GPU0里面
# device = torch.device('cuda:0')
# data = data.to(device)
data = data.cuda()

# 将网络同步到多个GPU中
model_p = torch.nn.DataParalle(model.cuda(), device_ids=[0, 1,  2, 3])
logits = model_p(inputs)
  
# 接下来计算loss
loss = crit(logits, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

















## 回归

### 线性回归

#### （1）损失函数

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcUm2nia.t56PXtXsHxCtLGkeFWdYbU5IammP1Dovmw.Ld.wWB03Y9fpZBJ4V1FST4CYFLyKs.5YAhInwa.KNS14!/b&bo=egKIAAAAAAADF8I!&rf=viewer_4)

通常，我们用训练数据集中所有样本误差的平均来衡量模型预测的质量，即

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcUm2nia.t56PXtXsHxCtLGkd36**WCGDNXCai0ixQ2Xtgh14xqd7MYrwk*Rw5YhnbKXptr8jO1tpZuxS3jgnSEI!/b&bo=DAWKAAAAAAADF7E!&rf=viewer_4)

在模型训练中，我们希望找出一组模型参数，记为w1\*，w2\*，b\*，来使训练样本平均损失最小

#### （2）优化算法

小批量随机梯度下降（mini-batch stochastic gradient descent）

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcUm2nia.t56PXtXsHxCtLGmSeGTDPCkmZmlklU9x2JC.cp1g57wIkRyGieLOeVlQZ47vAJ*mLUhWvrYaUeI9TDw!/b&bo=vgWWAQAAAAADFx4!&rf=viewer_4)

#### （3）代码

```python
num_inputs = 2  #特征项数
num_examples = 1000  #样本数
features = torch.randn(num_examples,num_inputs,dtype=torch.float32) #样本
true_w = [2.3,7]
true_b = 4.5
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


#3.定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1) #输入输出
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)

#4。初始化参数
from torch.nn import init
#们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

#5。定义损失函数
loss = nn.MSELoss()

#6。定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

#7。训练模型
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print(true_w, net.linear.weight)
print(true_b, net.linear.bias)
```

## 分类

###  二分类（Logistic回归）

> 逻辑回归（Logistic Regression）是一种用于解决二分类（0 or 1）问题的机器学习方法，用于估计某种事物的可能性。比如某用户购买某商品的可能性，某病人患有某种疾病的可能性，以及某广告被用户点击的可能性等。 注意，这里用的是“可能性”，而非数学上的“概率”，logisitc回归的结果并非数学定义中的概率值，不可以直接当做概率值来用。该结果往往用于和其他特征值加权求和，而非直接相乘。

逻辑回归（Logistic Regression）与线性回归（Linear Regression）都是一种广义线性模型（generalized linear model）。逻辑回归假设因变量 y 服从伯努利分布，而线性回归假设因变量 y 服从高斯分布。 因此与线性回归有很多相同之处，去除Sigmoid映射函数的话，逻辑回归算法就是一个线性回归。可以说，逻辑回归是以线性回归为理论支持的，但是逻辑回归通过Sigmoid函数引入了非线性因素，因此可以轻松处理0/1分类问题。

我们要先介绍一下Sigmoid函数，也称为逻辑函数（Logistic function）：

- ![[公式]](https://www.zhihu.com/equation?tex=g%28z%29%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)

其函数曲线如下：



![img](https://pic2.zhimg.com/80/v2-1562a80cf766ecfe77155fa84931e745_1440w.png)



从上图可以看到sigmoid函数是一个s形的曲线，它的取值在[0, 1]之间，在远离0的地方函数的值会很快接近0或者1。它的这个特性对于解决二分类问题十分重要

一个机器学习的模型，实际上是把决策函数限定在某一组条件下，这组限定条件就决定了模型的假设空间。当然，我们还希望这组限定条件简单而合理。而逻辑回归模型所做的假设是：

- ![[公式]](https://www.zhihu.com/equation?tex=P%28y%3D1%7Cx%3B%5Ctheta%29+%3Dg%28%5Ctheta%5ETx%29%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5ETx%7D%7D)

这个函数的意思就是在给定 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的条件下 ![[公式]](https://www.zhihu.com/equation?tex=y%3D1) 的概率。

这里 ![[公式]](https://www.zhihu.com/equation?tex=g%28h%29) 就是我们上面提到的sigmoid函数，与之相对应的决策函数为：

- ![[公式]](https://www.zhihu.com/equation?tex=y%5E%2A+%3D+1%2C+if+P%28y%3D1%7Cx%29%3E0.5)

选择0.5作为阈值是一个一般的做法，实际应用时特定的情况可以选择不同阈值，如果对正例的判别准确性要求高，可以选择阈值大一些，对正例的召回要求高，则可以选择阈值小一些。

#### （1）Logistic回归计算图

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5mYKQCVkYV0vdrzqLFlQghnHIu.VY9W.LZhuzSHgnwNlPWSeYSQ78s8aWGkx7nuNZVKbarThgM4YYZHqMBd6ke4!/b&bo=bAhsAwAAAAADByk!&rf=viewer_4)

#### （2）损失函数

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcReE0kuxCAEazuBS0kdsDm9zx8omFv4A*uTqavs6TUh8xXwAkD.w65q0lbFWjpiOF0BeOvHMKdkOFHO8BsgMjgs!/b&bo=0gZmAwAAAAADN6M!&rf=viewer_4)

**在使用BCE的时候，要在前一层加入sigmoid层，分散到1，0**。

当y=1时，y_hat要最大，loss才小；y=-1时，y_hat要最小，loss才小。y_hat在0-1。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcReE0kuxCAEazuBS0kdsDm8SwLh9b8hMPTYTCMpEa12ApBVAab2B.nK3JY.othaylrAvdfHeb6c9Tu9xRmKGyHI!/b&bo=XAgMAwAAAAADRzk!&rf=viewer_4)

**BCELoss 是CrossEntropyLoss的一个特例，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类。**

#### （3）代码

```python
import torch
import torchvision
import numpy as np
import sys
import config as cf
from collections import OrderedDict
from torch import nn as nn
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

txt = np.loadtxt("/Users/zoushiyu/Desktop/diabetes.csv",delimiter=',',dtype=np.float32);
x_data = torch.from_numpy(txt[:,:-1]);
y_data = torch.from_numpy(txt[:,[-1]]);#不能用-1，用[-1]表示这是一个矩阵，用-1就是一个向量了[759,1]和[759]区别

class DiabetsDataset(Dataset):
    def __init__(self,filePath):
        txt = np.loadtxt(filePath,delimiter=',',dtype=np.float32)
        self.len = txt.shape[0]
        self.x_data = torch.from_numpy(txt[:,:-1]);
        self.y_data = torch.from_numpy(txt[:,[-1]]);

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

data = DiabetsDataset("/Users/zoushiyu/Desktop/diabetes.csv")
train_data = DataLoader(dataset=data,batch_size=32,shuffle=True,num_workers=3)




#模型
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__();
        self.Linear1 = nn.Linear(8, 7);
        self.Linear2 = nn.Linear(7, 3);
        self.Linear3 = nn.Linear(3, 1);
        self.ReLU = nn.ReLU();
        self.Sigmoid = nn.Sigmoid();
    def forward(self,x):
        x = self.ReLU(self.Linear1(x));
        x = self.ReLU(self.Linear2(x));
        x = self.Sigmoid(self.Linear3(x));
        return x;
model = Model()

#构造损失，优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


#训练
for epoch in range(10000):
    for index, (x, y) in enumerate(train_data, 0):
        #forward
        y_hat = model(x);
        loss = criterion(y_hat,y);

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        mask = y_hat.ge(0.5).float()  # 以0.5为阈值进行分类
        correct = (mask == y_data).sum()  # 计算正确预测的样本个数
        acc = correct.item() / x_data.size(0)  # 计算精度
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(loss.item()))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度

```

### 多分类（softmax回归）

softmax回归跟线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，softmax回归的输出值个数等于标签里的类别数。

softmax回归同线性回归一样，也是一个单层神经网络。由于每个输出o1,o2,o3的计算都要依赖于所有的输入x1,x2,x3,x4，softmax回归的输出层也是一个全连接层。

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.4_softmaxreg.svg)

softmax运算符（softmax operator）通过下式将输出值变换成值为正且和为1的概率分布：

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5tpVi4P38TMkswUyC.K5ZpWEa*BzO2ZZCXGmEAd7rx99NT27rEaqAGcpWMnnjXfBEjh64rMV6ty6V7EVoKFIMcA!/b&bo=7gNYAQAAAAADB5Y!&rf=viewer_4)

#### （1）损失函数

交叉熵（cross entropy）只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcc7zNveRovNFf2l.ytULlqx7iIxVoXH9YCW17c6ZQ96v8gLh0B*xkHuUpTLX8KBFqqlil72CroxXgmq95rAc4Y0!/b&bo=ggJ4AAAAAAADF8o!&rf=viewer_4)

交叉熵损失函数定义为

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcc7zNveRovNFf2l.ytULlqxRtXndDSOIFsO1Dvx4tqXBFFXAGIcQz7209Zrd*3wlXYd6I4GK1J3Fwo6ugxpHxyM!/b&bo=EAKEAAAAAAADF6Q!&rf=viewer_4)

#### （2）代码

```python
num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 30
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
for epoch in range(1, num_epochs + 1):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:

        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch , train_l_sum / n, train_acc_sum / n, test_acc))
    #print('epoch %d, loss: %f' % (epoch, l.item()))
```

## 多层感知机

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.8_mlp.svg)

### 激活函数

> 如果只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。

- ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素x，该函数定义为

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5t3y1r1Gwq0g.ufHFhz0DR7NNtm6K.hQCOx8QKtmDmwBJITJX4DuZfywaBEAMHEH3kI8SHSpw7XHg.a1GDx291o!/b&bo=zAZOAAAAAAADB6Y!&rf=viewer_4)

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.8_relu.png)

- sigmoid函数可以将元素的值变换到0和1之间：

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcUm2nia.t56PXtXsHxCtLGlSLGzP5aWjcaD3*w3lFDkYP6I5PUf7oG1B6LFtieC9qrKVnsHNG0M.BLcesCkkvqI!/b&bo=FgdiAAAAAAADF0E!&rf=viewer_4)

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.8_sigmoid.png)

- tanh（双曲正切）函数可以将元素的值变换到-1和1之间：

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcUm2nia.t56PXtXsHxCtLGmR3hi7v*K0aFLYEFRyJ*Lkwjeso*MY1JSHm76H7xDEJz1wsXTba9Mw5Dd6Y*uaI3s!/b&bo=pgZ4AAAAAAADF.o!&rf=viewer_4)

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.8_tanh.png)

### 多层感知机

多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。

多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5kRE.buGn6oqLu8HV21KaycBqifVqbMRmA66ZcwNJSY9UyaZ4sg0bpvYTINx2k09ctzwvR9ne.7yUPLAfYgub4U!/b&bo=7AaUAAAAAAADB1w!&rf=viewer_4)

```python
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
batch_size = 30
train_iter, test_iter = cf.load_data_fashion_mnist(batch_size)
num_inputs , num_outputs , num_hiddens = 784, 10, 430
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)
)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 30
for epoch in range(1, num_epochs + 1):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = cf.evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch , train_l_sum / n, train_acc_sum / n, test_acc))
    #print('epoch %d, loss: %f' % (epoch, l.item()))
```

