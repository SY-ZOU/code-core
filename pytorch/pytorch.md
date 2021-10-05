demp0001

## 基础

> **矩阵运算其实就是维度的变化，空间变换的函数**

### 1. 张量

张量是一种包含某种标量类型（比如浮点数和整型数等）的 n 维数据结构。我们可以将张量看作是由一些数据构成的，还有一些**元数据描述了张量的大小、所包含的元素的类型（dtype）、张量所在的设备。**

<img src="https://pic3.zhimg.com/80/v2-7ee4489394b8e34d19009e40623382f2_1440w.jpg" width=500/>

另外还有一个你可能没那么熟悉的元数据：步幅（stride）。stride 实际上是 PyTorch 最别致的特征之一，所以值得稍微多讨论它一些。

<img src="https://pic2.zhimg.com/80/v2-d7c311ed86d5171c35d74ec26dbe0d8d_1440w.jpg" width=500/>

张量一个数学概念。但要在我们的计算机中表示它，我们必须为它们定义某种物理表示方法。最常用的表示方法是在内存中相邻地放置张量的每个元素（这也是术语「contiguous（邻接）」的来源），即将每一行写出到内存，如上所示。在上面的案例中，我已经指定该张量包含 32 位的整型数，这样你可以看到每一个整型数都位于一个物理地址中，每个地址与相邻地址相距 4 字节。为了记住张量的实际维度，我们必须将规模大小记为额外的元数据。

<img src="https://pic3.zhimg.com/80/v2-561af40cdd43ef4557c4136900f46836_1440w.jpg" width=500/>

假设我想要读取我的逻辑表示中位置张量 [0,1] 的元素。我该如何将这个逻辑位置转译为物理内存中的位置？步幅能让我们做到这一点：要找到一个张量中任意元素的位置，我将每个索引与该维度下各自的步幅相乘，然后将它们全部加到一起。在上图中，我用蓝色表示第一个维度，用红色表示第二个维度，以便你了解该步幅计算中的索引和步幅。进行这个求和后，我得到了 2（零索引的）；实际上，数字 3 正是位于这个邻接数组的起点以下 2 个位置。

<img src="https://pic2.zhimg.com/80/v2-8bb5e51f47b14b6571642c7dd8962029_1440w.jpg" width=500/>

使用高级的索引支持，我只需写出张量 [1, :] 就能得到这一行。重要的是：当我这样做时，不会创建一个新张量；而是会返回一个基于底层数据的不同域段（view）的张量。这意味着，如果我编辑该视角下的这些数据，它就会反映在原始的张量中。在这种情况下，了解如何做到这一点并不算太困难：3 和 4 位于邻接的内存中，我们只需要记录一个说明该（逻辑）张量的数据位于顶部以下 2 个位置的偏移量（offset）。（每个张量都记录一个偏移量，但大多数时候它为零)。

> 如果我取张量的一个域段，我该如何释放底层张量的内存？
>
> 你必须制作该域段的一个副本，由此断开其与原始物理内存的连接。你能做的其它事情实际上并不多。另外，如果你很久之前写过 Java，取一个字符串的子字符串也有类似的问题，因为默认不会制作副本，所以子字符串会保留（可能非常大的字符串）。

<img src="https://pic4.zhimg.com/80/v2-5c729a8c611af9f9060b956c56ee66ff_1440w.jpg" width=500/>

当我们查看物理内存时，可以看到该列的元素不是相邻的：两者之间有一个元素的间隙。步幅在这里就大显神威了：我们不再将一个元素与下一个元素之间的步幅指定为 1，而是将其设定为 2，即跳两步。（顺便一提，这就是其被称为「步幅（stride）」的原因：如果我们将索引看作是在布局上行走，步幅就指定了我们每次迈步时向前多少位置。）









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

### 2. 正向传播、反向传播

#### （1）正向传播

>  正向传播是指对神经网络沿着从输入层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。
>
> 正向传播是指数据从X传入到神经网络，经过各个隐藏层得到最终损失的过程。

#### （2）反向传播

> 反向传播就是根据损失函数L(y^,y)来反方向地计算每一层的z、a、w、b的偏导数（梯度），从而更新参数。

![](https://upload-images.jianshu.io/upload_images/5118838-e7f5f61e3aff398a.png?imageMogr2/auto-orient/strip|imageView2/2/w/1036)

- 正向传播的计算依赖于模型参数的当前值，这些模型参数是在反向传播的梯度计算后通过优化算法迭代的。

- 反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传播计算得到的。

- 在训练深度学习模型时，正向传播和反向传播相互依赖。

### 3. 梯度下降

#### （1）**批量梯度下降法BGD**

批梯度下降每次更新使用了所有的训练数据，最小化损失函数，**如果只有一个极小值，那么批梯度下降是考虑了训练集所有数据，是朝着最小值迭代运动的，**但是缺点是如果样本值很大的话，**更新速度会很慢**。

#### （2）**随机梯度下降法SGD**

随机梯度下降在每次更新的时候，只考虑了一个样本点，这样会大大加快训练数据，也恰好是批梯度下降的缺点，但是有可能由于训练数据的噪声点较多，**那么每一次利用噪声点进行更新的过程中，就不一定是朝着极小值方向更新，但是由于更新多轮，整体方向还是大致朝着极小值方向更新，又提高了速度。**

#### （3）**min-batch 小批量梯度下降法MBGD**

小批量梯度下降法是**为了解决批梯度下降法的训练速度慢，以及随机梯度下降法的准确性综合而来**

### 4. 欠拟合、过拟合

> 由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。模型应关注降低泛化误差。

- 欠拟合指模型无法得到较低的训练误差，过拟合指模型的训练误差远小于它在测试数据集上的误差。

#### （1）模型复杂度

![](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.11_capacity_vs_error.svg)

给定训练数据集，如果模型的复杂度过低，很容易出现欠拟合；

如果模型复杂度过高，很容易出现过拟合。

应对欠拟合和过拟合的一个办法是针对数据集选择合适复杂度的模型。

#### （2）训练数据集大小

如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生。

泛化误差不会随训练数据集里样本数量增加而增大。因此，在计算资源允许的范围之内，我们通常希望训练数据集大一些，特别是在模型复杂度较高时，例如层数较多的深度学习模型。

### 5. 解决过拟合

- 在对模型进行训练时，有可能遇到训练数据不够，即训练数据无法对整个数据的分布进行估计的时候

- 权值学习迭代次数足够多,拟合了训练数据中的噪声和训练样例中没有代表性的特征.

#### （1）正则化

> 正则化方法是指在进行目标函数或代价函数优化时，在目标函数或代价函数后面加上一个正则项

强制地让模型学习到比较小的权值。

如果权值对于噪声过于敏感，把训练样本里的噪声当作了特征，在测试集上的表现就会不好。当权值比较小时，当输入有轻微的改动（噪声）时，结果所受到的影响也比较小，所以惩罚项能在一定程度上防止过拟合。

- L1正则化

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcYiQ7lXJ9AF2DzWKpvRTWepCHsvfVc8ivdTKMjGX.nCpAfemCQTH*55NBitIsp0iJTZ4BS9j.zM6pp7JonJsBOU!/b&bo=MAZgAAAAAAADF2Q!&rf=viewer_4)

其中C0代表原始的代价函数，n是样本的个数，a就是正则项系数，权衡正则项与C0项的比重。后面那一项即为L1正则项。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcYiQ7lXJ9AF2DzWKpvRTWeoEyPgUR0An**I.D13A4oYXDh485Yxiu3QN2HohnkGrzQaB.E8gslC5OUfixzVdZTs!/b&bo=0gYAAQAAAAADF.c!&rf=viewer_4)

从上式可以看出，当w为正时，更新后w会变小；当w为负时，更新后w会变大；因此L1正则项是为了使得那些原先处于零附近的参数w往零移动，使得部分参数为零，从而降低模型的复杂度（模型的复杂度由参数决定），从而防止过拟合，提高模型的泛化能力。 

L1正则中有个问题，便是L1范数在0处不可导，即|w|在0处不可导，因此在w为0时，使用原来的未经正则化的更新方程来对w进行更新，即令sgn(0)=0 

- L2正则化(权重衰减)

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcYiQ7lXJ9AF2DzWKpvRTWepFmQVw2BrrA1.VMHfwtdpSzvR24qmCrhkmDXUb4*IdSQsoire.20UrrjAYOUEgXsc!/b&bo=AgZ6AAAAAAADF0w!&rf=viewer_4)

*L*2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积。

其中超参数a>0。当权重参数均为0时，惩罚项最小。当λ较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0。

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcYiQ7lXJ9AF2DzWKpvRTWerrlDW4LyCoJHzSOM9Zs9e.6t9nhSil0*d1rSlTdWWr6ukH2NNHFBilGy3jHjoc8HQ!/b&bo=fgX.AAAAAAADF7c!&rf=viewer_4)

*L*2范数正则化令权重w1和w2先自乘小于1的数，再减去不含惩罚项的梯度。

#### （2）丢弃dropout

dropout的作用对象是layer，对于某一层中的每个节点，dropout技术使得该节点以一定的概率p不参与到训练的过程中（即前向传导时不参与计算，bp计算时不参与梯度更新）

![](https://images2015.cnblogs.com/blog/722389/201611/722389-20161129205700568-1794787345.gif)

通过dropout，节点之间的耦合度降低了，节点对于其他节点不再那么敏感了，这样就可以促使模型学到更加好的特征；dropout layer层中的每个节点都没有得到充分的训练（因为它们只有一半的出勤率），这样就避免了对于训练样本的过分学习；

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

