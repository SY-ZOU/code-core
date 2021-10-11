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
>  正向传播是指数据从X传入到神经网络，经过各个隐藏层得到最终损失的过程。

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















