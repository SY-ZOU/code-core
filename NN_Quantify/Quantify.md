## 网络模型的问题

### 1. 神经网络爆炸

#### （1）问题

- **神经网络消耗的内存大小成为问题**，尤其是在移动设备上。手机一般配备 4GB 内存,三个模型运行一次通常就要占用1GB内存。

- **内存带宽问题**。模型在每次预测时都会使用模型的权重，这意味着至少 30 FPS。因此，如果部署相对较小的 ResNet-50 网络来分类，运行网络模型就需要 3GB/s 的内存带宽。网络运行时，内存，CPU 和电池会都在飞速消耗。

#### （2）想法

- **设计更有效的网络架构，用相对较小的模型尺寸达到可接受准确度，例如 MobileNet 和 SequeezeNet。**
- **通过压缩、编码等方式减小网络规模。量化是最广泛采用的压缩方法之一。**

这两个方面有时可以共同使用并取得令人瞩目的成果。例如，TensorFlow 量化的MobileNetV1 仅为 4.8MB。

<img src="https://pic3.zhimg.com/80/v2-0eb942478b9beb5cafae957bff92b19e_1440w.jpg" width=500/>

### 2. 提升速度

模型量化在最初的定义里是为了压缩模型参数。

> 比如韩松在ICLR2016上获得best paper的论文，首次提出了参数量化方法。其使用k-mean聚类，让相近的数值聚类到同一个聚类中心，复用同一个数值，从而达到用更少的数值表示更多的数，这是量化操作的一种方案。反过来，从量化数变到原始数的过程，称之为反量化，反量化操作完之后，模型就可以按照原来的方式进行正常的计算。

量化是否一定能加速计算？回答是否定的，许多量化算法都无法带来实质性加速。

> 理论计算峰值：单位时钟周期内能完成的计算个数 X 芯片频率

量化方法可以带来潜在、可落地的速度提升需要的条件：

- 量化数值的计算在部署硬件上的峰值性能更高 。

- 量化算法引入的额外计算（overhead）少 。

已知提速概率较大的量化方法主要有如下三类：

- **二值化**，其可以用简单的**位运算来同时计算大量的数**。

- **线性量化**，又可细分为非对称，对称和ristretto几种。在nvdia gpu，x86和arm平台上，均支持8bit的计算，效率提升从1倍到16倍不等，其中tensor core甚至支持4bit计算，这也是非常有潜力的方向。由于线性量化引入的额外量化/反量化计算都是标准的向量操作，也可以使用SIMD进行加速，带来的额外计算耗时不大。

- **对数量化**，一个比较特殊的量化方法。可以想象一下，两个同底的幂指数进行相乘，那么等价于其指数相加，降低了计算强度。同时加法也被转变为索引计算。但没有看到有在三大平台上实现对数量化的加速库，可能其实现的加速效果不明显。只有一些专用芯片上使用了对数量化。

**首先保证你实现的低比特计算效率超过原先浮点计算**。但低比特计算效率超过浮点计算其实并不容易，因为大家在浮点的计算库上已经做了非常多细致的优化比如winograd，间接卷积等等。

### 3. 降低内存

模型量化还有一个潜在的好处是降低运行时内存占用。

- 降低访存量，存在提升速度的可能 。

- 在同样硬件环境下，同时处理更多视频或者视频路数 。

- 训练更大的模型。

**参数weight只占很少一部分， 大部分内存占用来自激活值activation。如果你做低比特量化只关注卷积的话（很多论文其实也是只量化了卷积），那么是无法带来内存占用降低的。**

用量化降低内存占用，只有一个方式：将尽可能多的layer的激活值都进行量化 。

> 在这个方向上之前商汤的一位实习生李润东也有一个工作，做了除了卷积之外更多层的量化。但是这样做会带来更多的精度损失，这可能也是大家需要关心的。

生产一个量化模型的有以下几种方法，借鉴了ICCV2019上一篇data-free量化论文的定义。

L1：直接将一个浮点参数直接转化成量化数，一般会带来很大的精度损失，但使用上非常简单。

L2：基于数据校准的方案，很多芯片都会提供这样的功能，比如tensorRT，高通，寒武纪等。它需要转模型的时候提供一些真实的计算数据。

L3：基于训练finetune的方案，有很多论文都是使用这种方法，它的好处是可以带来更大的精度提升，缺点是需要修改训练代码，实施周期比较长。

### 4. 落地

阻碍模型量化算法落地的几个问题，核心当然是精度问题。

1、可落地的线性量化方案无法很好的刻画一些分布，比如高斯分布

2、比特数越低，精度损失就越大，实用性就越差

3、任务越难，精度损失越大，比如识别任务，就比分类任务要难非常多

4、小模型会比大模型更难量化

5、某些特定结构，如depthwise，对量化精度十分不友好

6、常见的对部署友好的方法比如merge BN，全量化，都会给精度带来更大的挑战

7、软硬件支持不好也是一个阻碍：不同的硬件支持的低比特指令是不一样的，同样训练得到的低比特模型，无法直接部署在所有硬件上。

8、不同软件库实现的量化方案和细节也不一样，量化细节里包括量化位置、是否支持perchannel、是否混合精度等等。即使硬件支持了量化，但你会发现不是所有硬件可以在低比特上提供更好的速度提升， 造成这个状况的主要原因有多个，一方面是指令集峰值提升可能本身就并不多，而要引入较多的额外计算，另一方面也取决于软件工程师优化指令的水平，同时由于网络结构灵活多样，不一定能在不同网络结构上达到同样好的加速比，需要优化足够多的的corner case才可以解决。

## 量化

### 1. 量化概念

> 量化是指将信号的连续取值近似为有限多个离散值的过程。

量化，即将网络的权值，激活值等从高精度转化成低精度的操作过程，例如将32位浮点数转化成8位整型数int8，同时我们期望转换后的模型准确率与转化前相近。

### 2. 量化优势

- **更小的模型尺寸**。以8bit量化为例，与32bit浮点数相比，我们可以将模型的体积降低为原来的四分之一，这对于模型的存储和更新来说都更有优势。

- **更低的功耗**。移动8bit数据与移动32bit浮点型数据相比，前者比后者高4倍的效率，**而在一定程度上内存的使用量与功耗是成正比的。**

- **更快的计算速度**。相对于浮点数，大多数处理器都支持8bit数据的更快处理，如果是二值量化，则更有优势（卷积过程中的乘加都可以转换为异或操作，并行程度更高，运算速度因此也更快）。

### 3. 量化方式

- **低精度**（Low precision）可能是最通用的概念。常规精度一般使用 FP32（32位浮点，单精度）存储模型权重；低精度则表示 FP16（半精度浮点），INT8（8位的定点整数）等等数值格式。不过目前低精度往往指代 INT8。TensorRT[5]通过最小化原始数据分布和量化后数据分布之间的KL散度来对激活值进行量化，将FP32降为INT8的操作如下：FP32(T) = scale_factor(s) * 8-bit(t) +FP32_bias(b)

- **混合精度**（Mixed precision）在模型中使用 FP32 和 FP16 。FP16 减少了一半的内存大小，但有些参数或操作符必须采用 FP32 格式才能保持准确度。如果您对该主题感兴趣，请查看 `《Mixed-Precision Training of Deep Neural Networks 》`。

- 根据存储一个权重元素所需的位数，还可以包括：
  - 二值神经网络：在运行时权重和激活只取两种值（例如 +1，-1）的神经网络，以及在训练时计算参数的梯度。二值量化模型以Binary Connect和Binarized Neural Networks为代表。
  - 三元权重网络：权重约束为+1,0和-1的神经网络。
  - XNOR网络：过滤器和卷积层的输入是二进制的。XNOR 网络主要使用二进制运算来近似卷积。

- **混合压缩**：其他一些研究更关注如何压缩整个模型而非存储一个元素的位数。`Deep Compression` 是该方向最重要的工作之一，作者将剪枝、量化和编码等技术结合起来，在不显著影响准确性的前提下，将存储需求减少 35x（AlexNet）至 49x（VGG-19）该论文还表明量化卷积层需要 8 位以避免显着的精度损失，而全连接只需要 4 位。

### 3. 工业界工作

工业界最终选择了 INT8 量化—— FP32 在推理（inference）期间被 INT8 取代，而训练（training）仍然是 FP32。TensorRT，TensorFlow，PyTorch，MxNet和许多其他深度学习软件都已启用（或正在启用）量化。

通常，可以根据 FP32 和 INT8 的转换机制对解决方案进行分类。一些框架简单地引入了 `Quantize` 和 `Dequantize` 层，当从卷积或全链接层送入或取出时，它将 FP32 转换为 INT8 或相反。在这种情况下，如图的上半部分所示，模型本身和输入/输出采用 FP32 格式。深度学习框架加载模型，重写网络以插入`Quantize` 和 `Dequantize` 层，并将权重转换为 INT8 格式。

其他一些框架将网**络整体转换为 INT8 格式**，因此在推理期间没有格式转换，如图的下半部分。该方法要求算子（Operator）都支持量化，因为运算符之间的数据流是INT8。对于尚未支持的那些，它可能会回落到 Quantize/Dequantize 方案。下文的讨论都基于这种方式。由于 INT8 使用的比特数只有 FP32 的 25% ，在 INT8 和 FP32 之间转换数值的方法非常重要，因为它会显着影响预测精度。

![](https://pic1.zhimg.com/80/v2-9eeb4d64d6def06445cd1973f5f602e4_1440w.jpg)

### 4. 量化过程

> - 将模型从 FP32 转换为 INT8
>
> - 使用 INT8 进行推理

#### （1）浮点与定点的量化

![](https://pic3.zhimg.com/80/v2-b3a014953508353ca3ac0b91da829d1e_1440w.jpg)

在指令集的内置数据类型中，定点是整数，浮点是二进制格式。

一般来说，指令集层面的定点是连续的，因为它是整数，且两个邻近的可表示数字的间隙是 1 。另一方面，浮点代表实数，其数值间隙由*指数*确定，因而具有非常宽的值域（32 位数值最大整数是 ![[公式]](https://www.zhihu.com/equation?tex=2+%5E+%7B31%7D+-1) ，而浮点值域为 ![[公式]](https://www.zhihu.com/equation?tex=%282++-++2+%5E+%7B+-++23%7D%29%C3%972+%5E+%7B127%7D) ，值越接近零就越准确。一个观察结果是，在给定指数时，浮点在不同范围内拥有数值数量相同数量，如图例如，[1,2) 中浮点值的数量与 [0.5,1)、[2,4]、[4,8] 等相同。



浮点数之间的间隙远超过我的想象，比如512.0f下一个浮点数的间隔大于0.00001，这意味这什么，如果神经网络训练过程中，反向传播中梯度过低，根本不会对参数有任何影响，梯度消失。

![](https://pic2.zhimg.com/80/v2-81744a7aa6c215719b22be72fbd6035d_1440w.jpg)







































## 相关文章

[1] Courbariaux M, Bengio Y, David J, et al. BinaryConnect: training deep neural networks with binary weights during propagations[C]. neural information processing systems, 2015: 3123-3131.

[2] Courbariaux M, Hubara I, Soudry D, et al. Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1[J]. arXiv preprint arXiv:1602.02830, 2016.

[3] Rastegari M , Ordonez V , Redmon J , et al. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks[J]. 2016.

[4] Qin H, Gong R, Liu X, et al. Binary neural networks: A survey[J]. Pattern Recognition, 2020.

[5] 8-bit-inference-with-tensorrt

[6] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[7] Han S, Mao H, Dally W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding[J]. arXiv preprint arXiv:1510.00149, 2015.

[8] Wang K, Liu Z, Lin Y, et al. HAQ: Hardware-Aware Automated Quantization with Mixed Precision[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 8612-8620.

[9] Zhu F, Gong R, Yu F, et al. Towards Unified INT8 Training for Convolutional Neural Network.[J]. arXiv: Learning, 2019.

[10] Zhang D, Yang J, Ye D, et al. LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks[C]. european conference on computer vision, 2018: 373-390.