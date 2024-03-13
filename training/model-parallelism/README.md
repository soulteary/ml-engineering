# 模型并行性

## 并行概述
在现代机器学习中，各种并行技术被用来：

1. 克服GPU内存限制。例如：
   - 适应非常大的模型——例如，t5-11b是仅模型参数就有45GB的规模
   - 处理非常长的序列——例如，
2. 显著加快训练速度——将原本需要一年的训练时间缩短到几个小时

我们将深入探讨各种1D平行化技术的优缺点，然后看看它们如何结合成2D和3D平行化，以实现更快的培训和支持更大的模型。我们还将介绍其他强大的替代方法。

虽然主要概念可能适用于任何其他框架，但本文的重点是基于PyTorch的实现。

为了支持比加速器内存大得多的模型的训练和推理，使用了两种主要的方法：
1. 三维并行性 —— 网络效率非常高，但在使它正常工作所需的建模代码修改方面可能会非常侵入式，并且需要大量的工作来确保正确运行
2. ZeRO并行性 —— 网络效率不是很高，但它几乎不需要对建模代码进行更改，而且使其工作的难度也非常低

## 可扩展性概念
以下是对本文件中将详细描述的主要概念的简要说明。

1. **数据并行性**（DP）—— 相同的设置被复制多次，每个设置都接收数据的切片。处理是在并行中完成的，并且在每次训练步骤结束时所有设置都会同步。
2. **张量并行性**（TP）—— 每个张量都被分割成多个块，因此整个张量不再驻留在单个GPU上，而是每个张量的分片（shard）分别位于其指定的GPU上。在处理过程中，每个分片都在不同的GPU上单独且并行地处理，结果在步骤结束时同步。这被称为水平并行化，因为分割发生在水平级别。
3. **管道并行性**（PP）—— 模型在垂直方向（层级）跨多GPU分割，以便一个或几个层的模型部分放在同一个GPU上。每个GPU并行处理不同阶段的管道，并在一个小批次的特定片段上工作。
4. **零冗余优化器**（ZeRO）—— 也执行类似于TP的张量分片操作，只是在整个前向传播或后向传播计算之前，完整的张量会被重建，因此无需修改模型即可实现这一点。此外，它还支持各种卸载技术以补偿有限的GPU内存。零散化的DDP是ZeRO基础概念的另一个名称，因为它用于各种其他ZeRO实现的上下文中。
5. **序列并行性** —— 对长输入序列的训练需要大量的GPU内存。这种技术将单一序列的处理分割到多个GPU上。

关于这些并行化方法的简明解释可以在论文[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953)的引言部分找到。

## 数据并行性
### DDP
大多数用户使用两个GPU就已经可以享受到数据并行（DP）和分布式数据并行（DDP）带来的训练速度提升，这两者几乎是微不足道的。这是Pytorch的内置特性。

有关详细信息，请参阅[DistributedDataParallel文档](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

### ZeRO数据并行性
由ZeRO驱动的数据并行性（ZeRO-DP）如图所示，来自此[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-of-models-with-over-100-billion-parameters/)中的图表：
![DeepSpeed-Image-1](images/parallelism-zero.png)

理解起来可能有些困难，但实际上这个概念很简单。这只是通常的数据并行（DP），只不过，而不是复制完整的模型参数、梯度和优化器状态，每个GPU只存储一部分。然后在运行时，当完整的一层参数需要在给定的层中使用时，所有GPU会同步以共享缺失的部分——这就是全部内容。

考虑这样一个简单的模型，有三个层，每层有三个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
层La具有权重a0、a1和a2。

如果我们有3个GPU，Sharded DDP（即ZeRO-DP）会将模型分成3个GPU如下：
```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

现在，每个GPU都将收到未修改的标准批次作为输入。

首先，输入到达层La。

让我们专注于GPU0：x0需要a0、a1和a2参数来进行它的正向路径，但是GPU0只有a0；它会从GPU1请求a1，从GPU2请求a2，从而在其他GPU的帮助下重新构建完整的张量。

同时，GPU1接收到mini-batch x1，它只拥有a1，但需要a0和a2参数；所以它也从GPU0和GPU2获取那些参数。

同样的事情发生在GPU2，它获得了输入x2，并从GPU0和GPU1获取了必要的参数。

所有3个GPU都通过同步重建的全局张量进行了前向传递。

一旦计算完成，不再需要的中间值就会被丢弃——它们只在计算期间有用。重构过程是通过预取高效执行的。

然后整个过程重复应用于层Lb，然后是Lc，向前推进，最后反向传播Lc -> Lb -> La。

对我来说，这听起来像是一个高效的背包重量分配策略：

1. 人物A携带帐篷
2. 人物B携带炉灶
3. 人物C携带斧头

现在，每天晚上他们都会分享彼此所拥有的东西并与其他人交换他们缺少的东西，第二天早上他们会打包各自的装备继续前进。这就是Sharded DDP / Zero DP。

相比之下，每个人都自己携带自己的帐篷、炉灶和斧头的简单策略将会更加低效。这就是Pytorch中的数据并行（DP和DDP）的工作方式。

在阅读文献时，您可能会遇到以下同义词：分片的、分区化的。

如果您仔细观察ZeRO的分区模型权重的方式，它看起来与稍后讨论的TP非常相似。这是因为它不仅分区/分片每一层的权重，而且还与其他垂直模型并行化相比，它在垂直方向上的分割更为精细。

ZeRO-DP阶段1+2+3的实现：
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch](https://pytorch.org/docs/stable/fsdp.html) （最初在[FairScale](https://github.com/facebookresearch/fairscale/)中实施，后来合并到了Pytorch的核心）

Deepspeed ZeRO集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [Determined.AI](https://docs.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html)

FSDP集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main/en/fsdp)
- [Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)

重要论文：

DeepSpeed ZeRO：
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)

PyTorch FSDP：
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)

主要DeepSpeed ZeRO资源：
- [项目的GitHub仓库](https://github.com/microsoft/deepspeed)
- [入门指南](https://www.deepspeed.ai/getting-started/)
- [API文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客帖子](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### ZeRO的多副本版本
默认情况下，ZeRO使用所有GPU创建单个模型副本——这意味着模型被分割并分布到所有的gpus上。这导致了一些限制，比如：

1. 全球批量大小缺乏灵活性——它总是 `total_gpus * micro_batch_size` 的函数，这在大型集群上可能导致非常大的全球批量大小，这可能不利于有效的收敛。当然，可以通过使用极小的微型批次大小来保持全局批次大小在一个可控范围内，但这会导致每个GPU上的矩阵变小，从而降低计算效率。
2. 更快的内节点网络的好处没有被充分利用，因为较慢的外部节点网络定义了通信的整体速度。

[ZeRO++](https://arxiv.org/abs/2306.10209)解决了第二个局限性，引入了层次权重分区（hpZ）的概念。在这种方法中，每个模型副本被限制在一个节点内，而不是像传统ZeRO那样分布在所有gpus上。这增加了内存的使用，因为在每个节点上有更多的模型副本，但现在两次全收集操作（用于聚集和再分发梯度）可以通过快得多的内节点连接执行。

第一个限制并没有完全解决，因为总体的全球批次大小仍然相同，但由于额外的内存压力很可能会限制可能的微型批次大小，这应该会提高系统的吞吐量。

PyTorch FSDP已经实现了这一功能，称为[shardingStrategy.HYBRID_SHARD](https://pytorch.org/docs/stable/fsdp.html)。

论文：

- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)

#### ZeRO的变化形式
已发表的研究论文中提出了对ZeRO协议的改进：

- [MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud](https://arxiv.org/abs/2205.00119) (2022)
- [AMSP: Super-Scaling LLM Training via Advanced Model States Partitioning](https://arxiv.org/abs/2311.00257) (2023)

## 管道并行性方法
### 原始模型并行性（垂直）
原始模型并行性（MP）是将模型层组跨多GPU垂直分割的地方。机制相对简单——只需将所需层切换为指定的设备，这样当数据进入和流出这些层时，数据就会自动在这些层之间移动。

我们称之为垂直MP，是因为如果你还记得大多数模型的绘制方式，我们可以沿着垂直方向切割模型。例如，对于下面显示的具有8层的模型：
```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
我们将其分为两部分，将层0-3放置在GPU0上，并将层4-7放置在GPU1上。

现在，当数据流经层0到1再到2再到3时，它是作为一个正常的模型处理的。但是，当数据需要从层3传输到层4时，它需要从GPU0复制到GPU1，这引入了通信开销。如果参与的GPU位于同一计算节点（例如，同一物理机），那么这种复制是非常快速的，但如果GPU位于不同的计算节点（例如，多个机器），那么通信开销可能会显著增加。

然后，层4到5到6到7就像一个普通的模型一样运作，直到第7层完成，我们经常需要将数据发送回层0，或者相反，将标签发送到最后一个层。这时损失可以被计算出来，优化器可以做它的工作。

问题：
- 主要的缺陷，也是为什么它被称为“原始”MP的原因，是除了一个GPU之外的所有GPU在任何一个时刻都是空闲的。因此，如果有4个GPU可用，实际上这与在单个GPU上使用4倍的内存容量没有什么区别，忽略了其余的硬件。此外，还有数据拷贝的开销。因此，即使你有4个6GB的卡，你也可以用一个24GB的卡更好地完成训练，因为后者没有数据拷贝的开销。
- 共享嵌入可能需要来回地在GPU之间复制。

### 管道并行性
管道并行性（PP）基本上与原始MP相同，但它解决了GPU闲置的问题，通过将一批数据分割成微观批次（Micro Batches），并人为地创建了一个流水线，允许不同的GPU在不同阶段并行地参与计算过程。

下面的图示来自[GPipe论文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)，展示了原始MP的上半部分和PP的下半部分：
![mp-pp](images/parallelism-gpipe-bubble.png)

很容易看出，在底部图中，PP减少了空闲区域的数量，即所谓的气泡。

两者都显示了4个GPU的并行化，即并行的程度为4。因此，在上面的例子中，如果我们有一个全局批次大小为1024，我们将把它分成4个迷你批次，每个迷你批次的大小为256（1024/4）。如果我们进一步将`chunks`设置为32，我们会得到一个微批次大小为8（256/32）。每个管道阶段一次只处理一个微批次。

要计算全局批次大小，我们需要将mbs乘以chunks乘以dp_degree：`mbs*chunks*dp_degree=1024`。

让我们回到上面的图示。

有了`chunks=1`，我们就得到了原始MP，这是非常低效的。随着`chunks`值的增大，气泡（空闲时间）可以进一步减少，但过大会导致微批次尺寸减小，这可能也会影响效率。因此，需要通过实验来确定最佳值，以最小化气泡的大小。

选择调度方案至关重要，以下是按发明顺序出现的常见调度方案：

- 顺序[Gpipe: Efficient training of giant neural networks using pipeline parallelism](https://arxiv.org/abs/1811.06965)
- 交错1F1B[Pipedream: Fast and efficient pipeline parallel dnn training](https://arxiv.org/abs/1806.03377)
- 循环，深度优先[Efficient large-scale language model training on gpu clusters using Megatron-LM](https://arxiv.org/abs/2104.04473)
- 广度优先[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953)

这里有一个交错的管道执行示例：
![interleaved-pipeline-execution](images/parallelism-sagemaker-interleaved-pipeline.png)

在这里，气泡（空闲时间）进一步减少，优先考虑反向传递。

传统的Pipeline API解决方案：
1. 传统Pipeline API解决方案：
   - Megatron-LM
   - DeepSpeed
   - PyTorch

2. 现代解决方案：
   - PiPPy
   - Varuna
   - SageMaker

传统Pipeline API解决方案存在的问题：
- 为了利用Pipeline的优势，模型需要进行相当重的修改，因为Pipeline要求将模块重新组织为一个`nn.Sequential`序列，这可能需要对模型的设计进行调整。
- 目前Pipeline API的功能非常有限。如果在Pipeline的第一阶段中有一些Python变量被传递进去，那么你可能需要找到一种绕过它的方法。当前，Pipeline接口要求输入要么是一个单一的Tensor，要么是一个包含Tensors的可迭代对象，并且这些张量必须将批次大小作为第一维，因为Pipeline需要将微型批次分割成微批次。可能的改进正在此处讨论：https://github.com/pytorch/pytorch/pull/50693
- 在管道阶段之间的控制流中存在条件是不可能的——例如，编码器-解码器模型如T5需要特殊的处理来处理条件编码阶段。
- 由于数据在每个阶段都需要在GPU之间复制，因此需要安排每个层，以便前一层的输出成为下一层的输入。

我还没有尝试过Varuna和SageMaker，但他们的论文报告说他们已经克服了上述提到的一些问题，并且他们对用户的模型需要进行的改动较小。

实现：
- [Pytorch](https://pytorch.org/docs/stable/pipeline.html) （初始支持在pytorch-1.8中，并在1.9和1.10中逐步增强）。一些[例子](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- [FairScale](https://fairscale.readthedocs.io/en/latest/tutorials/pipe.html)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 有内部实现，没有公开的API。
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972) ——这是一个专有的解决方案，只能在AWS上使用。
- [OSLO](https://github.com/eleutherAI/Oslo) ——基于Transformers的Tensor Parallelism的实现。
- [PiPPy: Pipeline Parallelism for PyTorch](https://github.com/pytorch/pippy) ——通过`torch.fx`自动进行Pipeline Parallelism。
- [nanotron](https://github.com/huggingface/nanotron)

## DP+PP
下面的图示来自DeepSpeed的[pipeline教程](https://www.deepspeed.ai/tutorials/pipeline/)，演示了如何组合DP和PP。
![dp-pp-2d](images/parallelism-zero-dp-pp.png)

重要的是要注意DP rank 0并不知道GPU2的存在，而DP rank 1不知道GPU3的存在。对DP来说，只有GPU 0和1可见，其中GPU0“秘密地”将一些负载卸载到GPU2使用PP。同样，GPU1也在不知情的情况下寻求GPU3的帮助。

由于每个维度至少需要2个GPU，这里你需要至少4个GPU。

实现：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)

## DP+PP+TP
为了获得更高效的训练，可以使用3D并行性，其中PP与TP相结合。这可以在下面的图示中看到。
![dp-pp-tp-3d](images/parallelism-deepspeed-3d.png)

在这个图示中，重要的是要注意DP rank 0不直接看到GPU2，而DP rank 1不直接看到GPU3。To DP，there are just GPUs 0 and 1 where it feeds data as if there were just 2 GPUs. GPU0 "secretly" offloads some of its load to GPU2 using PP. And GPU1 does the same by enlisting GPU3 to its aid.

由于每个维度至少需要2个GPU，这里你需要至少8个GPU。

实现：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed还包括一个非常可扩展的扩展版的DP，他们称之为ZeRO-DP。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)

## ZeRO DP+PP+TP
DeepSpeed的一个主要特点是ZeRO，这是一种超级可伸缩的扩展版DP。它已经在前面章节中讨论过了。通常，它作为一种独立的功能存在，而不依赖于PP或TP。然而，它可以与PP和TP结合起来。

当ZeRO-DP与PP（可选地加上TP）结合在一起时，它通常只启用ZeRO的第1阶段（优化器分片）。

尽管理论上有可能使用ZeRO第2阶段（梯度分片）与Pipeline Parallelism，但它会对性能产生负面影响。在进行梯度分片之前，需要对每个微批次进行额外的reduce-scatter集体操作，这将带来潜在的重大通信开销。由于Pipeline Parallelism的本质，微批次很小，因此关注的是平衡算术强度（微批次大小）和最小化Pipeline气泡（微批次数量）。因此，这些通信成本将对性能造成损害。

同样，ZeRO第3阶段也不是一个好的选择，原因相同——更多的inter-node通信需求。

由于我们有ZeRO，另一个好处是ZeRO-Offload。由于这是第1阶段，优化器状态可以卸载到CPU。

实现：
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 和 [Megatron-Deepspeed from BigScience](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，这是前者repo的一个分支。
- [OSLO](https://github.com/eleutherAI/Oslo)

重要的论文：

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](
https://arxiv.org/abs/2201.11990)

## 序列并行性
ML任务，如DNA测序，可能需要训练带有非常长序列长度（例如256K）的模型，即使是常规的LLM也可能需要训练于10k及以上的序列长度。

自注意力，Transformer的关键组件之一，随着序列长度的平方增长而消耗内存，因此在达到一定长度后，即使在单GPU上使用批次大小为1，序列的长度也无法容纳在一台GPU上，这就需要进一步的划分。一旦这样做，序列可以是任意长度。

根据Megatron-LM论文[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)中的概念，我们可以将并行化划分为以下几个类别：

### Deepspeed-Ulysses SP
论文：[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)

在Deepspeed-Ulysses的实现中，有两种元素被分片：
1. 多重头注意力的权重被横向分割到参与的GPU上，使得每个GPU拥有几个子头。这是在模型加载或创建时完成的。这有点类似[Tensor Parallelism](#tensor-parallelism)。
2. 在训练过程中，每个输入序列被分割成块，每个块被发送到一个特定的GPU上。这让人想起ZeRO-3的分片，只是输入被分片，而不是权重。

在计算过程中，每个序列块的查询（Q）、键（K）和值（V）投影被聚集到全局QKV上，然后计算注意力输出，最后再次分散到局部序列空间。

![deepspeed-ulysses sp](images/deepspeed-ulysses.png)

[来源](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses)

在图示中：
1. 输入序列N被分割成P个可用的设备。
2. 每个本地N/P序列部分的输入被投影到查询（Q）、键（K）和值（V）嵌入。
3. 接下来，local N/P QKV嵌入通过高度优化的全对全集合体在参与的计算设备之间进行聚合。
4. 然后，注意力计算按照每个头部进行：

![math](images/deepspeed-ulysses-math.png)

5. 最后，另一个全对全集合体将输出上下文张量转换为序列（N/P）并行，以便后续的操作（MLP MatMul、层归一化等）能够在剩余的模块中顺利进行。

示例：假设seqlen=8K，num_heads=128，以及一个具有num_gpus=8的单节点。

1. 每个GPU得到一个1K长的序列段(`8K/8`)。
2. 每个GPU被分配16个子头(`128/8`)。
3. a. 在gpu0上，在`forward`之前，原始序列被聚集回8Ktokens。
   b. 注意力计算针对gpu0拥有的16个子头进行。
同样的逻辑也适用于剩下的7个GPU，每个GPU计算8k注意力的16个子头。

你可以阅读关于通信量大幅减少的详细信息[这里](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses#significant-communication-volume-reduction)。

Deepspeed-Ulysses通过增加GPU的数量来保持一致的通信量，这与消息大小（M）或序列长度线性相关。

### Colossal-AI的SP
论文：[Sequence parallelism: Long sequence training from system perspective](https://arxiv.org/abs/2105.13120)

Colossal-AI的SP实现使用环形自我注意力，这是一种环形通信集体，其中查询投影是局部的，而关键和值投影则被传输以计算全局注意力，从而实现与消息大小（M）线性增长的通信复杂性。

### Megatron-LM的SP
论文：[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

Megatron-LM的SP与它的TP紧密集成。Megatron-LM沿序列维度分割序列，并通过allgather和reduce scatter集体操作聚合QKV投影以进行注意力计算。它的通信量随消息大小（M）线性增长，无论计算设备的数量是多少。

实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Deepspeed](https://github.com/microsoft/DeepSpeed)
- [Colossal-AI](https://colossalai.org/)

## FlexFlow
[FlexFlow](https://github.com/flexflow/FlexFlow)也是一种不同的并行化问题的解决方案。

论文：["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)

它通过Sample-Operator-Attribute-Parameter维度来实现一种类型的4D并行化。

1. Sample = Data Parallelism (样本并行)
2. Operator = Parallelize a single operation into several sub-operations (操作并行)
3. Attribute = Data Parallelism (属性并行)
4. Parameter = Model Parallelism (参数并行)

示例：
* Sample

让我们考虑10个序列长度为512的批次。如果我们在2个设备上并行化它们，我们将得到5个2x512的批次。

* Operator

如果我们执行层归一化，我们先计算std，然后再计算mean，然后我们可以做标准化。操作并行允许我们并行计算std和mean。因此，如果我们并行化它们到2个设备（cuda:0, cuda:1），首先我们将输入数据复制到这两个设备，然后cuda:0计算std，cuda:1计算mean的同时。

* Attribute

我们有10个512长度的批次。如果我们并行化它们到2个设备，10 x 512将成为10 x 2 x 256。

* Parameter

这类似于张量模型并行或原始层式模型并行。

![flex-flow-soap](images/parallelism-flexflow.jpeg)

FlexFlow的重要意义在于它能够利用（1）GPU/TPU/CPU vs.（2）RAM/DRAM vs.（3）fast-intra-connect/slow-inter-connect这样的资源，系统性地优化算法决策，决定在哪里使用哪种并行化策略。

FlexFlow的一个重要特征是它旨在优化固定工作负载的大型DNN并行化，因为动态行为的模型可能偏好不同的并行化策略。

因此，该框架承诺提供非常有吸引力的服务——它会在集群上模拟运行30分钟，并为该特定环境定制最优计划。如果添加/移除/替换任何部件，它将再次运行并重新优化计划。然后你可以开始训练。一个新的设置会有它自己的自定义优化。

## 总结
综上所述，模型并行化是一种复杂的领域，有许多技术和权衡需要考虑。选择正确的策略取决于多种因素，包括模型架构、硬件配置、通信带宽、应用场景等等。在实际应用中，研究人员和工程师通常会根据自己的具体情况进行试验和调优，以找到最适合他们需求的并行化方案。