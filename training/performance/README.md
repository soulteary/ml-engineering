软件调优以获得最佳性能

让模型训练得更快，模型完成训练的时间就越早，这对于抢先发布研究成果以及潜在地节省大量资金都至关重要。

一般来说，最大化吞吐量就是关于运行多个实验并测量结果，然后选择表现最好的那个。

在某些情况下，您的建模团队可能会要求您选择一些对吞吐量有害的超参数，但总体上对整个模型的成功有益。

## 词汇表和概念

- HFU：硬件浮点运算利用率
- MFU：模型浮点运算利用率

### MACs与FLOPs、FLOPS与FLOP/s的区别

本节旨在澄清常见的性能指标定义及其之间的关系。

**MAC与FLOP：**

- 一个FLOP（浮点操作）可以是加法、减法、乘法或除法的任何一种操作。
- 一个MAC（乘积累加）操作是一个乘法后跟一个加法，即：`a * b + c`

因此，1个MAC等于2个FLOP。同样常见的是现代硬件能够在单个时钟周期内执行1个MAC。

请注意，为了计算MACs相对于FLOPs的数量关系，逻辑是相反的，即MACs = 0.5 FLOPs -这有点令人困惑，因为我们刚刚说过1个MAC = 2个FLOP，但它确实有效 - 观察：100个FLOP = 50个MAC，因为每个MAC中有2个FLOP。

此外，虽然1个MAC = 2个FLOP，但反之并不一定成立。也就是说，2个FLOP不一定等于1个MAC。例如，如果将`.5*.6`重复执行100次，它将是100个FLOP，在这里将等于100个MAC，因为在这些示例中只执行了MAC中的乘法部分。

**FLOP与FLOPS与FLOP/s：**

- 1个FLOP（浮点操作）是任何浮点加法、减法、乘法或除法操作。
- 1个FLOPS（每秒浮点操作数）是指在1秒钟内执行的浮点操作数量 - 参见[FLOPS](https://en.wikipedia.org/wiki/FLOPS)。

进一步，您会遇到以下缩写：GFLOPS = 千兆FLOPS，TFLOPS = 太FLOPS等，因为它们更容易被快速理解，而不是150万亿FLOPS这样的数字。

FLOPS的使用存在歧义，因为它有时用于表示总操作量，而在其他时候则用于表示每秒的操作量。后者是最常用的用法，也是本书中使用的定义。

在科学写作中，使用FLOP/s来明确告诉读者这是每秒的操作量更为常见。尽管如此，这种特定的方法很难转换为变量名，因为非法字符需要删除。

在一些地方，您可能还会看到FLOPs，这再次可能是指总量或每秒的操作量，因为大写和小写字母“s”之间的切换很容易发生。

如果定义不明确，尝试搜索上下文可能会有所帮助，这将有助于推断其含义：

- 如果它是数学方程的一部分并且有时间的分母，那么它指的是每秒的操作量。
- 如果讨论速度或性能，通常指的是每秒的操作量。
- 如果谈论完成某项任务所需的计算量，它指的是总的操作量。

### TFLOPS作为性能指标

在开始优化培训设置性能之前，您需要一个可以用来查看是否正在改进的度量标准。您可以测量每迭代秒数、每秒迭代次数或其他类似的计时信息，但是有一个更有用的指标称为TFLOPS。

测量TFLOPS的优势在于，它可以指示您距离硬件制造商报告的理论峰值性能有多近。

在本节中，我将使用BLOOM的培训作为范例。我们使用了80GB的NVIDIA A100 GPU进行混合bf16模式下的培训。让我们看看[A100规格](https://www.nvidia.com/en-us/data-center/a100/)，其中告诉我们：

```
BFLOAT16 Tensor Core 	312 TFLOPS
```

这意味着如果我们仅在巨大的bf16矩阵上运行`matmul`而不涉及从设备到设备的复制或磁盘IO通信，我们应该能够实现大约312 TFLOPS的最大值。

实际上，由于磁盘IO、通信和数据在GPU内存与计算单元之间传输的开销，我们可以期望远低于这个数值。对于A100在2022年的实际可持续吞吐量超过50%（即155 TFLOPS左右）是非常了不起的。

脚注：在2023年，发明的[闪存注意力](https://github.com/Dao-AILab/flash-attention)和其他技术已经将这一比例提高到了超过50%。

当我们第一次开始调整时，我们的TFLOPS不到100，几周后当我们启动培训时，我们已经设法将其提升至150 TFLOPS。

重要的是要注意这里，我们知道我们不能通过太多方式来推动它，而且我们知道没有更多理由继续对其进行优化甚至更多。

因此，在进行大规模模型培训准备时的一般经验法则是在给定的加速器上预期可以达到的最佳TFLOPS水平附近进行优化，一旦接近该水平就停止优化并开始培训。

脚注：对于80GB A100s在2022年，那是155，在2023年，它已被推高到约180 TFLOPS。

脚注：当启用梯度检查点时，计算TFLOPS需要考虑额外的计算成本。通常，这相当于额外的前向路径的成本，但在最近的研究中发现了一些方法可以减少部分重新计算。

对于解码器转换器模型，以下是一种估算公式的简化形式，它略微低估了实际的TFLOPS：

TFLOPS：`model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

因子4适用于激活/梯度检查点的情况，否则它将为3。对于100B+模型，激活检查点几乎总是打开的。

```
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=127; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```
(ng = 总共gpus, ms = 模型大小 in B, gbs = 全球批量大小, sp = 吞吐量 in 秒)

这里的`bash`环境变量的相同公式如下所示，它分解了GBS为`MBS*DP*GAS`（GAS在这种情况下对应于`pp_chunks`，这是管道中的块数量，但正常情况下GAS只是代表梯度累积步骤）：
```
echo "($MSIZE*4*2*SEQLEN*$MICRO_BATCH_SIZE*$DP_SIZE*$GAS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
```

确切的公式可以在《高效的大规模语言模型训练》论文的第5.1节的方程式3中找到。您可以在此处找到相应的代码[提交](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251)。

脚注：对于推理只有：`24Bsh^2 + 4Bs^2h` 浮点运算 per 层。

### MFU与HFU：

模型浮点运算利用率（MFU）和硬件浮点运算利用率（HFU）估计硬件在模型的前向和反向传递期间（包括同步网络开销和可能的DataLoader I/O）的实际利用情况。

HFU衡量实际使用的浮点运算。例如，[梯度检查点/激活重算](#gradient-checkpointing)功能重复了前向传递的部分内容第二次，因此事实上使用了更多的浮点运算。相比之下，MFU忽略实现细节，仅根据理论计算需求进行评估，因此不太准确。

[减少大型转换器模型中的激活重算](https://arxiv.org/abs/2205.05198)是一篇值得阅读的论文，介绍了这些概念。


对于Bloom的培训，Megatron-LM发布了以下统计数据：

| 模型尺寸 | 模型FLOPs利用率 | 硬件FLOPs利用率 |
| :---: | :---: | :---: |
| 22B | 41.5% | 43.7% |
| 175B | 51.4% | 52.8% |
| 530B | 56.0% | 57.0% |
| 1T | 56.3% | 57.0% |

最近的H100+A100 MFU/HFU数字已发表[此处](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking#mfu-and-hfu)。

## 如何改善速度并节省内存

拥有更多的GPU内存可用于更大的批处理大小（BS），这使得GPU更加高效地进行计算，从而加快任务的完成速度。

当然，这部分对于即使BS=1时出现GPU OOM的情况也尤为重要，这时您不想租用/购买更多硬件。

下面是对可以帮助提高速度或节省内存的方法的概述：

| 方法 | 速度 | 记忆 |
| :---: | :---: | :---: |
| 梯度积累 | 是 | 是 |
| 梯度检查点 | 是 | 是 |
| 混合精度训练 | 是 | 否 |
| 批次大小 | 是 | 是 |
| 优化器选择 | 是 | 是 |
| Dataloader | 是 | 否 |
| Deepspeed零 | 否 | 是 |
| 闪光注意 | 是 | 是 |

### 模型操作解剖

变压器架构包含三个主要操作组，按计算强度分组如下：

1. **张量收缩**

    线性层和多头自注意组件都执行批处理的**矩阵-矩阵乘积**。这些操作是训练变压器中最密集的计算部分。

2. **统计归一化**

    软最大和层归一化比张量收缩更轻量级，涉及一个或多个**降维操作**，其结果随后通过映射应用。

3. **元素操作**

    其余的操作包括偏差、dropout、激活和残差连接。这些都是最轻量级的操作。

了解这一点有助于分析性能瓶颈。

此总结源自[数据移动就是你所需要的：2020年转换器优化案例研究](https://arxiv.org/abs/2007.00072)

### 模型内存使用解剖

我们看到训练模型除了将模型放在GPU上之外还消耗了大量内存。这是因为有许多组件在训练过程中使用GPU内存。这些组件驻留在GPU内存中的如下：

1. 模型权重
2. 优化器状态
3. 梯度
4. 正向激活保存用于梯度计算
5. 临时缓冲区
6. 与特定功能相关的内存

典型模型在混合精度和AdamW优化器的训练中需要18字节/模型参数加上激活内存和临时内存。

让我们详细了解一下。

**模型权重：**

- 4字节 * 参数数量用于fp32训练
- 6字节 * 参数数量用于混合精度训练（保持模型在fp32和fp16/bf16中的一个内存中）

**优化器状态：**

- 8字节 * 参数数量的正常AdamW（维护两个状态）
- 4字节 * 参数数量的bf16混合精度训练中的AdamW（见[这项工作](https://github.com/huggingface/transformers/pull/21312))
- 4字节 * 参数数量的其他优化器如SGD带动量（仅维护1个状态）或狮子或Adafactor（以及其他） （Adafactor使用一些额外的内存）
- 2字节 * 参数数量的8位AdamW量化优化器，如[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

**梯度：**

- 4字节 * 参数数量的fp32或半精度混合精度训练中的参数
- 2字节 * 参数数量的非混合半精度或半精度混合精度训练中的参数

**正向激活：**

- 大小取决于许多因素，关键因素包括序列长度、隐藏大小和批处理大小。

有输入和输出被向前和向后函数传递和返回，以及正向激活保存用于梯度计算。

**临时内存：**

此外，还有各种临时变量可能在计算过程中短暂分配，并在计算完成后释放。然而，这些可能导致OOM，因此在编码时战略性地思考此类临时变量并及时显式释放它们非常重要。

**特定功能的记忆：**

然后，您的软件可能有特殊的内存需求。例如，在使用beam search生成文本时，软件需要维护多个输入和输出的副本。

对于**推理**，数学非常类似于训练，但没有优化器和梯度的内存需求。对于模型权重，只有一个倍增器用于模型参数的数量：

- 6字节在混合精度（4+2）
- 4字节在fp32
- 2字节在半精度
- 1字节在量化int8精度

另一个很好的资源是[EleutherAI Cookbook](https://github.com/EleutherAI/cookbook)，它包含了基于配置和设置的[计算脚本](https://github.com/EleutherAI/cookbook/tree/main/calc)，这些脚本可以输出理论上的内存开销。

Alexander Smirnov提供了非常有用的[GPU VRAM Estimator](https://vram.asmirnov.xyz/)，以及[有关其工作的说明](https://asmirnov.xyz/vram)。

### 额外的GPU内存使用

除了上述描述的内存使用外，还有一些GPU内存消费者，因此您不会得到完整的可用内存用于模型使用。

#### 预加载的CUDA内核内存使用

当PyTorch首次使用CUDA时，它会预先占用0.5-2GB的GPU内存，减少了GPU的总可用内存。

CUDA内核预加载所需的内存量因GPU而异，并且在不同的PyTorch版本中也不同。让我们分配一个4字节的张量到cuda并检查有多少GPU内存被预先占用。

使用`pytorch==1.10.2`：
```
$ CUDA_MODULE_LOADING=EAGER python -c "import torch; x=torch.ones(1).cuda(); free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info()); \
used=total-free; print(f'pt={torch.__version__}: {used=:0.2f}GB, {free=:0.2f}GB, {total=:0.2f}GB')"
pt=1.10.2: used=1.78GB, free=77.43GB, total=79.21GB
```

使用`pytorch==1.13.1`：
```
$ CUDA_MODULE_LOADING=EAGER python -c "import torch; x=torch.ones(1).cuda(); free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info()); \
used=total-free; print(f'pt={torch.__version__}: {used=:0.2f}GB, {free=:0.2f}GB, {total=:0.2f}GB')"
pt=1.13.1: used=0.90GB, free=78.31GB, total=79.21GB
```

较旧的PyTorch浪费了A100上的1.78GB，而较新的PyTorch只需要0.9GB，从而节省了近0.9GB，这可能成为避免OOM的关键。

`CUDA_MODULE_LOADING=EAGER` 在较新版本的PyTorch中需要强制提前加载CUDA内核，否则它们将在需要时懒惰加载。不要在生产环境中使用此设置，因为这可能会导致比需要的内存更多。懒惰加载的目的正是只在必要时加载内核。

使用`pytorch==2.1.1`：
```
$ CUDA_MODULE_LOADING=EAGER python -c "import torch; x=torch.ones(1).cuda(); free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info()); \
used=total-free; print(f'pt={torch.__version__}: {used=:0.2f}GB, {free=:0.2f}GB, {total=:0.2f}GB')"
pt=2.1.1+cu121: used=0.92GB, free=78.23GB, total=79.15GB
```
与懒惰模式相比：
```
$ python -c "import torch; x=torch.ones(1).cuda(); free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info()); \
used=total-free; print(f'pt={torch.__version__}: {used=:0.2f}GB, {free=:0.2f}GB, {total=:0.2f}GB')"
pt=2.1.1+cu121: used=0.47GB, free=78.68GB, total=79.15GB
```
这里有450MB的差异，但这里我们只加载了用于`torch.ones`的CUDA内核 - 实际的内存分配在运行时的其他Torch API调用中会有所不同，介于0.47和0.92GB之间。

#### 内存碎片化

随着模型分配和释放张量，内存可能会碎片化。这可能导致足够的连续空闲内存不足以容纳较大的内存分配，即使理论上应该足够。因此，即使在OOM的情况下，也可能有足够的可用内存分散在整个内存空间中，无法使用，除非进行非常小的分配。

环境变量`PYTORCH_CUDA_ALLOC_CONF`可以帮助解决这个问题，允许您替换默认的内存分配机制为更有效的。更多信息请参阅[内存管理](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)。

### 批次大小

首先，有两种批次大小：

1. 微型批次大小（MBS），也称为每个gpu的批次大小 - 这是单个gpu在一次模型`forward`调用中消费的样本数量。

2. 全局批次大小（GBS） - 这是所有参与GPU在两次优化器步之间消耗的所有样品的总数。

模型副本是指一次需要多少gpu来容纳完整模型。

- 如果模型适合单GPU，那么模型副本只需1个GPU。通常，可以通过[数据并行性](../../training/model-parallelism#data-parallelism)使用多个GPU来进行训练。
- 如果模型不适合单GPU，那么它通常需要某种形式的分割技术 - 它可以是[张量并行性](../../training/model-parallelism#tensor-parallelism)（TP），[管道并行性](../../training/model-parallelism#pipeline-parallelism)（PP），或者[ZeRO数据并行性](../../training/model-parallelism#zero-data-parallelism)（ZeRO-DP）。

你可以有尽可能多的数据流，就像你有副本一样。这与DP的大小相匹配。
- 所以在一个简单的例子中，模型适合单GPU。有N_GPUS=8个GPU，MBS=4，DP=8。GBS=32，因为：

```
GBS = MBS*DP = 4*8 = 32
```

如果使用TP且程度为2（TP=2）和PP且程度为2（PP=2），这意味着每个模型副本需要4个GPU(`TP*PP`)，现在我们有N_GPUS=8：

```
DP = N_GPUS/(TP*PP) = 8 / (2*2) = 2
```

GBS现在是：

```
GBS = MBS*DP*GAS = 4*2*4 = 128
```

通常，您希望使微型批次大小尽可能大，以便GPU内存接近满载，但又不能过于紧张。

对于非常大的模型，全球批次大小可能会变得非常大。在这种情况下，您可以使用较小的微型批次大小或较少GPU或切换到不同的数据并行形式，以便GPU更有效地工作。

### 梯度积累

梯度积累背后的想法是，与其一次性计算整个批次的梯度，不如逐步迭代地计算它们。通过这种方式，我们可以显著增加整体批次大小，远远超出GPU内存所能容纳的范围。当然，额外的前向和后向传播可能会稍微降低训练的速度。

梯度积累步骤（GAS）定义了在更新模型权值之前等待多少步才进行梯度积累。

当使用管道并行时，非常大的梯度积累步骤是必须的，以将[管道气泡降至最低](../../training/model-parallelism/README.md#naive-model-parallelism-vertical)。

由于优化器步骤不那么频繁，使用梯度积累还可以减少网络开销，特别是在使用[数据并行](../../training/model-parallelism#data-parallelism)时，因为梯度减少是通过`all_reduce`集体完成的，这需要梯度大小的2倍。因此，例如，如果您将GAS从1增加到8，网络开销将减少8倍，这在慢速节点间网络上可以显著提高训练的吞吐量。

### 梯度检查点

梯度检查点也称为激活重算、激活检查点或检查点激活。

这种方法仅在训练期间相关，不在推理期间相关。

启用梯度检查点允许我们在训练吞吐量方面进行交易以换取加速器的内存。当此特性处于活动状态时，模型输出的中间结果不再保留直到`backward`阶段结束。这极大地解放了大量的加速器内存。然而，当然，在`backward`阶段，这些输出必须重新计算。

这当然因模型而异，但通常付出的代价是训练吞吐量下降约20-25%（有时更高，因为大多数激活需要在`backward`中重新计算）。但是，由于释放了大量的GPU内存，我们现在可以大幅增加每个gpu的批次大小，从而整体上提高系统的有效吞吐量。在某些情况下，这使我们能够将批次大小加倍甚至四倍，如果我们在没有检查点的较小批次大小下已经能够做到的话。（最近的论文报道了高达30-40%的额外开销。）

在HF Transformers模型中，您可以通过`model.gradient_checkpointing_enable()`来激活它，或者如果您使用HF Trainer，则可以通过`--gradient_checkpointing 1`来激活它。

XXX：扩展来自[减少大型转换器模型中的激活重算](https://arxiv.org/abs/2205.05198)的新技术的纸，该技术找到了一种避免大多数激活重算的方法，从而同时节省内存和计算。

### 内存高效的优化器

最常见的优化器是Adam。它及其衍生产品占用了每个参数的8字节（2x fp32张量 - 一个是每个动量），这几乎占模型、优化器和梯度总内存分配的一半。因此，在某些情况下，使用其他优化器可能会拯救世界，只要它们能成功训练即可。并非所有的优化器都适合所有的训练任务。

4字节优化器：

- 有像Adafactor这样的优化器，它们只需要4字节。相同的LION优化器最近也被发明了。

- `AnyPrecisionAdamW`。有些勇敢的人试图完全在BF16（不是混合精度！）中进行训练，包括优化器，因此他们只需要4字节/参数用于优化状态。请参阅[此项工作](https://github.com/huggingface/transformers/pull/21312)。提示：这个优化器需要Kahan求和和/或随机舍入，请参阅[回顾BFloat16培训（2020）](https://arxiv.org/abs/2010.06192)。您只需要8字节/参数用于权重、优化状态和梯度！而不是18！

2字节优化器：

- 有量化解决方案，如`bnb.optim.Adam8bit`，它只使用2字节而不是8（1字节用于每个动量）。可以从[这里](https://github.com/TimDettmers/bitsandbytes)获取它。安装完毕后，如果使用HF Trainer，您可以通过简单地将`--optim adamw_bnb_8bit`传递给它来启用它！

对于速度比较，请参阅[基准测试](https://github.com/huggingface/transformers/issues/22101)
在速度方面：`apex`的`apex.optimizers.FusedAdam`优化器到目前为止是最快的Adam实现。自从PyTorch 2.0以来，[torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)添加了对`fused=True`选项的支持，这使其几乎与`apex.optimizers.FusedAdam`相当。

## 模型执行速度

### `forward`与`backward`执行速度对比

对于卷积和线性层，`backward`中的浮点运算量通常是`forward`的两倍，这通常意味着~2倍的减速（有时更多，因为`backward`中的尺寸往往更加尴尬）。激活通常是带宽受限的，在`backward`中，激活通常需要读取比`forward`中更多的数据（例如，激活`forward`读一次、写一次；`backward`中的`gradOutput`和`forward`的输出都需要读，然后`gradInput`需要写一次）。

## 内存剖析工具

在这一章中，我们讨论了理论上模型大小和批次大小是如何计算内存需求的。但实际上事情并不总是这样。因此，您计划了一个特定的模型大小和批次大小，但当您真正使用它时，突然发现内存不足。您需要与实际代码和模型一起工作，找出哪些部分消耗了多少内存，以及是否有未计入的额外开销。

为此，您需要使用某种内存剖析工具。市场上有很多内存剖析工具。

一个有用的小工具，我开发它是为了轻松地对每一行或代码块的CPU/GPU内存分配/释放进行快速和容易的剖析，是[IPyExperiments](https://github.com/stas00/ipyexperiments)。您只需将代码加载到jupyter笔记本中，它就会自动告诉您每个代码块分配/释放了多少CPU/GPU内存。因此，例如，如果您想查看加载模型消耗了多少内存，以及在单个推理步骤中额外增加了多少内存，包括峰值的报告。

## 矢量和矩阵尺寸的可分性

论文[为硬件设计模型架构的案例](https://arxiv.org/abs/2401.14489)调查了变压器尺寸对底层硬件的影响。关联的[脚本](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing)允许您自行运行基准测试，如果您不在NVIDIA V100/A100硬件上运行。

对于GEMMs（全连接的层），NVIDIA提供针对[输入特征](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features)和[批处理大小](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)的建议。

[Tensor Core Requirements](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)定义了基于dtype和硬件的乘数。例如，对于fp16，建议的多倍数为8，但对于A100，它变成了64！

[The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489)提供了有关tile/wave量化和关注头数量的更多详细信息，但要点是：

### Tile和wave量化

注释：

- `a`: 关注头的数量
- `h`: 隐含维度大小
- `s`: 序列长度
- `b`: 微批次大小
- `t`: 张量并行大小

首先，一些背景知识。

NVIDIA GPUs将输出矩阵划分为区域或瓷砖，并将它们调度到一个可用的streaming multiprocessor（SM）上。每个tile或thread block由一个Tensor Core处理，Tensor Core是由NVIDIA引入的，用于快速的tensor操作。Tensor Cores只能充分利用满足特定条件的GEMM。例如，V100和A100 GPU的FP16元素是2字节，这意味着GEMM的`m`、`k`和`n`维度必须是16字节和128字节整数倍，分别用于V100和A100。由于一个FP16元素是2字节，这对应于要素尺寸应是8和64元素的整数倍。如果这些尺寸不是可能的，Tensor Cores在处理Tile时会更好，它们可以接受更大倍数的2字节。

![tiling](images/tiling.png)

有多种tile大小可供kernel选择。如果GEMM尺寸不能整齐地划分成tile大小，将会产生浪费的计算，线程块必须在SM上完全执行，但只有一部分输出是有必要的。这就是所谓的**tile量化**效应，因为输出被量化为离散的tile。

另一种量化效果被称为**波形量化**。当线程块被安排到SM上时，最多108个线程块可以被安排。如果需要调度109个线程块，则需要两轮或多轮调度。第一轮将有108个线程块，第二轮将只有1个。第二轮的延迟将与第一轮相似，但其有用计算的比例要小得多。随着矩阵尺寸的增加，最后一波或尾波会增长。吞吐量会增加，直到需要一个新的波。那时，吞吐量将下降。

这对转换器来说意味着什么？对于给定的`h/a`比率，我们需要确保我们位于波浪的顶部。如果使用NVIDIA V100/A100 GPU，我们已经为您完成了这项工作，详情请见https://arxiv.org/pdf/2401.14489.pdf

对于32个关注头的一个例子：

![wave quantization](images/wave-quant.png)

更多`h/a`的幂可以更好地帮助我们！

### 关注的头部数量和大小

总的来说，保持最大的`h/a`比率而不影响准确性是最节能的。一个好的数字来自于[为硬件设计模型架构的案例](https://arxiv.org/abs/2401.14489)：

![attention heads](images/attention-less-heads.png)

### 闪存注意

如果您使用[闪存注意](https://github.com/Dao-AILab/flash-attention)，好消息是这些MHA尺寸约束得到了照顾。您的唯一约束是保持`h/a`比率足够大以饱和您的GPU核心：

![flash attention](images/flash-attention.png)

### 最终推荐尺寸

完整的推荐是：
1. 词汇量应可被64整除
2. 微批次大小应尽可能大
3. `b*s`、`h/a` 和 `h/t` 应该是2的幂
4. `(b*a)/t` 应该是一个整数
5. `t` 应该尽量小

## 贡献者

[Quentin Anthony](https://github.com/Quentin-Anthony)

翻译：

Stas Sorokin
