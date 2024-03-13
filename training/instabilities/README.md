# 避免、恢复和理解不稳定情况
子部分：
- [了解训练损失模式](training-loss-patterns.md) - 类型如尖峰、分歧、顿悟时刻、重启等。

## 从培训日志中学习
从公开可用的LLM / VLM培训日志书中阅读是最好的学习方式，因为您可以在那里看到确切发生的事情以及如何克服问题。

## STD初始化
正确地初始化张量的初始分布会对训练的稳定性产生巨大的影响。`std`值不是固定的，它取决于隐藏维度的大小。

在我们的预BLOOM 104B实验中，我们发现默认的`--init-method-std`值为0.02在Megatron-LM中的设置对于我们的模型来说太高了，这导致我们在几千次迭代后遇到了稳定性的问题。

我们参考了以下两个来源来确定更合适的`std`值：
1. "Transformers without Tears"论文（https://arxiv.org/abs/1910.05895）建议使用公式：`sqrt(2/(NHIDDEN*5))`
2. 而530B训练论文（https://arxiv.org/abs/2201.11990）使用了甚至更小的初始化公式：`sqrt(1/(NHIDDEN*3))`

我们决定采用530B的建议，因为它导致了更小的初始化值。为了更容易比较这两个公式，我们可以将它们重写为：
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

对于`NHIDDEN=14336`，计算结果是`sqrt(1/(14336*3)) = 0.00482`，这就是我们所使用的值。虽然这不是我们能够在BLOOM-176B训练期间保持稳定的唯一原因，但我认为这是关键因素之一。

## 数值不稳定性
某些数学运算可能在处理低精度数字时是不稳定的。例如，请参阅这个非常有趣的[PyTorch指南关于数值稳定性](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)。现在让我们看一个这种概念在实际中的一个具体例子。

在104B训练实验中，当使用fp16混合精度时，提出了以下改进以使自注意力更加稳定。具体地说，这一行显示了规范因子可能是在查询键矩阵乘法之后相乘的：
```python
matmul_result = torch.baddbmm(
    matmul_result,
    query_layer.transpose(0, 1),   # [b * np, sq, hn]
    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    beta=0.0 if alibi is None else 1.0, alpha=1.0)
```
如果Q和K的维度过大，输出可能会爆炸，而规范因子将无法挽救这种情况。提案是将规范因子向内移动，以便在矩阵乘法之前缩放Q和K：
```python
matmul_result = torch.baddbmm(
    matmul_result,
    1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
    1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    beta=0.0 if alibi is None else 1.0, alpha=1.0)

# change view to [b, np, sq, sk]
attention_scores = matmul_result.view(*output_size)
```
为了使操作在数学上等效，向内移动规范因子需要再次取平方根，如果n是一个标量，A和B是矩阵：
```
n * (A dot B) === (sqrt(n) * A) dot (sqrt(n) * B)
```
现在A和B的维度可以显著更大。对于CUDA内核编写者，[CuBlas](https://docs.nvidia.com/cuda/cublas/index.html)的`GemmStridedBatchedEx`在撰写本文时也有类似的问题定义如下：
```
C+i*strideC=αop(A+i*strideA)op(B+i*strideB)+β(C+i*strideC), for i ∈[0,batchCount−1]
```
问题是`alpha`在矩阵矩阵乘积完成后再相乘，这可能引起不稳定。

## “坏”的数据批量和模型参数状态组合
Palm团队观察到在训练大型模型时每隔“高度不规则的时间间隔”就会出现几十个损失尖峰。尽管他们没有找到根本原因，但他们通过从一个较早的检查点重新启动并跳过潜在的有问题的数据批次来解决这个问题。[第5.1节训练不稳定](https://arxiv.org/pdf/2204.02311.pdf)

