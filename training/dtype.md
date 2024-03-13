# 张量精度 / 数据类型
这些是在撰写本文时在机器学习中常用的数据类型（通常称为`dtype`）：
浮点数格式：
- fp32 - 32位
- tf32 - 19位（英伟达安培+）
- fp16 - 16位
- bf16 - 16位
- fp8 - 8位（E4M3和E5M2格式）
为了直观比较，请参考以下表示形式：
![fp32-tf32-fp16-bf16](images/fp32-tf32-fp16-bf16.png)
(来源:[英伟达开发者博客](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/))
![fp16-bf16-fp8](images/fp16-bf16-fp8.png)
(来源:[英伟达文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html))
整数格式用于量化：
- int8 - 8位
- int4 - 4位
- int1 - 1位
ML数据类型的演进过程：
最初，ML使用的是fp32，但它非常慢。接下来，[混合精度的概念被发明出来，它结合了fp16和fp32的使用](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/)，这极大地提高了训练速度。但是，fp16被证明不太稳定，并且训练大型语言模型极其困难。幸运的是，bf16出现了，并取代了fp16，使用了相同的混合精度协议。这使得LLM的训练更加稳定。然后，fp8出现了，混合精度切换到了[这种新的格式](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)，这进一步加快了培训速度。有关详细信息，请参阅论文：[FP8 Format for Deep Learning](https://arxiv.org/abs/2209.05433)。为了了解不同格式之间的速度提升差异，请查看此表中的NVIDIA A100 TFLOPS规格（不包括稀疏性）：
| 数据类型            | TFLOPS     |
| ---               | ---      |
| FP32                | 19.5         |
| Tensor Float 32 (TF32) | 156          |
| BFLOAT16 Tensor Core | 312           |
| FP16 Tensor Core     | 312           |
| FP8 Tensor Core       | 624           |
| INT8 Tensor Core      | 624           |
每一种后续的数据类型大约比前一种快两倍（除了fp32，它的速度要慢得多）。同时，随着混合训练模式的普及，机器学习社区开始提出各种量化方法。其中一个很好的例子是Tim Dettmers的[bitsandbytes项目](https://github.com/TimDettmers/bitsandbytes)，该项目提供了许多4位和8位的量化解决方案。DeepSpeed团队也有一些[有趣的量化方案](https://www.deepspeed.ai/tutorials/model-compression/)。TF32：
TF32是一种神奇的数据类型，自Ampere架构以来可在NVIDIA GPU上使用，它允许以远高于常规fp32矩阵乘法速度的速度执行fp32矩阵乘法，但会带来一定的精度损失。以下是A100 TFLOPS的一个示例（未启用稀疏性）：
| 数据类型            | TFLOPS     |
| ---               | ---      |
| FP32                | 19.5         |
| Tensor Float 32 (TF32) | 156          |
如你所见，TF32比FP32快八倍！默认情况下它是禁用的。要在程序的开头启用它，请添加以下代码行：
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
关于实际精度损失的更多信息，请参阅[PyTorch官方文档](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)。
何时使用fp32累加器：
每当使用低精度数据类型时，都需要小心不要在这些数据类型中累积中间结果。类似`LayerNorm`的操作必须不在半精度下工作，否则可能会丢失大量数据。一般来说，只是累积操作需要用fp32完成，因为将很多低精度数值相加是非常容易导致数据丢失的过程。这里有一些例子：
1. 减少集合操作
- fp16: 在fp16中进行是可以接受的，如果存在损失缩放机制的话
- bf16: 只能在fp32中进行
2. 梯度积累
- 最好在fp32中对fp16和bf16进行，但对于后者来说这是必需的
3. 优化器步骤 / 消失梯度问题
- 当一个小梯度被加到一个大数字上时，这个加法往往会被抵消，因此通常使用fp32主权重和fp32优化状态。
- 如果使用[卡汉求和算法](https://zh.wikipedia.org/wiki/%E5%8D%A1%E7%94%B7%E5%B9%B2%E7%AE%97%E6%B3%95)(Kahan summation algorithm)或[随机舍入](https://zh.wikipedia.org/wiki/%E9%9B%B6%E6%9C%BA%E8%8A%BD%E7%BC%96%E7%AD%BE)（在[重新审视BFloat16训练](https://arxiv.org/abs/2010.06192)中被引入），可以在fp16中安全地维护主权重和优化状态。
对于后者的一个例子，请看：[任意精度优化器](https://github.com/pytorch/torchdistx/pull/52)的最新版本在这里找到：[Facebook Research的multiModal库](https://github.com/facebookresearch/multimodal/blob/6bf3779a064dc72cde48793521a5be151695fc62/torchmultimodal/modules/optimizers/anyprecision.py#L17)。
转换后的文本：