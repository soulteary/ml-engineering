# 推断（Inference）

请注意，以下内容处于非常早期的阶段，目前可以忽略。我们正在收集内容，以便将来进一步开发和细化。

## 词汇表

- LPU：语言处理单元™ （Language Processing Unit™）

## 概念

### 预填充与解码（Prefill and Decode）

在进行推断时，存在两个主要步骤：

1. **预填充**（Prefill）：由于提示中的所有令牌都是已知的，因此可以在一次操作中处理整个提示长度（类似于训练过程），并将中间状态缓存到内存中（通常称为“键值”（Key Value，简称 KV）缓存）。这个阶段的延迟贡献很小，因为即使是一个包含 1000 个令牌的提示也可以在足够的内存下以相当快的速度进行处理。

2. **解码**（Decode）：这是生成新令牌的过程，逐个令牌地进行（采用自回归方法），基于所有的先前令牌（即提示以及到目前为止生成的任何新令牌）。因此，这个阶段对生成过程中的总延迟贡献最大，因为它不像预填充那样能够并行化。

### 批处理（Batching）

逐个令牌地处理解码阶段是非常低效的使用硬件资源的。通过将多个查询组合在一起（即**批量处理**（Batch）），我们可以显著提高硬件利用率，并且能够同时处理多个请求。

可能的最大批次大小取决于加载模型权重后剩余的内存量，以及在预填充过程中已经填入 KV-cache 的中间状态的大小。

#### 静态批处理（Static Batting）

这是一种最直接、最简单的批处理方式，其中前 N 个查询被一起打包成一个大批次。这种方法的缺点是如果许多查询已经完成生成，它们必须等待最长的那个生成查询完成才能返回给调用方，这会大大增加响应时间。

#### 在途批处理（In-Flight Batting）

在途批处理是一种动态的处理方式，其中生成引擎会在结果完成后立即移除已完成的结果，并用新的查询替换它们，而不需要等待整个批次完成。这样做的结果是每个序列的位置都可以独立于其他位置进行不同的进度，例如位置 0 的序列可能在生成它的第 10 个令牌，而位置 1 的序列可能刚刚开始第一个令牌的生成，位置 3 的序列则可能在产生最后一个令牌。

这种方法提高了响应时间，因为在不需要等待整个批次完成的情况下，完成的序列可以被立即返回，并且新的提示无需等待下一个可用批次即可开始处理。当然，如果在计算资源全负荷运行且没有空闲容量的情况下，一些请求可能会遇到延迟，直到有可用的计算资源来处理这些请求。

### 投机性推断（Speculative Inference）

由于逐个令牌生成速度极慢，有时可以通过使用一个小得多的快速草稿模型来欺骗系统以加快这个过程。例如，如果我们有一个正常情况下使用 Llama-70B 进行的缓慢推断任务，但我们可以尝试使用 Llama-7b 作为草稿模型，然后一次性验证其预测是否正确。

示例：假设我们有这样一个提示 “I'm turnin', turnin', turnin', turnin', turnin' around and all that I can see is just” 和接下来的预测 “another lemon tree”。现在，我们可以用 Llama-7b 快速预测这三个令牌，然后在 Llama-70b 上运行一组三个提示的批量处理：

```
[...I can see is just]
[...I can see is just another]
[...I can see is just another lemon]
```
我这里简化了完整的提示，实际上应该包括省略的部分（...），只是为了演示的目的。而且我在这里假装每个令牌都是一个完整单词，但实际上并不是这样的。

接下来，Llama-70b 将一次性生成如下结果：

```
[...I can see is just] another
[...I can see is just another] lemon
[...I can see is just another lemon] tree
```

在这种情况下可能有几种情况发生：
- 如果一切匹配，那么我们在三次快速的步骤和一个较长的步骤内就得到了最终结果，而不是使用三次长步骤。
- 如果只有 "another lemon" 匹配，我们仍然可能节省了一些时间。
- 如果没有任何或很少的内容匹配，我们就浪费了一点时间。

显然，如果令牌数量更多，节约的时间可能会更明显。

不要忽视这样一个事实：虽然我们从整体上做了相同的计算量，但我们通过这种方式有可能极大地减少了用户的平均等待时间——如果草稿模型的性能足够好。

### 键值缓存（Key-Value Caching）

每次重新计算所有之前的键值（key value，简称 KV）之前的状态是非常昂贵的，因此它们会被缓存在加速器的内存中。新的计算出的 KV 值会被追加到现有的缓存中。

![带有缓存的计算流程图](images/infer-kv-cache.png)

（来源：[NVIDIA Developer Blog](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/)）

有些缓存是与模型相关的，而另一些则是特定层的。

### 内存需求（Memory Requirements）

1. 模型参数存储 - 模型参数的数量乘以每字节的数据类型大小，例如，半精度浮点数（fp16/bf16）为 2 字节，单精度浮点数为 4 字节。对于一个 70B 参数的模型，如果使用 bf16，我们需要大约 140GB 的加速器内存。
2. 激活内存（Activation Memory） - 这是用于模型内部运算的临时内存，它依赖于批处理大小和序列长度。
3. KV 缓存注意力张量（KV Cache of Attention Tensors） - 对每个令牌来说，缓存的大小通常是 `2 * hidden_size * num_hidden_layers * dtype_size_in_bytes`，这里的 2 代表的是 K 和 V 缓存。例如，对于 Llama2-70B 模型，如果使用 bf16，这个数字将是 `2 * 8192 * 80 * 2`，大约等于 2.6 MB 每令牌（考虑到 `hidden_size = 8192` 和 `num_hidden_layers = 80`）。对于 1024 个令牌和 16 的批处理大小，这将总计约 42.5GB。

### 模型并行（Model Parallelism）

当模型太大以至于无法放入单个加速器或者即使在勉强放入之后效率不高时，来自训练阶段的相同[模型并行技术](../training/model-parallelism)也适用于推断阶段。

## 推断框架（Inference Frameworks）

### Deepspeed FastGen

[Deepspeed FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) 是 DeepSpeed 团队开发的用于大型语言模型（LLM）的推理系统框架。

最新进展：[Update #1](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen#update--january-19-2024)

论文：[DeepSpeed-FastGen: High-Throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/abs/2401.08671)

#### 动态分割融合（Dynamic SplitFuse）

[Dynamic SplitFuse](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen#b-dynamic-splitfuse-) 利用动态提示分解和统一机制来优化连续批处理和系统吞吐量。

### vLLM

[vLLM](https://github.com/vllm-project/vllm)

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)（以前名为 FasterTransformer，现已合并至 TensorRT-LLM 中）

### TGI

### Orca

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - 这是一个由 C++ 编写的推理引擎，基于 NVIDIA 的 FasterTransformer 作为执行引擎（看起来 FasterTransformer 现在已经集成到了 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 中）。

## 推断芯片（Inference Chips）

### Groq

- [Groq](https://groq.com/)

