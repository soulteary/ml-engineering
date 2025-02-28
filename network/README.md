请将以下文本翻译成中文：

# 节点间和节点内网络硬件

**小节：**

- [基准测试](benchmarks)

## 介绍

仅仅购买或租赁昂贵的加速器来快速训练和推理模型是不够的。你需要确保你的存储IO、CPU和网络足够快，足以“喂饱加速器的熔炉”。如果这一点没有得到保证，那么昂贵的加速器将被低效利用，导致经济损失、更慢的训练时间和推理吞吐量降低。尽管其他组件也可能受到影响，但网络在训练期间（假设数据加载器已经很快）往往成为瓶颈。

如果你的模型适合单个加速器，你不必过于担心。但是现在大多数模型需要多个加速器来加载，而大型语言模型（LLMs）和大容量视觉模型（VLMs）甚至需要在多个计算节点上进行培训，有时甚至在推理时也需要。

大多数计算节点包含8个加速器，有些有4个，还有一些可能有16个，甚至更多加速器。最近，一些节点配备了单个超级加速器。

当模型跨越几个加速器且不离开单个节点时，你所要担心的只是快速的[节点内网络](#节点内网络)。一旦模型的需求扩展到多个节点——这在训练中很常见，因为可以采用多份策略来并行化和加快训练速度——那么快速的[节点间网络](#节点间网络)就变得至关重要。

本文覆盖了这两种类型的网络硬件，报告了它们的理论和有效带宽，并解释了它们如何相互作用。

## 词汇表和概念

你可以安全地忽略下面列出的许多概念和缩写，直到需要理解其中之一为止。

- AR：自适应路由（但也可能意味着聚合路由器）
- ALU：算术逻辑单元
- DMA：直接内存访问
- EFA：弹性结构适配器
- HCA：主机通道适配器
- IB：Infiniband
- MFU：模型浮点运算利用率（例如，`mfu=0.5` 在半精度下使用 A100 时的值为 0.5，因为它从获得 156TFLOPs 的峰值半精度规格中获得了 156/312=0.5）
- NIC：网络接口卡
- OPA：Omni-Path架构
- RDMA：远程直接内存访问
- RoCE：通过汇聚以太网实现的RDMA
- RoE：通过以太网的RDMA
- SHARP：可伸缩层次聚集减少协议
- VPI：虚拟协议互连
- xGMI：socket到socket全局存储器接口

与速度相关的术语：
- 单向：从一个点到另一个点的传输方向为 A -> B
- 双向，双工：两个方向的传输，通常速度是单向速度的两倍，即 A <-> B
- GBps，GB/s：千兆字节每秒（1GBps = 8Gbps）的数据传输速率
- GT/s：吉太位每秒——每秒钟发生的数据传输操作的数量。
- Gbps，Gb/s：千兆比特每秒（1Gbps = 1/8GBps）的数据传输速率
- 截面宽度：将网络分为两部分所需的最小链路数量（不一定相等）。这些链路的带宽称为截面带宽，它经常用作实际网络带宽的度量。有时也被称为最坏情况下的网络容量。这里有一个[很好的答案](https://networkengineering.stackexchange.com/a/29662/93656)解释了这个和其他相关概念，但你不太可能需要理解它们除了知道人们在谈论什么之外，因为你集群的拓扑很可能已经被提供商设计好了。
- 自适应路由改进了静态路由，允许网络中的分组以不同顺序到达目的地。每个交换机上的分组负载平衡被优化以更好地分配网络工作负载。
- 远程直接内存访问类似于节点内的DMA，但它跨节点工作。它允许多个节点之间在不涉及本地处理器、操作系统内核和缓存的开销的情况下进行数据交换，这是TCP/IP使用的。这里有[一个好的概述文章](https://community.fs.com/article/roce-vs-infiniband-vs-tcp-ip.html)介绍了三种主要的实现方式：（1）Infiniband，（2）基于Converged Ethernet的RDMA（IB或UDP-based RDMA），以及（3）iWARP（基于TCP的RDMA）。
