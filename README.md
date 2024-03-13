# 《机器学习工程》开放书籍

这本著作是一份有关大型语言模型和多模式模型训练方法的综合性资料汇编，涵盖了理论框架、工具使用以及详细的步骤指引。

这些资料专为大型语言模型（LLM）和视觉语言模型（VLM）培训工程师和管理员设计，包含了大量的脚本文档和可以直接复制的命令行示例，旨在帮助读者快速解决问题。

该存储库汇集了我多年来在开源大型语言模型（例如2022年的[BLOOM-176B](https://huggingface.co/bigscience/bloom)）和多模式模型（例如2023年的[IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)）的训练过程中积累的专业知识和经验。目前，我正任职于[Contextual.AI](https://contextual.ai/)，专注于开发和训练开源的基于检索增强生成的（RAG）模型。

我将这些信息汇总在一起的主要目的是方便自己在需要时可以快速查找解决方案，但我也非常愿意与更广阔的机器学习社区分享这些内容。

## 目录

如果发现页面布局不稳，这可能是因为我一直在不断地新增章节并逐步优化内容的结构，使之更为清晰易懂。敬请理解！

### 第一部分：洞见

1. **[人工智能战场工程](./insights/ai-battlefield.md)** - 在这一领域的成功所需的知识

### 第二部分：硬件

1. **[计算资源](compute)** - GPU、CPU 和 CPU 内存。

1. **[存储系统](storage)** - 本地、分布式和共享文件系统。

1. **[网络](network)** - 节点内部和节点之间的网络连接。

### 第三部分：调度

1. **[SLURM](orchestration/slurm)** - 主要的管理系统。

### 第四部分：训练

1. **[训练指南](training)** - 与模型训练相关的指南。

### 第五部分：开发

1. **[调试与排错](debug)** - 如何轻松处理简单的或复杂的调试问题。

1. **[更多的调试技巧](https://github.com/stas00/the-art-of-debugging)**

1. **[测试](testing)** - 许多提示和工具，使编写测试变得愉悦。

### 第六部分：其他

1. **[资源链接](resources)** - LLM/VLM的历史记录。

## 更新通知

任何重大更新的公告都会在我的Twitter频道上公布：[@StasBekman](https://twitter.com/StasBekman)。

## PDF版本

下载本书的[PDF](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.pdf?download=true)版本。

我会尽量保持每周更新一次，但如果想要最新版本，你也可以按照[此处](build)的说明自行编译。

感谢Hugging Face允许我在其平台上托管此书的PDF版本。

## 讨论区

如果你想在机器学习工程的任何方面展开讨论，可以在本仓库的[社区讨论板块](https://github.com/stas00/ml-engineering/discussions)中发起新的话题或者加入已有的讨论。我们鼓励大家分享经验和相互学习！

## 快速链接

以下是一些你可能频繁访问的资源的直接链接：

### 工具类

- [all_reduce_bench.py](network/benchmarks/all_reduce_bench.py) - 一个比 `nccl-tests` 更易于使用的网络吞吐量基准测试工具。
- [torch-distributed-gpu-test.py](debug/torch-distributed-gpu-test.py) - 一个用于快速测试节点之间连接的工具。

### 指南类

- [debugging pytorch applications](debug/pytorch.md) - 快速修复 PyTorch 应用程序崩溃或冻结的有效技巧。
- [slurm for users](orchestration/slurm/users.md) - SLURM 用户指南和小贴士。
- [make tiny models/datasets/tokenizers](debug/make-tiny-models-tokenizers-datasets.md) - 制作微型模型的指南。
- [LLM/VLM chronicles collection](resources#publicly-available-training-llmvlm-logbooks) - 公开可用的 LLM/VLM 训练日志精选。

## 鸣谢

如果没有过去委托给我的一些大规模模型训练项目，我不会有今天这样的成就。这种特权只属于少数人，因为租赁庞大的 ML 计算集群成本极为高昂。我希望他人可以通过阅读这些笔记来间接学习我的经验教训。

特别感谢[Thomas Wolf](https://github.com/thomwolf)，是他建议我领导 BLOOM-176B 的训练工作，尽管当时我对大规模训练几乎一无所知。正是那个项目点燃了我深入探索的热情。当然，也要感谢 Hugging Face 给了我机会全职投入到 BLOOM-176B 和后来的 IDEFCIS-80B 项目的训练工作中去。

我将这些信息汇总在一起的主要目的是为了让自己在需要时可以快速找到解决方案，但我很高兴也很愿意与更广泛的机器学习社区分享这些内容。

## 贡献

如果您发现任何错误、拼写错误或者其他需要改进之处，请毫不犹豫地通过提交[问题报告](https://github.com/stas00/ml-engineering/issues)或者直接提交[拉取请求](https://github.com/stas00/ml-engineering/pulls)的方式帮助我们改善这份文档。

## 许可证

本网站内容遵循[知识共享署名-相同方式共享 4.0 国际许可协议](LICENSE-CC-BY-SA)。