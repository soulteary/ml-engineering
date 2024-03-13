请将以下文本翻译成中文：
# 训练

**子部分:**

- [模型并行性](model-parallelism)

- [性能](performance)

- [容错能力](fault-tolerance)

- [可再现性](reproducibility)

- [不稳定性](instabilities)

- [检查点](checkpoints)

- [训练超参数和模型初始化](hparams.md)

- [张量精度/数据类型](dtype.md)

- [使用单一节点模拟多节点设置](emulate-multi-node.md) – 关于如何仅使用单个节点模拟多节点设置的说明 – 我们在这里使用了 `deepspeed` 启动器。

- [从零开始重新训练 HF 中心模型，使用微调示例](re-train-hub-models.md)

**工具:**

- [printflock.py](tools/printflock.py) – 一个微型库，它使您的 `print` 调用在多 GPU 环境中不会交错。

- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py) – 一个基于 `flock` 的包装器，用于 `print`，可以防止当多个进程同时打印时消息被交错——这是在使用 `torch.distributed` 和多个 GPU 时的典型情况。