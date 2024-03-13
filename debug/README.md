调试和故障排除指南：

- 调试PyTorch程序（[./pytorch.md]）
- 在多节点多GPU Python程序中诊断挂起和死锁（[./torch-distributed-hanging-solutions.md]）
- NVIDIA GPU的故障排除（[../compute/accelerator/nvidia/debug.md]）
- 下溢和上溢检测（[./underflow_overflow.md]）
- NCCL调试与性能优化（[./nccl-performance-debug.md]）——用于调试基于NCCL的软件并将其调优至最佳性能的备忘录。

工具：
- 调试工具（[./tools.md]）
- torch-distributed-gpu-test.py（[./torch-distributed-gpu-test.py]）——这是一个针对`torch.distributed`进行诊断的脚本，它检查集群中的所有GPU（一个或多个节点）是否能够相互通信以及分配GPU内存。
- NicerTrace（[./NicerTrace.py]）——这是对Python模块`trace`的一个改进版本，增加了更多的构造函数参数和更实用的输出信息。
