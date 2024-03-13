# 性能：SLURM

本节将讨论影响性能的特定于SLURM的设置。

## `srun` 的 `--cpus-per-task` 可能需要显式指定

您需要确保通过 `srun` 启动的程序接收到预期数量的CPU核心。例如，在典型的机器学习训练场景中，每个GPU至少需要一个CPU核心来驱动进程，再加上几个用于数据加载器的核心。您需要多个核心，以便每个任务都可以并行执行。如果您的节点上有8个GPU和每个GPU上的两个数据加载器工作者，那么您总共需要至少 `3 * 8 = 24` 个CPU核心。

每个任务的CPU数量由 `--cpus-per-task` 定义，它被传递给 `sbatch` 或 `salloc`，并且最初 `srun` 将继承此设置。然而，最近这种行为已经改变：

引自 `sbatch` 的手册页：

> **注意**：从版本 22.05 开始，`srun` 将不会继承 `salloc` 或 `sbatch` 中请求的 `--cpus-per-task` 值。如果需要在任务之间共享资源，则必须在调用 `srun` 时再次请求该值，或者使用环境变量 `SRUN_CPUS_PER_TASK` 进行设置。

这意味着如果您以前的SLURM脚本是这样的：

```shell
#SBATCH --cpus-per-task=48
[...]

srun myprogram
```

并且 `srun` 之前会继承来自 `sbatch` 或 `salloc` 的 `--cpus-per-task=48` 设置，根据上述文档中的说明，从版本 22.05 开始，这种情况不再成立。

脚注：我在测试中发现，即使是在 SLURM 版本为 22.05.09 的情况下，旧的行为仍然有效，但正如这里所报告的那样（https://github.com/Lightning-AI/pytorch-lightning/issues/18650#issuecomment-1872577552），这确实发生在 23.x 系列中。因此，更改可能在后来的 22.05 系列中发生。

所以，如果你保持原样，现在你的程序只会得到一个CPU核心（除非 `srun` 的默认设置已经被修改）。

要轻松测试您的 SLURM 配置是否受到影响，可以使用 `os.sched_getaffinity(0)`，它会显示当前进程可以使用的CPU核心数。因此，很容易用 `len(os.sched_getaffinity(0))` 计算这些核心的数量。

下面是如何测试你是否受到影响的示例脚本：

```shell
#!/bin/bash
#SBATCH --job-name=test-cpu-cores-per-task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # 根据您的环境调整，如果你的节点没有超过 48 个 CPU 核心
#SBATCH --time=0:10:00
#SBATCH --partition=x        # 根据您的环境调整到正确的分区名称
#SBATCH --output=%x-%j.out

srun python -c 'import os; print(f"visible cpu cores: {len(os.sched_getaffinity(0))}")'
```

如果输出是：

```
可见的CPU内核：48
```

那么你不需要做任何事情；如果是：

```
可见的CPU内核：1
```

或其他小于 48 的值，那么你就受到了影响。

为了解决这个问题，你需要在你的SLURM脚本中添加以下内容之一：

```shell
#SBATCH --cpus-per-task=48
[...]

srun --cpus-per-task=48 myprogram
```

或者：

```shell
#SBATCH --cpus-per-task=48
[...]

export SRUN_CPUS_PER_TASK=48
srun myprogram
```

## 启用超线程还是不启用？

如[用户指南](users.md#hyper-threads)中所述，如果您的CPU支持超线程技术，您可以理论上将可用的CPU核心数量增加一倍，这在某些工作负载下可能会带来整体更快的性能提升。

但是，你应该测试启用和禁用HT时的性能，比较结果，并根据最佳效果选择设置。

案例研究：在AWS的p4实例上，我发现启用HT使网络吞吐量慢了四倍。从那时起，我们在那个特定的设置上就小心地禁用了HT。