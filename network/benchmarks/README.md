网络性能基准测试

工具：

- [`all_reduce_bench.py`](all_reduce_bench.py) - 一种用于衡量执行大量数据上的 `all_reduce` 操作时的实际网络带宽的工具。这对于了解现实中的网络性能相对于广告规格的表现非常有用。

- [`all_gather_object_vs_all_reduce.py`](all_gather_object_vs_all_reduce.py) - 一个快速基准测试，显示了在收集进程组中的完成状态时，从 `all_gather_object` 切换到 `all_reduce` 的速度提升了 23 倍。例如，当实现某种所有进程都完成的标志时使用此技术。这种技巧通常用于同步 GPUs 在不同迭代数上可能完成的任务——这适用于跨多个 DP（数据并行）通道的推理，或者当需要同步 `StopIteration` 事件以终止训练循环时。请参阅也包含类似比较的 [`all_gather_object_vs_all_gather.py`](./all_gather_object_vs_all_gather.py)。

- [`all_reduce_latency_comp.py`](all_reduce_latency_comp.py) - 展示了单个 4GB 缩减比 1000 个 4MB 缩减快得多的情况。


关键的可重复性要求：

进行一系列成功实验的最重要要求之一是能够一次又一次地重新创建实验环境，同时只改变一两个设置变量。因此，当你试图确定某个变化是否会提高或降低性能时，你必须找出保持稳定的方法。例如，你需要找到一种防止网络流量波动的办法。当我们为 [108B 预 BLOOM 实验](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide) 做性能优化时，由于我们共享了一个内部节点网络并且相同设置的吞吐量会根据其他用户的使用情况而有所不同，所以几乎不可能进行有效的优化。在 BLOOM-176B 期间，我们被分配了一个专用的 SLURM 分区和一个隔离的网络，其中唯一的流量是我们自己的。在这样的环境中进行性能优化非常完美。

网络吞吐量：
理解你的特定模型大小和框架需求与网络带宽、吞吐量和延迟的关系至关重要。如果你没有支付足够的网络费用，你可能会发现你的 GPU 有空闲时间，从而浪费金钱和时间。另一方面，如果过度支付了超快的网络费用，但你的 GPU 较慢，那么你同样是在浪费时间和金钱。如果你的网络速度很慢，你的培训很可能受限于网络，在这种情况下，对训练设置的许多改进都不会有助于提高性能。

注意：[EAI 食谱](https://github.com/EleutherAI/cookbook)包含了每个集合体的[通信基准](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)，你可以用来快速测量你的内节点或节点间网络的吞吐量。这里有一个简单的全减少基准示例，你可以用来快速测量你的节点间网络的吞吐量：

[`all_reduce_bench.py`](all_reduce_bench.py)

通常建议至少在四个节点上进行基准测试，但是当然，如果你已经可以使用所有将在训练过程中使用的节点，最好在这些节点上运行基准测试。以下是如何在四节点上运行它的示例命令：

```bash
GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py
```

注释：
- 如果这不是 SLURM 环境，你可能需要调整 `MASTER_ADDR` 以匹配排名 0 的主机名。

以下是如何在 SLURM 环境下运行上述命令的示例：
```bash
salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash
srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 --nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d all_reduce_bench.py
```

注释：
- 你可能需要调整 `--cpus-per-task` 和 `--partition` 参数。
- 你只需要执行一次 `salloc` 然后可以在同一个分配上多次重复 `srun`。

结果可能在 5Gbps 和 1600Gbps（截至撰写本文时）之间。对于深度学习模型的大规模训练（64+GPUs），使用 Deepspeed ZeRO Stage 3 时，为了获得合理的 GPU 吞吐量：

1. 100Gbps 是不足够的
2. 200-400 Gbps 是可接受的
3. 800-1000 Gbps 是理想的

[详细信息](https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)

当然，对于 A100 GPU 节点的效率来说，这些要求更高，而对于 H100s 的效率则甚至更高（尽管目前还没有分享这样的基准信息）。