# 解决 NVIDIA GPU 问题

## Xid 错误

硬件设备并非完美无瑕，由于制造缺陷或长期使用（尤其是在高温环境下）导致的磨损，GPU 可能会遇到各种硬件问题。虽然许多问题可以自动修复而无需深入了解其原因，但如果应用程序因此崩溃，了解问题的根源并采取适当的措施至关重要。

对于普通用户来说，如果只使用了少量 GPU，可能永远不需要理解与 GPU 相关的硬件问题。但是，如果您参与大规模的机器学习训练，可能会用到数百甚至数千个 GPU，那么理解和处理不同类型的硬件问题是必不可少的。

在您的系统日志中，您可能会偶尔看到类似于以下内容的 Xid 错误：

```
NVRM: Xid (PCI:0000:10:1c): 63, pid=1896, Row Remapper: New row marked for remapping, reset gpu to activate.
```

要查看这些日志，可以使用以下命令之一：
```
sudo grep Xid /var/log/syslog
sudo dmesg -T | grep Xid
```
通常情况下，只要训练没有中断，这些错误往往表明硬件问题已经得到自动纠正。

Xid 错误的完整列表及其解释可以在 Nvidia 的官方文档中找到：[Xid 错误参考](https://docs.nvidia.com/deploy/xid-errors/index.html)。

可以通过运行 `nvidia-smi -q` 来检查是否有任何报告的错误计数：

```
Timestamp                                 : Wed Jun  7 19:32:16 2023
Driver Version                            : 510.73.08
CUDA Version                              : 11.6

Attached GPUs                             : 8
GPU 00000000:10:1C.0
    Product Name                          : NVIDIA A100-SXM4-80GB
    [...]
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 1
        Uncorrectable Error               : 0
        Pending                           : Yes
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 639 bank(s)
            High                          : 1 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
[...]
```
在这里，我们可以看到 Xid 63 与以下内容相关联：

```
ECC page retirement or row remapping recording event
```
这可能有三种原因：硬件错误、驱动程序错误或帧缓冲区（FB）损坏。这个错误意味着内存的一个行出现了故障，需要在重启和/或显卡重置时使用备用内存行进行替换。从上面的报告中可以看出，只有 639 个银行可用（总共 640 个）。

`ECC Errors` 部分的 `Volatile` 和 `Aggregate` 部分分别记录了自上次重启/显卡重置以来的错误数量以及自从开始使用该 GPU 以来的总错误数量。

有两种类型的错误——可纠正的和不可纠正的。可纠正的是单比特 ecc 错误（SBE），尽管内存有缺陷，但驱动程序仍然能够恢复正确的值。不可纠正的是双比特 ecc 错误（DBE），在这种情况下，驱动程序会退休整个内存页面，因为在一个内存地址上发生了不止一个比特的错误。关于详细信息，请参阅[这份文档](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html)。

如果存在计划退休的页面，输出将类似这样：

```
    Retired pages
        Single Bit ECC             : 2
        Double Bit ECC             : 0
        Pending Page Blacklist    : Yes
```
每个退役的页面都会减少可用于应用的内存总量。但由于每个页面的容量仅为 4 MB，它不会显著降低总的可用 GPU 内存。

为了更深入地调试 GPU，请参阅[此文档](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)，其中包含了一个有助于确定何时需要更换 GPU 的故障排除图表。该文档还提供了有关 Xid 63 样错误的信息。例如，它建议：

> 如果与 XID 94 关联，则需要重新启动遇到错误的应用程序。所有其他系统上的应用程序都可以继续正常运行，直到方便的时候再重启以激活内存映射。
> 根据内存映射失败的情况，以下是一些关于何时应考虑更换 GPU 的指南。

如果在重启后相同的条件再次出现于相同的内存地址，这意味着内存映射尝试失败，并且 Xid 64 将再次被触发。如果这种情况持续发生，说明存在无法通过软件手段解决的硬件问题，需要更换 GPU。

在其他时候，您可能会收到 Xid 63 或 64 错误，导致应用程序崩溃。这可能还会生成额外的 Xid 错误，但在大多数情况下，这意味着错误是不可纠正的（即它是某种 DBE 类型的错误，然后会出现 Xid 48 等错误）。

如前所述，可以通过以下方式重置 GPU：

```
nvidia-smi -r -i gpu_id
```
其中 `gpu_id` 是您想要重置的 GPU 的序号。不带 `-i` 选项的所有 GPU 都将被重置。

## 执行诊断

如果您怀疑某个节点上的一个或多个人工智能计算公司 NVIDIA GPU 有故障，`dcgmi` 是一个非常强大的工具，可以帮助快速查找故障 GPU。

NVIDIA®数据中心GPU管理器（DCGM）[在此处文档化](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html)，可以从[此处下载](https://github.com/NVIDIA/DCGM#quickstart)。

这里有一个示例 Slurm 脚本，用于运行非常详细的诊断（级别为 3），大约需要 10 分钟才能完成：

```shell
#!/bin/bash
#SBATCH --job-name=dcgmi-1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -x -e
echo "START TIME: $(date)"
srun --output=%x-%j-%N.out dcgmi diag -r 3
echo "END TIME: $(date)"
```
现在，要在特定节点上运行它：
```shell
sbatch --nodelist=node-115 dcgmi-1n.slurm
sbatch --nodelist=node-151 dcgmi-1n.slurm
sbatch --nodelist=node-170 dcgmi-1n.slurm
```
编辑 nodelist 参数以指向要运行的节点名称。

如果节点处于维护状态或已下线且您无法通过 Slurm 作业启动它，只需 SSH 到节点并在那里直接运行命令：
```shell
dcgmi diag -r 3
```
如果诊断未发现任何问题，但应用程序仍无法工作，您可以再次运行诊断，这次使用级别 4，这将花费超过一小时的时间来完成：
```shell
dcgmi diag -r 4
```
例如，如果您遇到重复出现的 Xid 64 错误，诊断报告可能包括以下内容：

```
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
|-----  Deployment  --------+------------------------------------------------|
| Error                     | GPU 3 has uncorrectable memory errors and row  |
|                           |  remappings are pending                        |
```
此时，您知道应该对有问题的 GPU 提交 RMA 请求，如果内存映射失败。

`dcgmi` 工具包含了多种级别的诊断，有些可以在几分钟内完成，适合作为 SLURM 作业的后记快速执行，以确保节点准备好为下一个 SLURM 任务服务，而不是在用户已经开始他们的任务并发现它崩溃之后才发现问题。

当您准备提交一份 RMA 报告时，会被要求运行 `nvidia-bug-report` 脚本来收集相关信息，以便随 RMA 请求一起提交。我通常会将日志保存下来以备将来参考，使用以下命令之一：
```shell
dcgmi diag -r 3 | tee -a dcgmi-r3-`hostname`.txt
dcgmi diag -r 4 | tee -a dcgmi-r4-`hostname`.txt
```
## 如何检测节点是否缺少 GPU

如果您刚刚创建了一个新的虚拟机实例，有时可能会出现实际拥有的 GPU 数量少于预期的情况。下面是如何快速测试您是否确实拥有预期的 8 个 GPU：

```shell
cat << 'EOT' >> test-gpu-count.sh
#!/bin/bash

set -e

# test the node has 8 gpus
test $(nvidia-smi -q | grep UUID | wc -l) != 8 && echo "broken node: less than 8 gpus" && false
EOT
```
然后运行：
```shell
bash test-gpu-count.sh
```

## 如何检测是否总是分配到同一个坏节点

这对于云用户尤为重要，他们租用 GPU 节点。所以你启动了一个新虚拟机实例，却发现它有一些或者全部的 NVIDIA GPU 坏了。你丢弃了这个实例并启动一个新的，结果还是遇到了同样的问题。

很可能你一直在接收同一台带有同样坏 GPU 的节点。这里是怎样去确认这一点。

在你丢弃当前节点之前，记录如下：

```shell
$ nvidia-smi -q | grep UUID
    GPU UUID                              : GPU-2b416d09-4537-ecc1-54fd-c6c83a764be9
    GPU UUID                              : GPU-0309d0d1-8620-43a3-83d2-95074e75ec9e
    GPU UUID                              : GPU-4fa60d47-b408-6119-cf63-a1f12c6f7673
    GPU UUID                              : GPU-fc069a82-26d4-4b9b-d826-018bc040c5a2
    GPU UUID                              : GPU-187e8e75-34d1-f8c7-1708-4feb35482ae0
    GPU UUID                              : GPU-43bfd251-aad8-6e5e-ee31-308e4292bef3
    GPU UUID                              : GPU-213fa750-652a-6cf6-5295-26b38cb139fb
    GPU UUID                              : GPU-52c408aa-3982-baa3-f83d-27d047dd7653
```
这些 UUID 是每块 GPU 的唯一标识符。

当你下次又创建了一个新的 VM 实例时，再次运行同样的命令，如果得到的 UUID 是一样的，你就知道你得到了同一个坏的 GPU。

为了自动化这个过程，使得每次都有这样的数据，因为你可能在丢弃实例后再也来不及获取这些信息，你可以将其添加到你启动流程中的某一部分：

```shell
nvidia-smi -q | grep UUID > nvidia-uuids.$(hostname).$(date '+%Y-%m-%d-%H:%M').txt
```
确保将日志保存在持久性文件系统中，以便它们能在重启后存活。如果没有这样的系统，可以将日志保存在本地并立即复制到云端。这样，无论什么时候你需要，它们都将会存在。

有时候只是重启一下节点就会获得新的硬件。而在某些情况下，几乎每一次重启都会有新的硬件。这种行为在不同供应商之间可能会有所不同。

如果你不断地得到同一个坏节点，一个技巧是先分配一个新的 VM，同时保持旧的坏节点在线，然后在新的 VM 上运行完毕后丢弃旧的坏节点。这样可以保证你总能得到新的硬件——除了不能保证它们不会有问题。如果你的用例允许这样做，可以考虑购买静态集群，在那里更容易保持良好的硬件。

云提供商通常都有一个机制来报告坏掉的节点。因此，除了丢弃一个坏节点之外，帮助你自己和其他用户并向云提供商报告坏节点是有益的。由于大多数用户只是丢弃坏节点，技术人员可能不会立即注意到问题并将坏节点放回循环中。因此，如果您不是在使用静态集群并且在按需获取随机 VM 的情况下，你可能希望始终保留一份坏 UUID 的日志，以便在第一时间就知道你拿到了柠檬而不是在十个小时的使用后才意识到问题。

云提供商通常有一个机制来报告坏节点。因此，除了丢弃一个坏节点外，向云提供商报告坏节点对自己和其他用户都是有帮助的。由于大多数用户只是丢弃坏节点，技术人员可能不会立即注意到问题并将坏节点放回循环中。因此，如果您不是在使用静态集群并且在按需获取随机 VM 的情况下，您可能希望始终保留一份坏 UUID 的日志，以便在第一时间就知道您收到了柠檬而不是在十个小时的使用后才意识到问题。