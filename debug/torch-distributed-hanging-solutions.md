# 在多节点多 GPU 的 Python 程序中诊断挂起和死锁问题

本文中的方法是在使用基于 PyTorch 的分布式训练时开发的，当然它们也可以帮助任何多进程、多节点的 Python 程序。

## 辅助工具

尝试使用以下脚本 [torch-distributed-gpu-test.py](torch-distributed-gpu-test.py) 来诊断情况。

这将主要有助于发现网络相关的问题。此外，它还将快速了解多 GPU 通信的工作原理。

对于代码相关的问题，请阅读本文件的其他部分。

## 诊断多 GPU 挂起 / 死锁的方法

### py-spy

首先安装 `py-spy`：
```
pip install py-spy
```

现在你可以通过以下方式附加到每个进程：
```
py-spy dump -n -p PID
```
这会告诉你进程在何处挂起（通常是在 nccl 集合函数或 `barrier` 中）。

- `PID` 是挂起的 Python 进程的进程 ID。
- `-n` 选项很有用，如果你想查看 C、C++ 等编写的扩展中的堆栈跟踪，因为程序可能在其中一个扩展中挂起。
- 你可能需要在使用 `py-spy` 之前以管理员身份运行命令，更多信息见 [这里](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo)。

如果无法以管理员权限运行 `py-spy`，你的系统管理员可能会为你执行此操作：
```
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
这将允许你在不具有管理员权限的情况下运行 `py-spy` 和 `strace`。注意这可能存在潜在的安全隐患，因此建议只在隔离的环境中进行这样的配置更改。

要永久保存该设置，编辑 `/etc/sysctl.d/10-ptrace.conf` 并添加：
```
kernel.yama.ptrace_scope = 0
```

下面是一个 `py-spy dump` 输出的 Python 堆栈跟踪示例：
```
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
第一行是程序挂住的位置。

如果在 CPP 扩展内部发生挂起，可以使用 `--native` 参数启用 `py-spy` 来显示非 Python 代码的堆栈信息（如果有的话）。

#### 多进程 py-spy

那么如何处理多个进程呢？逐一处理太慢了。让我们一次完成所有工作。

如果你的启动命令是 `python`，你可以这样做：
```
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```
如果是 `deepspeed`：
```
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {}
```
或者 `accelerate`：
```
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}
```
你明白了。

这种方法只会分析主进程以及这些进程中没有分叉出来的子进程。所以如果你有 8 个 GPU 和 8 个进程，上述命令将生成 8 个堆栈跟踪。

如果你想要所有的进程及其子进程的信息，你需要运行：
```
pgrep -f python | xargs -I {} py-spy dump --pid {}
```
(同样，将 `python` 替换为实际的启动器名称，如上所示)

这个更长的命令将会捕获所有 Python 进程的堆栈跟踪。

如果你不是获取所有子进程的信息，而是只想查看主进程的情况，你可以这样写：
```
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```
调整 `python` 如果需要，就像上面解释的那样。

如果要在多个节点上运行 `py-spy`，你需要使用 `srun`。

#### 跨节点使用 srun 时的 py-spy

如果你有多台机器怎么办？

你可以手动登录每台机器交互式地收集堆栈跟踪。

如果你在一个使用 SLURM 作为作业管理器的环境中，你可以使用 `srun` 来自动化这个过程。

首先，在其他终端中获取 `SLURM_JOBID`（或者从 `salloc` 日志中获取）：
```
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```
然后使用下面的 `srun` 命令，并将 `SLURM_JOBID` 从上面的输出替换为正确的值：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```
注释：
- `--gres=gpu:0` 用于监控 `srun`，否则它会等待主 `srun`（正在运行训练的那个）退出后才结束。
- 每个节点都会生成一个唯一的日志文件 `trace-nodename.out`，这可以帮助识别哪些节点有问题。你可以移除 `--output=trace-%N.out` 来让所有内容都打印到标准输出。
- 在某些版本的 SLURM 中你可能还需要添加 `--overlap`。
- 在某些版本的 SLURM 中报告的 `SLURM_JOB_ID` 可能与报告中分配 GPU 的那个不同，所以你需要从正在尝试“连接”到的作业的日志中获取正确的 `SLURM_JOB_ID`。
- 有时 `bash` 不起作用，但 `sh` 可以。我认为这与加载了一些特定的环境变量有关。
- 你可能还需要在你的自定义 Python 环境中激活一些环境变量，可以通过这种方式实现：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'conda activate myenvname; ps auxc | ... ' || echo "failed"
```
或者你可以在 `~/.bashrc` 或其他shell的初始化脚本来做这件事。

正如前面提到的，如果你只需要主进程的信息，你可以改写成：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}' || echo "failed"
```
根据需要调整 `python`。

如果 `py-spy` 不能在没有 `sudo` 权限的情况下正常工作，如前所述，你需要：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'sudo bash -c "source ~/.pdshrc; pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}"' || echo "failed"
```
当然，你需要确保 `~/.pdshrc` 有合适的初始化代码以便 `py-spy` 能够正确执行。

如果你得到的都是空结果，可以从基本的调试开始：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
```
一旦你知道你已经成功触达所有节点，你可以逐步深入调用层次结构，比如：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'ps aux | grep python | grep -v grep | grep `whoami` | awk "{print \$2}"'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
每次检查输出是否合理——例如，第二和第三步应该返回进程的 PID。

#### 跨节点使用 pdsh 时的 py-spy

`pdsh` 似乎是一种很好的简单工具，可用于对多个节点上的远程工作进行自动化。假设你在一组名为 `nodename-5` 和 `nodename-8` 的两个节点上运行，你可以快速测试远程执行的可用性，如下所示：
```
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] "date"
```
这将输出类似的内容：
```
nodename-5: Wed Oct 25 04:32:43 UTC 2023
nodename-8: Wed Oct 25 04:32:45 UTC 2023
```

注解：`pdsh` 应该是可以通过标准的操作系统包管理器安装的。

一旦你验证了 `pdsh` 可以工作，就可以尝试使用 `py-spy`。

为了在所有参与的节点上运行 `py-spy`，你可以这样做：
```
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'source ~/.pdshrc; ps aux | grep python | grep -v grep | grep `whoami` | awk '{print \$2}' | xargs -I {} sudo py-spy dump --pid {}'
```

注解：
- 将 `~/.pdshrc` 复制到一个包含所有必要的初始化代码的新文件中，并在 `pdsh` 命令中 `source` 它。
- 如果 `py-spy` 需要在无 `sudo` 权限下运行，你可能需要添加额外的步骤来绕过安全限制。
- 你可能还需要禁用 `ssh` 提示确认主机密钥的功能，这样可以避免每次连接到新节点时都被询问是否继续连接的提示。你可以通过以下方式做到这一点：
```
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```
在这里我假设你在一个封闭的网络环境中，因此不必担心安全性问题。