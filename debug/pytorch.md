请将以下文本翻译成中文：

# 调试 PyTorch 程序

## 让节点相互通信

一旦你需要使用多个节点来扩展训练，例如，如果你想使用分布式数据并行（DDP）来更快地训练模型，那么你必须要确保这些节点能够彼此之间进行通信，以便它们可以执行诸如收集梯度等通信集体操作。这通常是通过像 [NCCL](https://github.com/nVIDIA/nccl) 这样的通信库实现的。在 DDP 的例子中，每次训练步骤结束时，所有GPU都需要执行一个 `all_reduce` 调用以同步各个进程的梯度。

在这一节中，我们将讨论一个非常简单的案例，即只有两个节点（每个节点有八个GPU）互相通信的情况，然后你可以很容易地将这个方案扩展到更多节点。假设这两个节点的IP地址分别是 10.0.0.1 和 10.0.0.2。

有了IP地址之后，我们需要选择一个端口来进行通信。

在Unix系统中，有64k个可用端口。前1千个端口是保留给常见服务的，这样任何连接到互联网上的计算机都可以事先知道要连接到的端口号。例如，端口22被保留用于SSH服务。因此，当你输入 `ssh example.com` 时，实际上是在建立与 `example.com:22` 的连接。

由于存在数千种不同的服务，1千个保留端口不足以满足需求，因此各种服务可以使用几乎任意的高端口编号。但是不用担心，当你在云上或高性能计算环境中获得一个新的Linux实例时，它不太可能预先安装了许多会占用高数字端口的常用服务，所以大多数端口应该都是可用的。

让我们选择端口 6000 作为我们的示例。

现在我们有了 `10.0.0.1:6000` 和 `10.0.0.2:6000`，我们希望它们能够相互通信。

首先要做的是在两台机器上都打开端口 6000 用于入站和出站流量。该端口可能已经开放，或者你可能需要阅读你的特定环境的文档来了解如何打开特定的端口。

以下是一些测试端口是否已打开的方法示例：

```
telnet localhost:6000
nmap -p 6000 localhost
nc -zv localhost 6000
curl -v telnet://localhost:6000
```

大多数这些命令可以通过 `apt install` 或其他包管理器轻松获取。

在这里，我们可以使用 `nmap` 作为示例工具。如果运行以下命令：

```
$ nmap -p 22 localhost
[...]
PORT   STATE SERVICE
22/tcp open  ssh
```

我们可以看到端口 22 是开放的，并且它告诉我们分配了哪个协议和服务。

现在运行以下命令：

```
$ nmap -p 6000 localhost
[...]

PORT     STATE  SERVICE
6000/tcp closed X11
```

这里我们看到端口 6000 是关闭的。

既然你已经理解了如何进行测试，你现在可以在 `10.0.0.1:6000` 和 `10.0.0.2:6000` 上进行相同的检查。

首先通过终端 A 中的 SSH 登录第一台机器，并在第二台机器上测试端口 6000：

```
ssh 10.0.0.1
nmap -p 6000 10.0.0.2
```

如果一切正常，然后在终端 B 中 SSH 登录第二台机器，并对第一台机器进行反向检查：

```
ssh 10.0.0.2
nmap -p 6000 10.0.0.1
```

如果两个端口都打开了，你应该就可以开始使用了。如果任何一个或两个端口被关闭，你需要找到方法来打开它们。由于大多数云提供商都有自己的解决方案，只需在网上搜索 “open port” 和你的云提供商的名称即可找到相关信息。

接下来重要的是理解计算节点通常会有多张网络接口卡（NIC）。你可以通过运行以下命令来发现这些接口：

```
$ sudo ifconfig
```

其中一个接口通常是用户用来通过SSH连接或用于非计算相关服务（如发送电子邮件或下载数据）的。这个接口通常被称为 `eth0`，其中 `eth` 代表以太网，但它也可能有其他名字。

然后有一个用于节点间通信的网络接口，可以是 Infiniband、EFA、OPA、HPE Slingshot 等技术之一（更多信息见[网络部分](../network#inter-node-networking)）。可能有单个或多达数十个这样的接口。

下面是几个 `ifconfig` 输出的示例：

```
$ sudo ifconfig
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        [...]
```

我删除了大部分输出，只显示了一些关键信息。这里的重点是 `inet` 后面的IP地址，在上述示例中它是 `10.0.0.23`。这是 `enp5s0` 接口的IP地址。

再看另一个例子：

```
$ sudo ifconfig
ib0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
        inet addr:172.0.0.50  Bcast: 172.0.0.255  Mask:255.255.255.0
        [...]
```

在这个例子中，`ib` 表明这是一个 InfiniBand 卡，但实际上它可以代表任何其他供应商的产品。同样，`inet` 告诉了我们这个接口的IP地址是 `172.0.0.50`。

如果你对上面的解释感到困惑，我们想要的是这些节点的IP地址，这样我们才能测试 `ip:port` 对是否已经打开。

最后，回到我们的案例研究，让我们使用 `torch-distributed-gpu-test.py` 脚本来做一次 `all_reduce` 测试，使用两个终端和一个作为协调主机的主节点。对于测试，我们将使用这个辅助调试脚本 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py)。

在终端 A 中：

```
$ ssh 10.0.0.1
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

在终端 B 中：

```
$ ssh 10.0.0.2
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

注意我在两个情况下都使用了相同的 `--master_addr 10.0.0.1 --master_port 6000`，因为我们之前确认过端口 6000 在两台机器上是打开的，而且我们选择了 `10.0.0.1` 作为主控主机。

这种手动从每台机器启动事物的做法很痛苦，因此存在自动在这些节点上部署相同命令的工具。

**pdsh**

`pdsh` 就是这样一个解决方案——它类似于 `ssh`，但可以自动在多个节点上运行相同的命令：

```
PDSH_RCMD_TYPE=ssh pdsh -w 10.0.0.1,10.0.0.2 \
"python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py"
```

你可以看到我将两个命令集折叠成了一个。如果有更多的节点，只需要添加更多的节点名到 `-w` 参数中。

**SLURM**

如果在 SLURM 环境下工作，很可能已经有人为你配置好了所有的端口，所以事情应该能直接工作。但如果不是，上述内容应该有助于诊断问题。

这里是使用 SLURM 的一个例子：

```
#!/bin/bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
#
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role $(hostname -s): --tee 3 \
torch-distributed-gpu-test.py'
```

如果有多于两个节点，只需要改变 `--nodes=X` 中的 `X` 值，这个脚本就会自动适用于任何数量的节点。

**MPI**

另一种流行的方法是使用 [消息传递界面（MPI）](https://en.wikipedia.org/wiki/Message_Passing_Interface)。有几个开源的实现可供选择。

为了使用 MPI，首先需要创建一个包含目标节点列表和每个主机上的进程数目的 `hostfile`。在我们的例子中有两个节点和总共十六个GPU，所以文件的内容将是：

```
$ cat hostfile
10.0.0.1:8
10.0.0.2:8
```

然后运行测试就像这样：

```
$ mpirun --hostfile  -np 16 -map-by ppr:8:node python my-program.py
```

注意，我使用了 `my-program.py` 因为 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) 是为 `torch.distributed.run`（也称为 `torchrun`）而写的，而 `mpirun` 使用的环境变量可能不同，所以在实际应用中，你需要根据你所选择的 MPI 实现来调整环境变量。

注意事项：
- 你可能会遇到需要在运行时指定哪些接口应该被使用的情况。在这种情况下，你可以通过添加 `--mca btl_tcp_if_include 10.0.0.0/24` 来匹配我们的示例。如果你的节点上有许多网络接口，这可能很有用，因为它允许你排除某些你不打算用于节点间通信的接口。
- 你也可以反过来，排除某些特定的接口。例如，如果你的节点上有 `docker0` 和 `lo` 接口，你可以通过添加 `--mca btl_tcp_if_exclude docker0,lo` 来排除这些接口的使用。

`mpirun` 有大量的选项，所以我建议阅读它的手册页以获取更多信息。我的目的是展示如何使用它，而不是详细介绍其功能。此外，不同的 `mpirun` 实现可能会有不同的命令行选项。

### 解决 Infiniband 之间的节点连接问题

在一个 Azure 场景中，我有两个位于同一子网的节点，当我尝试运行 NCCL 测试时：

```
NCCL_DEBUG=INFO python -u -m torch.distributed.run --nproc_per_node=1 --nnodes 2 --rdzv_endpoint 10.2.0.4:6000  --rdzv_backend c10d torch-distributed-gpu-test.py
```

我看到 Infiniband 接口被检测到了：

```
node-2:5776:5898 [0] NCCL INFO NET/IB : Using [0]ibP111p0s0:1/IB [1]rdmaP1111p0s2:1/RoCE [RO]; OOB eth0:10.2.0.4<0>
```

但是连接失败，返回错误：

```
node-2:5776:5902 [0] transport/net_ib.cc:1296 NCCL WARN NET/IB : Got completion from peer 10.2.0.5<33092> with error 12, opcode 0, len
0, vendor err 129 (Recv)
node-2:5776:5902 [0] NCCL INFO transport/net.cc:1134 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:679 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:858 -> 6 [Proxy Thread]
```

没有成功。所以这里以太网连接在节点之间起作用，但 Infiniband 不起作用。

有很多原因可能导致这种情况发生，但在这种情况下，当我在 Azure 上重新创建节点时，我发现问题的根源在于节点没有被分配在一起。因此，虽然以太网连接性在节点之间有效，但由于速度不够快，Infiniband 连接未启用。

在云环境中，节点通常不设计为直接进行节点间的通信，而是倾向于使用特殊的集群概念，这些集群已经被预配置好，可以一起工作。

### 为日志添加 `node:rank` 前缀，交错断言

在本节中，我们将使用 `torchrun` （`torch.distributed.run`）进行演示，并在本节的末尾列出与其他启动器的类似解决方案。

当您收到警告或堆栈跟踪（或调试打印）时，在日志中为每个 `hostname:rank` 前缀添加 `--role` 和 `--tee` 标志可以帮助很多。这样做后，每个日志行都将带有 `[hostname:rank]` 的前缀。

如果您的节点数量较少，比如只有一个节点，那么您不需要传递 `--role`。在这种情况下，仅使用 `-tee 3` 就足够了。

如果您在 SLURM 环境中，则可以将上述命令改写如下：

```
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role $(hostname -s): --tee 3 \
torch-distributed-gpu-test.py'
```

请注意，在上面的命令中，`hostname -s` 命令被延迟执行直到它在每个节点上运行。如果不这么做，`hostname -s` 将只在启动作业的主节点上执行，导致所有节点共享同一个 `hostname` 作为前缀，这将失去使用这些标志的意义。因此，如果使用双引号包围整个命令，你需要将其重写为：

```
srun --jobid $SLURM_JOBID bash -c "python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role \$(hostname -s): --tee 3 \
torch-distributed-gpu-test.py"
```

在这个版本中，`hostname -s` 命令被正确地延迟执行，以确保每个节点都有一个独特的 `hostname` 作为前缀。

重要提示！请注意，在单节点的情况下，您可以直接设置 `CUDA_VISIBLE_DEVICES=""` 来隐藏所有GPU，从而强制程序在CPU上运行。这样可以更容易地生成有意义且易于理解的堆栈跟踪。

如果您的程序依赖于多个GPU并且不能简单地在CPU上运行，您可以尝试设置 `CUDA_LAUNCH_BLOCKING=1` 环境变量。这会将CUDA的异步行为切换为同步模式，使得程序崩溃时的上下文更加清晰。

注意：[NCCL==2.14.3 与 `pytorch==1.13` 一起出现时会挂起](https://github.com/NVIDIA/nccl/issues/750) 当 `CUDA_LAUNCH_BLOCKING=1` 被使用时。因此，在使用 `pytorch==1.13` 时不要使用此变量的最新版本。这个问题已经在 `nccl>=2.17` 中得到了修复，预计将在 `pytorch==2.0` 中包含。