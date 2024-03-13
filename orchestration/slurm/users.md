请将以下文本翻译成中文：

# SLURM for users

## 快速入门

只需复制这个[example.slurm](./example.slurm)并将其适应您的需求即可。

## SLURM分区

在本文档中，我们将使用一个示例设置，其中包含两个集群名称：

- `dev`
- `prod`

要了解节点的主机名及其可用性，可以使用：

```
sinfo -p dev
sinfo -p prod
```

Slurm配置位于`/opt/slurm/etc/slurm.conf`。

## 资源分配等待时间

```
squeue -u `whoami` --start
```
将显示任何待定作业的预计开始时间。

如果其他用户取消了他们的预订，这些任务可能会提前启动。


## 通过依赖关系请求分配

为了安排一个新的工作当一个或多个当前计划的工作结束（无论它们是否已经运行或者还没有开始），使用依赖机制，告诉`sbatch`在新工作的开始依赖于当前正在运行的工作的成功完成，使用：

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

使用`--dependency`可能导致较短的等待时间，而不是使用`--begin`，因为如果在指定的时间内允许几分钟的延迟，调度程序可能在最后一项工作停止后立即开始其他工作，即使它们的优先级较低。这是因为调度程序忽略了带有`--begin`的所有工作，直到指定时间到达为止。


## 在特定时间进行分配

为了推迟资源的获取以用于给定的时间点，使用：
```
salloc --begin HH:MM MM/DD/YY
```

同样适用于`sbatch`。

它只是简单地将作业放入队列中，以便在请求的时间被执行，就好像您在该时间执行该命令一样。如果有资源在那个时间可用，分配将立即给出。否则，它会排队等候。

有时相对开始时间很有用。还可以使用其他格式。例子：

```
--begin now+2hours
--begin=16:00
--begin=now+1hour
--begin=now+60  # 秒默认为单位
--begin=2010-01-20T12:34:00
```

时间单位可以是`seconds`（默认值）、`minutes`、`hours`、`days`或`weeks`:

## 预分配不设时限的节点

这对于运行重复性的交互式实验非常有用——因此无需等待分配进度。策略是一次性为一段延长的时间分配资源，然后使用交互式的`srun`作业调用使用此分配。

设置`--time`到所需的窗口（例如6小时）：
```
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
现在使用已保留的节点多次运行作业，通过传递作业的`SLURM_JOBID`：
```
srun --jobid $SLURM_JOBID --pty bash
```
如果从`salloc`开始的交互式shell中运行。但也可以直接从另一个shell启动它，在这种情况下需要明确设置`--jobid`。

如果`srun`作业超时或手动退出，您可以再次在同一保留节点上重新启动它。

`srun`可以当然地直接调用真正的训练命令而不只是`bash`。

重要提示：当仅分配一个节点时，分配的shell不在节点上（它永远不会）。您必须找出节点的hostname（报告于分配期间或在`squeue`和`ssh`中）。

当完成后，释放资源，要么通过退出`salloc`启动的shell，要么通过`scancel JOBID`。

实际上，如果只是一个节点，那么甚至不需要使用`salloc`，而是直接使用`srun`来同时分配和提供shell：
```
srun --pty --partition=dev --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --time=60 bash
```

## 超线程

默认情况下，如果cpu具有超线程（HT）功能，则SLURM会利用这一点。如果您不想使用HT，可以通过指定`--hint=nomultithread`来禁用它。

脚注：HT是Intel特有的命名，一般概念是同步多线程（SMT）

例如，对于一个拥有2个具有24核和每个核心2个超线程的CPU的集群，总共可用的处理单元数量是96个超线程或48个物理内核。因此，为了充分利用节点，您需要配置：

```
#SBATCH --cpus-per-task=96
```
或者如果不想要HT：
```
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
```

这最后的方法将为每个核心分配一个线程，并且在这个模式下只有48个实际的核心可用于使用。

注意：根据应用程序的不同，在这两种模式之间可能会有相当大的性能差异。因此，尝试两者并查看哪个给出了更好的结果。

在某些设置（如AWS）上，启用`--hint=nomultithread`会导致all-reduce吞吐量显著下降！而在其他环境中，情况正好相反——没有HT的吞吐量更差！


## 重用分配

例如，当希望在一个相同的节点分配上运行各种作业时。

在一端shell中：
```
salloc --partition=prod --nodes=16 --ntasks=16 --cpus-per-task=96 --gres=gpu:8 --time=3:00:00 bash
echo $SLURM_JOBID
```

在另一端shell中：
```
export SLURM_JOBID=<JOB ID FROM ABOVE>
srun --jobid $SLURM_JOBID ...
```

可能需要设置`--gres=gpu:0`来运行一些诊断作业在节点上。例如，让我们检查所有主机上的共享内存：
```
srun --jobid 631078 --gres=gpu:0 bash -c 'echo $(hostname) $(df -h | grep shm)'
```


## 具体节点选择

要排除特定的节点（在知道某些节点损坏但仍处于空闲状态时有用）：

```
sbatch --exclude nodeA,nodeB
```
或者通过：`#SBATCH --exclude ...`

要使用特定的节点：

```
sbatch --nodelist= nodeA,nodeB
```
也可以使用简短形式`-w`代替`--nodelist`


管理员还可以定义一个名为`feature=example`的特征并在`slurm.conf`中定义它，然后用户可以通过`--constraint=example`要求使用这些节点的一个子集。


## 信号发送给运行中的作业

由于每个SLURM运行都有有限的时间范围，它可以被配置为在预定时间之前向程序发送一个选择的信号。
```
--signal=[[R][B]:]<sig_num>[@<sig_time>]
```
TODO：需要对此进行实验以帮助培训在不保存最后一个检查点后顺利结束。


## 详细的作业信息

虽然大多数有用的信息已经在各种`SLURM_*`环境变量中预设，但在某些情况下，缺少的信息只能通过以下方式获得：
```
scontrol show -d job $SLURM_JOB_ID
```
然后解析所需的内容。

对于已完成的工作，请参阅：
```
sacct -j JOBID
```

例如，与更多详细信息一起查看：
```
sacct -u `whoami` --partition=dev  -ojobid,start,end,state,exitcode --format nodelist%300  -j JOBID
sacct -u `whoami` --partition=prod -ojobid,start,end,state,exitcode --format nodelist%300  -j JOBID
```



## 显示作业


显示我的所有作业：
```
squeue -u `whoami`
```

按作业ID显示作业：
```
squeue -j JOBID
```

按分区显示作业：
```
squeue --partition=dev
```


## 别名

方便的别名：

```
alias myjobs='squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias groupjobs='squeue -u foo,bar,tar -o "%.16i %u %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p prod -o '%A'"
```



## 僵尸进程

如果遗留了一些跨节点的僵尸进程，可以用一条命令杀死它们全部。

```
srun pkill python
```

## 详细的访问SLURM会计账目

`sacct`显示了对Slurm作业会计日志或数据库中的所有作业和作业步骤的会计数据。

因此，这是一个很好的工具用来分析过去的事件。

例如，查看哪些节点最近用于运行gpu作业：

```
sacct -u `whoami` --partition=dev -ojobid,start,end,state,exitcode --format nodelist%300
```

`%300`在这里告诉它在输出中使用300个字符宽度，这样就不会被截断。

查看`man sacct`以获取更多信息字段。


## 队列


### 取消作业

要取消一个作业：
```
scancel [jobid]
```

要取消我所有的作业：
```
scancel -u <userid>
```

要取消某个分区的我所有的作业：
```
scancel -u <userid> -p <partition>
```

### 小贴士

- 如果看到`salloc`ed交互式作业被安排在未来比需要的晚得多，试着取消作业并请求较短的时间段——通常会有一个更接近的时段可用。


## 日志记录

如果我们需要将日志分离到不同的日志文件中，每台节点一个，我们可以添加`%N`（用于缩写主机名），所以我们有：

```
#SBATCH --output=%x-%j-%N.out
```

这将使我们能够确定是否有任何节点行为异常——例如，GPU损坏。这是因为在PyTorch中，错误不会标记来自哪个节点/GPU排名，而日志文件的单独化可以帮助我们识别问题节点。

希望它将成为PyTorch的内置特性https://github.com/pytorch/pytorch/issues/63174，届时将不再需要在日志记录方面做复杂的事情。


## 显示节点的状态
```
sinfo -p PARTITION
```

非常实用的命令是：
```
sinfo -s
```

以及查看主要统计数据，例如：

```
NODES(A/I/O/T) "allocated/idle/other/total".
597/0/15/612
```
这里我们看到总共有612个节点，其中597个已经被分配，0个闲置，15个出于某种原因不可用。

```
sinfo -p gpu_p1 -o "%A"
```

给出：
```
NODES(A/I)
236/24
```

因此，我们可以看到`gpu_p1`分区中有多少节点可用。

### sinfo状态


- idle: 没有正在运行的任务
- alloc: 节点已被分配给正在执行的任务
- mix: 节点有一些CPU被分配，而其他的则是空闲的
- drain: 节点由于管理原因不可用
- drng: 节点正在运行一个任务，但是在任务结束后将不可用（由于管理原因）



### 停用节点

要查看所有停用节点及其停用原因（编辑`%50E`使其成为更长的理由字段）：
```
% sinfo -R -o "%50E %12U %19H %6t %N"
```

或者只使用`-R`，如果只需要简短版本：

```
% sinfo -R
```



## 作业数组


为了运行一系列的作业，使得下一个slurm作业在当前运行的作业结束后的20小时内自动开始，我们使用作业数组。

创建一个作业脚本：

```
$ cat train-64n.slurm
#!/bin/bash
#SBATCH --job-name=array-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 至关重要 - 每个分布式节点只有一个任务！
#SBATCH --cpus-per-task=1            # 每个任务的核数
#SBATCH --time 00:02:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=dev

echo $SLURM_JOB_ID
echo "我是第${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}号作业"
date
sleep 10
date
```

注意`$SLURM_ARRAY_JOB_ID`与`$SLURM_JOB_ID`相同，而`$SLURM_ARRAY_TASK_ID`是作业的索引。

启动它如下：
```
sbatch --array=1-10%1 array-test.slurm
```
这里的`%1`限制了同时运行的任务的数量从这个作业数组中为1。如果没有它，它将试图一次运行所有的工作，这可能不是我们所期望的（在这种情况下，删除`%1`），但是当我们训练时，我们需要每个任务一次。

此外，作为始终如此，这个参数也可以是脚本的一部分：
```
#SBATCH --array=1-10%1
```

现在，玩具slurm脚本准备就绪，我们可以看到它是如何工作的：
```
$ squeue -u `whoami` -o "%.10i %9P %26j %.8T %.10M %.6D %.20S %R"
     JOBID PARTITION                       NAME    STATE       TIME  NODES           START_TIME NODELIST(REASON)
591970_[2-   dev             array-test  PENDING       0:00      1  2021-07-28T20:01:06 (JobArrayTaskLimit)
```
现在作业2正在运行。

要取消整个数组，取消作业ID（数字前面的`_`之前的那个）：
```
scancel 591970
```

要取消单个作业：
```
scancel 591970_2
```

如果重要的是让日志文件包含数组ID，请添加`%A_%a`:

```
#SBATCH --output=%x-%j.%A_%a.log
```

有关更多详细信息，请参见https://slurm.schedmd.com/job_array.html


## 作业数组训练及其暂停和释放

在本食谱中，我们实现了两项操作：

1. 允许对下一个作业的slurm脚本进行修改
2. 允许暂停和恢复作业数组，而不损失其在队列中的位置，即使在未准备好继续运行作业的情况下也是如此

SLURM是一个非常严厉的环境，一个小小的错误可能会浪费几天的时间等待。但是有一些策略可以缓解这种情况的一些严酷性。

SLURM作业有一个“年龄”的概念，即他们在队列中的存在时间，除了项目优先级之外，这决定了他们何时会被调度执行。如果刚刚提交了一个新作业，它没有任何“年龄”，因此在正常情况下，它将在那些已经在队列中一段时间的其他作业之后运行。除非，当然，这个新作业属于一个高优先级的项目，在这种情况下，它将更快地前进。

因此，我们的想法是这样的：

1. `sbatch`一个长作业数组，比如`-array=1-50%1`
2. 在slurm脚本内部不要有任何代码除了一行`source another-script.slurm` - 这样你可以在下次作业运行前随时修改或切换到另一个脚本。
3. 如果需要停止作业数组火车，不要取消它，而是挂起它，这样它的“年龄”就会保持不变。
4. 当你准备好了，解挂作业 - 只有在挂起期间才计算的时间不计入其“年龄”，但所有先前的“年龄”都保留。

唯一限制这种方法的设置是，一旦作业数组启动，你就不能改变节点数量、时间和硬件约束。

下面是如何实现这一目标的一个例子：

创建一个作业脚本：

```
$ cat train-64n.slurm
#!/bin/bash
#SBATCH --job-name=tr8-104B
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # 至关重要的 - 每个节点上一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # 使用的gpu数量
#SBATCH --time 20:00:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=dev

source tr8-104B-64.slurm
```
开始它像这样：
```
sbatch --array=1-50%1 train-64.slurm
```

现在你可以很容易地在`tr8-104B-64.slurm`中编辑脚本，甚至在第一个作业完成之前，并且在下一次作业开始时使用更新的脚本。

如果你需要几秒钟或几个小时来解决一个问题，你可以暂停整个列车：

```
scontrol hold <jobid>
```

然后，当你准备好了，释放它：

```
scontrol release <jobid>
```


## 如何在退出shell后保持scalloc分配

如果您通过以下方式分配了一个节点：

```
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00 bash
```
然后在退出shell后，分配将被丢失。

如果要打开一个交互式shell，该shell应在其生命周期内保持分配，请使用`--no-shell`而不是`bash`，就像这样：

```
salloc --no-shell --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00
```
现在，如果您需要加入节点，请参阅[如何重新连接分配的节点进行交互](#如何重新连接到分配的节点进行交互)。



## 如何重新连接到分配的节点进行交互

要在已分配的节点上保持交互式shell，请使用`--overlap`。

例如，在控制台A中，让我们分配一个节点：
```
$ salloc --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00 bash
salloc: Granted job allocation 1916
salloc: Node my-node-1 is ready for job
```

在控制台B中：
```
$ srun --overlap --pty --jobid 101 bash
```
现在，如果您需要进入特定节点，可以使用`-w`对其进行指定。例如，假设您得到了`node-[1-4]`的分配，并且您想进入`node-3`，请指定：
```
srun --pty -p dev --gpus 8 --time=2:00:00 -w node-3 bash
```
如果出现错误：
```
srun: error: Unable to create step for job 1930: Invalid generic resource (gres) specification
```
请确保添加回`--gres=gpu:8`设置。如果不是最初分配作业时使用了此标志，则可能不需要这样做。

您也可以通过`ssh`访问节点，但这并不总是有效，因为它不会反映虚拟化的视图（例如，节点上的GPU数量或`/tmp/`或`/scratch`上的自动清理）。

这种方法也适用于多节点分配，默认情况下，您将得到第一个节点上的交互式shell。如果需要进入其他节点，可以使用`--overlap`。



## 故障排除



### `SLURM_PROCID`早期插值

在使用SLURM的多节点设置中，正确设置这一点非常重要：
```
"--machine_rank \$SLURM_PROCID"
```
它必须在运行时进行插值，因为如果设置为`"--machine_rank $SLURM_PROCID"`，启动器将挂起。

最好将启动器和程序隔离开来：

```
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
ACCELERATE_CONFIG_FILE=path/to/accelerate.config.yaml # edit me
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf \"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "
PROGRAM="myprogram.py"

CMD="$LAUNCHER $PROGRAM"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --unbuffered \
    --jobid $SLURM_JOBID \
    "

srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a main_log.txt
```

现在启动器将始终工作，用户只需调整`PROGRAM`变量。

有了`torchrun`：

```
export $GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \$SLURM_PROCID
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`:--tee 3 \
    "
```

请参阅[单节点和多节点启动器与SLURM](launchers/)以获取完整的运行示例。


### 节点数目不匹配

如果PyTorch启动器失败，通常这意味着SLURM节点数目与启动器节点数目不匹配，例如：

```
grep -ir nodes= tr123-test.slurm
#SBATCH --nodes=40
NNODES=64
```

这不会奏效。它们必须匹配。

您可以通过设置`NNODES=$SLURM_NNODES`来修复这个问题，或者在初始分配命令中使用正确的数值。



### 查找故障节点并排除它们

有时候，一个节点坏了，这阻止了你训练，特别是重启作业经常遇到同样的节点集合。因此，你需要能够隔离坏节点并从中排除`sbatch`。

要找到一个坏的节点，编写一个小型脚本，它报告节点的健康状况。

例如，测试所有节点上的CUDA是否可用：
```
python -c 'import torch, socket; print(f"{socket.gethostname()}: {torch.cuda.is_available()}")'
```

或者，为了只报告错误的节点：
```
python -c 'import torch, socket; torch.cuda.is_available() or print(f"Broken node: {socket.gethostname()}") '
```

当然，问题可能是别的什么——比如GPU无法分配内存，所以在这种情况下，更改测试脚本来做一些小型的内存分配。这里是另一种方法：

```
python -c "import torch; torch.ones(1000,1000).cuda()"
```

但由于我们需要在所有节点上运行测试脚本，我们不能直接运行上面的命令，我们必须通过`srun`运行它。因此，我们的第一个诊断脚本可以这样写：

```
srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
```

我稍微改变了它，因为有关于引号的错误。

您可以将上述内容转换为一个实际的脚本，而不是一行，这样就没有引号的问题。

现在，一旦找到了有问题的节点，就可以将其反馈给`#science-support`，以便更换它们。

以下是几种不同情况的更多解决方案，以及如何在这些情况下查找坏节点：

### 破碎的NCCL

如果您正在测试需要分布式设置的某样东西，事情变得更加复杂。这就是为什么我们要测试NCCL是否工作。它设置了NCCL并检查屏障是否工作：

```
#!/bin/bash
#SBATCH --job-name=test-nodes-nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 至关重要 - 每个分布式节点只有一个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # 使用的gpu数量
#SBATCH --time 0:05:00               # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

source $six_ALL_CCFRWORK/start-prod

NNODES=2

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export SCRIPT=test-nodes-nccl.py

cat << EOT > $SCRIPT
#!/usr/bin/env python
import torch.distributed as dist
import torch
import socket
import os
import fcntl

def printflock(*msgs):
    """打印"""
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"]);
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
header = f"{socket.gethostname()}-{local_rank}"
try:
    dist.barrier()
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is OK")
except:
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is broken")
    raise
EOT

echo $LAUNCHER --node_rank $SLURM_PROCID $SCRIPT

srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank $SLURM_PROCID $SCRIPT"
```

脚本使用`printflock`来避免交错的打印输出问题。


### GPU内存检查


这个测试CUDA设备上的每个GPU是否能成功分配77Gb（例如，为了测试80GB A100s）（必须减去几个GBS才能容纳cuda内核）。

```python
import torch, os
import time
import socket
hostname = socket.gethostname()

local_rank = int(os.environ["LOCAL_RANK"]);

gbs = 77
try:
    torch.ones((gbs*2**28)).cuda(local_rank).contiguous() # 首先在cpu上分配，然后移动到gpu
    print(f"{local_rank} {hostname} is OK")
except:
    print(f"{local_rank} {hostname}未能分配{gbs}GB DRAM")
    pass

time.sleep(5)


```


### 网络中断

另一个节点问题是在网络上发生中断的情况。你可能遇到类似这样的错误：
```
work = default_pg.barrier(opts=opts)
RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch_1616554793803/work/torch/lib/c10d/ProcessGroupNCCL.cpp:825, unhandled system error, NCCL version 2.7.8
ncclSystemError: System call (socket, malloc, munmap, etc) failed.
```

这里是如何调试这个问题：

1. 添加：
```
export NCCL_DEBUG=INFO
```
在`srun`命令之前，重新运行你的slurm脚本。

2. 现在研究日志。如果发现：
```
r11i6n2:486514:486651 [1] include/socket.h:403 NCCL WARN Connect to 10.148.3.247<56821> failed : Connection refused
```
让我们看看哪个节点拒绝接受连接。我们从错误中得到IP地址，然后反向解析它以获取其名称：
```
nslookup 10.148.3.247
247.3.148.10.in-addr.arpa       name = r10i6n5.ib0.xa.idris.fr.
```

接下来，将`--exclude=r10i6n5`添加到`sbatch`命令中，并向JZ admins报告问题。


### 跨所有节点运行py-spy或任何其他监控程序

在处理挂起时，这是如何自动记录`py-spy`跟踪每个进程。

当然，这个过程也可以用于在任何给定作业的节点上运行其他程序。也就是说，它可以在正常的运行过程中用于运行某些东西，例如通过`nvidia-smi`或任何其他程序来检查每个进程的内存使用情况。

```
cd ~/prod/code/tr8b-104B/bigscience/train/tr11-200B-ml/

salloc --partition=prod --nodes=40 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time 20:00:00

bash 200B-n40-bf16-mono.slurm
```

在另一个shell中获取JOBID：
```
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```
调整jobid并按上述方式输入`srun`命令：
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```
现在所有`py-spy`跟踪都将进入`trace-$nodename.out`文件，位于`cwd`下。

关键是要使用`--gres=gpu:0`或类似的东西，否则第二个`srun`将阻塞等待第一个释放gpus。

此外，假设您在`~/.bashrc`中加载了conda环境，其中安装了`py-spy`。如果没有，请在`py-spy`命令之前加载环境。

不要忘记在过程完成后手动释放分配。
