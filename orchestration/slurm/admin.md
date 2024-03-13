# SLURM管理

## 在多个节点上运行命令

### 避免每次登录新节点时被提示信任连接

1. 为了避免每次登录一个新的、之前未访问过的节点时都被提示是否继续连接的对话框，你可以通过以下方式禁用此检查：
   ```bash
   echo "Host *\n\tStrictHostKeyChecking no" >> ~/.ssh/config
   ```
   当然，请确保这样做符合你的安全需求。这里假设你已经在SLURM集群内部，并且不会使用SSH连接到集群外部的主机。如果你不想设置这个选项，那么每次登录新的节点时都需要手动确认。

2. 安装 `pdsh`（并行远程 shell）

现在你可以将想要执行的命令发送到多个节点。例如，要显示所有节点的当前时间，可以使用如下命令：
   ```bash
   PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
   ```
   这将输出类似这样的结果：
   ```
   node-25: Sat Oct 14 02:10:01 UTC 2023
   node-21: Sat Oct 14 02:10:02 UTC 2023
   node-23: Sat Oct 14 02:10:02 UTC 2023
   node-24: Sat Oct 14 02:10:02 UTC 2023
   node-26: Sat Oct 14 02:10:02 UTC 2023
   ```

接下来是一个更复杂的例子，比如杀死所有没有正常退出且占用GPU的进程。首先，我们需要一个列出这些进程ID的命令：
   ```bash
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
   ```
然后我们可以使用 `pdsh` 将这些进程全部强制结束：
   ```bash
   PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] \
       "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
   ```

## SLURM配置

查看SLURM的配置信息：
   ```bash
   sudo scontrol show config
   ```
配置文件位于控制器节点的 `/etc/slurm/slurm.conf` 中。如果对 `slurm.conf` 进行了修改，需要重新加载配置：
   ```bash
   sudo scontrol reconfigure
   ```

## 自动重启节点

当节点需要安全重启（如镜像更新后）时，可以按照以下步骤操作：
   ```bash
   scontrol reboot ASAP node-[1-64]
   ```
对于非空闲状态的每个节点，该命令将在作业结束后等待，然后重启节点并将状态恢复为空闲。注意，需要在控制器的 `slurm.conf` 中设置 `RebootProgram` 为 `/sbin/reboot`，并且在更改配置文件后需要重新配置SLURM守护程序。

## 改变节点状态

可以通过 `scontrol update` 来更改节点状态。例如，要将一个准备使用的节点解除维护状态：
   ```bash
   scontrol update nodename=node-5 state=idle
   ```
或者，要从SLURM资源池中移除一个节点：
   ```bash
   scontrol update nodename=node-5 state=drain
   ```

## 从缓慢退出的任务中释放节点

有时，在作业取消后，进程可能无法及时退出，如果SLURM配置为不无限期等待，可能会自动将这些节点标记为维护状态。然而，实际上这些节点可能是可用的，因此没有理由不让它们提供给用户使用。

为了自动化这个过程，关键是要获取由于“kill任务失败”而被标记为维护状态的节点列表，这可以通过以下命令获得：
   ```bash
   sinfo -R | grep "Kill task failed"
   ```
接着，我们提取和扩展节点列表，检查节点上的用户进程是否已经退出（或尝试先杀死它们），然后将它们切换回可用状态。

你已经学习了如何[在多节点上执行命令](#run-a-command-on-multiple-nodes)，我们将在这个脚本中使用这一技能。下面是一个帮你完成上述工作的脚本示例：[undrain-good-nodes.sh](./undrain-good-nodes.sh)

现在你可以运行这个脚本，任何实际上是空闲但出于某种原因处于维护状态的节点都将被设置为 `idle` 状态，并向用户开放。

## 调整作业的时间限制

要为一个正在进行的作业设置新的时限，比如说两天：
   ```bash
   scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
   ```
如果要额外增加一段时间到之前的设置，比如说再加三个小时：
   ```bash
   scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
   ```

## 如果SLURM出现问题

分析SLURM日志中的事件记录可以帮助理解为什么某些节点会在预定时间前终止作业，或者为什么节点会被完全从系统中移除。
   ```bash
   sudo cat /var/log/slurm/slurmctld.log
   ```
