在 SLURM 环境中工作
除非您非常幸运，拥有一个完全由您控制的专用集群，否则您可能需要在与其他用户共享 GPU 的 SLURM 中进行时间调度。即使您在 HPC 上培训并且被分配了一个专用分区，您仍然需要使用 SLURM。

SLURM 是 Simple Linux Utility for Resource Management（资源管理）的缩写，现在被称为 The Slurm Workload Manager。它是一个免费的开源作业调度程序，适用于 Linux 和类 UNIX 内核，许多世界超级计算机和计算机集群都使用了这个工具。

本章节不会试图彻底教您如何使用 SLURM，因为已经有大量的手册可供参考，而是将涵盖一些有助于训练过程的有用特定细节。

- [SLURM 对用户](./users.md) —— 在 SLURM 环境中进行培训所需了解的一切。
- [SLURM 管理](./admin.md) —— 如果不幸的是除了使用之外，您还需要管理 SLURM 集群，那么这份不断增长的食谱列表可以帮助您更快地完成任务。
- [性能](./performance.md) —— SLURM 的性能细微差别。
- [启动器脚本](./launchers) —— 在 SLURM 环境中使用 torchrun、加速器、pytorch-lightning 等来启动作业的方法。
