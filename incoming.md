# 要添加或整合的内容

# PDF书籍笔记

来自Sam的想法：https://github.com/saforem2：https://github.com/stas00/ml-engineering/pull/17#discussion_r1439912709
https://quarto.org/，https://quarto.org/docs/gallery/，https://kevinheavey.github.io/modern-polars/，https://quarto.org/docs/output-formats/pdf-basics.html

# 性能

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


# 存储章节

### 存储基准测试：

https://github.com/argonne-lcf/dlio_benchmark

即将到来的罗斯·怀特曼（<NAME>）的建议进行集成：

- 我尝试将卷按工作负载分离，因此保持“大量小文件”的高周转率环境，代码与数据集和检查点等低周转率的大容量存储分离。可能甚至需要对这些进行进一步细分，因为数据集通常是静态的，而检查点则不断旋转。

- 当数据集在网络存储上时，就像桶存储一样，它们应该由大文件组成，并且作为大文件读取（顺序地以大的块为单位，而不是内存映射！）。避免在数据集中搜索。

- HF数据集这样的设置可能会误导人，看起来像一个很大的文件，但实际上经常被内存映射，IO读模式非常疯狂，可能是普通单个文件读取的三到四倍。
  内存映射加载可以关闭，但如果这样做，对于许多数据集，您会将问题转移到DataLoader进程中，这要求一次从磁盘读取过多的数据进入内存。对不同用例中的权衡有更好的了解，特别是在适当时使用Iterable流式传输。

- 在某种程度上，像S3这样的桶存储通过接口限制强制执行合理的使用模式。它看起来像是挂载为一个文件夹，我可以做任何我想做的事情（内存映射文件、写入大量的小文件、删除所有内容等），这就是问题所在。

- 同样，不能期望将分布式文件系统当作本地磁盘对待。如果按照工作负载分离卷，您可能能够利用总存储量的更高百分比。不要混合高周转率的小型文件和低周转率的超大型文件。

- 此外，请注意，一旦您的datasets已优化为适合大型分布式网络文件系统的格式，通常可以直接从云系统中的bucket存储流式传输。在这种情况下，最好将其移出网络文件系统。

# 调试

内存泄漏检查：
```bash
cuda-memcheck --leak-check full python program.py
```

竞争检测：
```bash
cuda-memcheck --tool racecheck
```
附加选项：
--save 保存输出到磁盘
--print-level 控制输出级别

例如：
```bash
cuda-memcheck --tool racecheck --racecheck-report analysis
```

gdb与cuda结合使用：
```bash
cuda-gdb
```

- 将debug_utils.py的功能集成


# 模型并行性

这里有一个很好的表格，显示了每种类型的并行性的缩放方程。
https://www.cerebras.net/blog/cerebras-sets-record-for-largest-ai-models-ever-trained-on-single-device#summary


# 网络

创建一个新的基准部分：

1. nccl-tests
2. `all_reduce_bench.py`
3. https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/communication
4. 与nccl-tests类似，另一个在高性能计算站点广泛使用的基准套件是OSU微基准测试，如osu_lat、osu_bw和osu_bibw。

InfiniBand参考资料：
- [System Administrator Pocket Survival Guide - InfiniBand](https://tin6150.github.io/psg/infiniband.html)

诊断工具：
- `ibstat` - 查看IB设备状态（三种不同的视图）
- `ibstatus`
- `ibv_devinfo`
- `ibnetdiscover` - 扫描拓扑
- `ibroute` - 显示交换机的单播和多播转发表
- `ibdiagnet` - InfiniBand诊断网
- `ibcheckerrors` - 检查端口/节点错误计数器是否在预定义阈值内
- `ibchecknet` - 对子网的端口/节点/错误进行检查。
- `iblinkinfo`
- `ibcheck`
- `wwibcheck`
- `ibswitch` - 验证IB-QNEM是否安装在外壳中
- `ibhosts` - 列出IB网络中的所有主机。
- `ibswitches` - 列出所有IB交换机

跟踪工具：
- `ibping` - 在InfiniBand节点之间发送ping/pong消息
- `ibsysstat` - 从远程节点获取基本信息（主机名、CPU、内存、利用率）
- `ibswitches` - 扫描网络或使用现有网络拓扑文件列出所有交换机
- `ibhosts` - 扫描网络或使用现有网络拓扑文件列出所有主机

显示网络拓扑：
- `iblinkinfo -R`

使用`ifconfig`查找`IPoIB`网络，例如如果您得到`ib0`设备的`inet addr:100.1.1.102`，您可以连接到该地址，例如`ping 100.1.1.102`

找到控制器：
`lspci | grep Mellanox`

打印驱动程序配置（接口名称来自`ifconfig`）：
`ethtool -i enP49239s1`

打印统计信息和性能报告：
`mlnxofedutil stat`

性能测试工具：
- `perftest`包包括：
  - `ib_send_bw`
  - `ib_send_lat`
  - `ib_write_bw`
  - `ib_write_lat`
  - `ib_read_bw`
  - `ib_read_lat`
  - `ib_atomic_bw`
  - `ib_atomic_lat`
- `qperf`测量两个节点之间的带宽和延迟（TCP/IP和RDMA传输）

如果网络速度远低于预期，可能需要在`NCCL_IB_HCA`环境中指定要使用的HCAs（使用`ibv_devinfo`获取HCAs列表）。
```bash
export NCCL_IB_HCA=mlx5
```

Verbs允许命令在对功能丰富的IB交换机上执行操作。


# 测试

- 将testing_utils.py中的功能集成


# <NAME>'s团队在LLNL的工作

NUMA亲和力：
- https://github.com/LLNL/mpibind/tree/master/python
mpibind for Python使mpibind算法能够在任意Python程序中使用。

训练挂起检测工具：
这个是为了扩展：
https://github.com/stas00/ml-engineering/tree/master/fault-tolerance#is-job-hanging-watchdog

Adam的注释：
- https://github.com/LLNL/STAT - 堆栈跟踪分析工具
- https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

- https://github.com/grondo/io-watchdog

关于如何集成STAT的信息可以在以下页面找到：
- https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

有一些"行动"脚本需要编写，这些脚本将在io-watchdog检测到挂顿时执行。这些脚本的详细信息没有在页面上展示，但我可以为您提供更多信息，如果有兴趣的话。用户将创建一个类似于以下的配置文件：
```text
search /usr/local/tools/io-watchdog/actions
timeout = 20m
actions = STAT, kill
```
这将配置io-watchdog，如果在20分钟内未收到rank 0的输出，就认为作业卡住，然后运行"STAT"来收集堆栈跟踪，并运行"kill"来scancel作业。我们还有一些其他的，比如向用户发送电子邮件通知io-watchdog检测到一个挂起。然后可以通过以下方式启动：
```bash
srun --io-watchdog mpi_application
```

快速演示SCR。Python中的用法相当整洁。

安装SCR库（C+++MPI）
https://scr.readthedocs.io/en/v3.0/users/build.html#cmake

安装scr.py模块：
https://github.com/LLNL/scr/tree/develop/python#installing-the-scr-python-module

Python中的示例检查点：
https://github.com/LLNL/scr/blob/1878de8756c2b51882a7cda7b97b142eae4e3995/python/scr_example.py#L64-L105

翻译：