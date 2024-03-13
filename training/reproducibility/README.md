# 可复现性

## 在基于随机性的软件中实现确定性

在调试时，请始终为所有使用的随机数生成器（RNG）设置固定的种子值，以便每次重新运行时都能获得相同的数据/代码路径。尽管由于存在许多不同的系统，可能很难覆盖它们的所有内容。以下是尝试涵盖其中一些内容的示例：

```python
import random, torch, numpy as np

def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"正在使用种子：{seed}")

    random.seed(seed)  # Python的RNG
    np.random.seed(seed)  # NumPy的RNG

    # PyTorch的RNGs
    torch.manual_seed(seed)  # CPU和CUDA上的
    torch.cuda.manual_seed_all(seed)  # 多GPU - 可以在没有GPU的情况下调用
    if use_seed:  # 较慢的速度！https://pytorch.org/docs/stable/notes/randomness.html#cuda-卷积基准测试
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
```
如果使用的是其他子系统或框架，则可能还需要以下几行来确保它们的随机性是可复现的：
```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```
当您一遍又一遍地运行相同的代码以解决某个问题时，请在代码的开头设置特定的种子：
```python
enforce_reproducibility(42)
```
但是，如上所述，这仅适用于调试，因为它激活了各种Torch标志以帮助提高确定性，但可能会降低速度，因此不希望在生产环境中使用这些标志。然而，你可以这样调用它来进行生产环境中的操作：
```python
enforce_reproducibility()
```
即，不指定显式的种子。然后它会选择一个随机的种子并记录下来！所以如果在生产过程中出现问题，现在可以重现观察到的问题所处的RNG状态。而且这次没有性能惩罚，因为只有在明确提供种子时才会设置`torch.backends.cudnn`标志。例如，日志显示：
```
正在使用种子：1234
```
你只需要将代码改为：
```python
enforce_reproducibility(1234)
```
就可以得到完全相同的RNG配置。

正如第一段提到的那样，系统中可能有多个RNG涉及，例如，如果你想要数据按照相同的顺序被喂入`DataLoader`，你需要[设置它的种子](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)。

附加资源：
- [PyTorch中的可复现性](https://pytorch.org/docs/stable/notes/randomness.html)

## 再现软件系统和环境

这种方法在发现结果中的不一致时非常有用——质量或吞吐量等。

想法是在启动训练（或推理）时记录关键组件的环境信息，以便以后需要精确复制该环境时可以使用这些信息。由于系统的多样性和组件的使用方式多种多样，不可能提供一个总是适用的解决方案。让我们讨论一种可能的配方，然后你可以将其适应到你特定环境的需要。

这个方法添加到了你的Slurm启动脚本中（或者无论你用来启动训练的其他方式是什么）——这是一个Bash脚本：

```bash
SAVE_DIR=/tmp # 编辑为一个真实存在的目录
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. 模块（写入stderr）
module list 2>$REPRO_DIR/modules.txt
# 2. 环境变量
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. Pip（包括开发安装的SHA）
pip freeze > $REPRO_DIR/requirements.txt
# 4. Git克隆未提交的差异，如果它们安装在conda中
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; 如果(! -z "\$(git diff)");那么git diff > \$REPRO_DIR/$2.diff;否则]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

如你所见，这个食谱是在SLURM环境下工作的，所以每一次新的训练都会将特定于SLURM作业的环境信息保存下来。

1. 我们保存加载的哪些“模块”的信息，比如在云集群/高性能计算环境中，你可能需要通过这种方式加载CUDA和cuDNN库。

   如果不使用“模块”，请删除这一部分。

2. 我们将环境变量导出到一个文件中。这可能至关重要，因为像`LD_PRELOAD`或`LD_LIBRARY_PATH`这样的单个环境变量可能在某些环境中对性能产生巨大影响。

3. 我们还导出了Conda环境中的Python包及其版本——这在任何虚拟Python环境中都适用。

4. 如果使用了`pip install -e .`进行开发安装，那么`pip freeze`不会捕获本地修改过的文件的差异，因为在安装过程中并不知道这些差异的存在。因此，这部分会遍历那些不是从Conda环境中安装的包（我们从`site-packages/*.dist-info/direct_url.json`找到它们的位置）。

一个额外的有用工具是[conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)，它可以帮你找出两个Conda环境之间的确切区别。

经验之谈，我和我的同事在一个云端集群上遇到了一个问题，我们在那里运行相同的代码——实际上是从同一个共享目录中调用了slurm脚本。首先，我们使用[conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)比较了我们各自的Conda环境，发现了差异——我安装了她拥有的所有包来匹配她的环境，但它仍然显示出巨大的性能差异。然后我们比较了`printenv`的输出，发现在我有`LD_PRELOAD`设置了而她没有的情况下，这导致了很大的不同，这是因为那个特定的云供应商要求设置几个环境变量到自定义路径才能充分利用他们的硬件。