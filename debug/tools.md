# 调试工具

## 与 Git 相关的工具

### 有用的别名

显示当前分支中所有相对于 HEAD 的更改的差异：
```bash
alias brdiff='def_branch=$(git symbolic-ref refs/remotes/origin/HEAD | sed "s@^refs/remotes/origin/@@"); git diff origin/$def_branch...'
```

同样，但忽略空白差异，添加 `--ignore-space-at-eol` 或 `-w`：
```bash
alias brdiff-nows='def_branch=$(git symbolic-ref refs/remotes/origin/HEAD | sed "s@^refs/remotes/origin/@@"); git diff -w origin/$def_branch...'
```

列出在当前分支与 HEAD 比较时新增或修改的所有文件：
```bash
alias brfiles='def_branch=$(git symbolic-ref refs/remotes/origin/HEAD | sed "s@^refs/remotes/origin/@@"); git diff --name-only origin/$def_branch...'
```

一旦我们有了列表，我们可以自动打开编辑器来加载仅添加和修改的文件：
```bash
alias bremacs='def_branch=$(git symbolic-ref refs/remotes/origin/HEAD | sed "s@^refs/remotes/origin/@@"); emacs $(git diff --name-only origin/$def_branch...) &'
```


### git-bisect

（注意给自己：这是从 `the-art-of-debugging/methodology.md` 同步而来的，该文档是真正的来源）

讨论的下一种方法应该适用于任何支持二分查找的版本控制系统。我们将使用 `git bisect` 在本文中进行讨论。

`git bisect` 帮助快速找到导致特定问题的提交。

用例：假设你之前使用了 `transformers==4.33.0`，然后你需要一个更新的功能，所以你升级到了最新的 `transformers@main`，这时你的代码出现了问题。在这两个版本之间可能有数百个提交，通过逐一检查这些提交来找出哪个导致了问题是非常困难的。下面是如何快速找到负责的提交的步骤。

脚注：Hugging Face 的 Transformer 实际上相当好，很少出现破坏性变化，但是考虑到它的复杂性和庞大的规模，仍然会出现一些问题，并且这些问题会在报告后很快得到修复。由于它是一个非常流行的机器学习库，因此成为了一个很好的调试案例。

解决方案：在已知的好坏提交之间对所有提交进行二分搜索以找到责任提交。

我们将使用两个shell终端：A 和 B。终端 A 将用于 `git bisect`，而终端 B 将用于测试您的软件。虽然技术上您可以使用单个终端完成这一切，但从两个终端工作会更容易些。

1. 在终端 A 中克隆 Git 仓库并将其安装到开发模式（`pip install -e .`）到您的Python环境中。
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
现在，这个克隆中的代码将自动用于运行您的应用程序，而不是您以前在其他地方（如PyPI、Conda或其他地方）安装的版本。

此外，为了简化操作，我们假设所有的依赖项都已经预先安装好了。

2. 接下来，启动二分查找过程——在终端 A 中执行以下命令：

```bash
git bisect start
```

3. 确定已知的好提交和坏的第一个提交

`git bisect` 需要两个数据点才能开始工作。它需要知道一个较早的已知良好的提交（`good`）和一个较晚的已知有问题的提交（`bad`）。如果查看提交序列图，你会发现有两个已知的点和许多未知质量的提交围绕在这些点周围：

```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->---------------->时间
```

例如，如果您知道 `transformers==4.33.0` 是好的，而 `transformers@main` （`HEAD`）是有问题的，那么您需要在标签 `4.33.0` 上找到对应的提交。访问[发布页面](https://github.com/huggingface/transformers/releases)并在其中搜索 `4.33.0`。我们会发现它对应于SHA字符串为 [`5a4f340d`](https://github.com/huggingface/transformers/commit/5a4f340d74b42b594aedf60199eea95cdb9bed0) 的提交。

脚注：通常情况下，前八个十六进制字符就足以在一个给定的存储库中标识唯一的提交，但你也可以使用完整的四十个字符的字符串。

所以在终端 A 中指定第一个已知的好提交：
```bash
git bisect good 5a4f340d
```

然后，正如我们所说，我们将使用 `HEAD` 作为坏的提交，在这种情况下，我们不需要查找相应的SHA字符串，可以直接使用 `HEAD`：
```bash
git bisect bad HEAD
```
如果你知道问题是在 `4.34.0` 之后出现的，你可以按照上述方式找到对应的提交哈希值并用它替换 `HEAD`。

现在我们已经告诉了 `git bisect` 哪些是好提交和坏提交，它会切换到一个中间位置的提交：

```
...... orig_good ..... .... current .... .... ..... orig_bad ....
------------->--------------->---------------->时间
```

您可以通过运行 `git log` 来查看它切换到的具体提交。

而且要提醒一下，我们通过 `pip install -e .` 安装了这个repo，所以Python环境会立即更新到当前提交版本的代码。

4. 好还是坏？

下一步是在终端 B 中运行您的程序一次。

然后在终端 A 中运行：
```bash
git bisect bad
```
如果程序失败，或者：
```bash
git bisect good
```
如果程序成功。

如果结果不好，`git bisect` 内部会将上一个提交标记为新的坏提交，并将范围减半，切换到另一个新的当前提交：
```
...... orig_good ..... .... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->---------------->时间
```
反之亦然，如果结果是好的，你会看到：
```
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->---------------->时间
```

5. 重复直到没有更多提交

继续重复第4步，直到找到引起问题的提交。

完成后，`git bisect` 会告诉你哪个提交造成了问题。

```
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->---------------->时间
```
这将是 `first_bad` 提交。

您可以转到 `https://github.com/huggingface/transformers/commit/` 并附加提交哈希到URL，这将带您到相应的提交详细信息页面（例如，`https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0`），在那里您可以查看与该提交关联的拉取请求。然后，您可以向维护者报告问题或在相关PR中寻求帮助。

如果您的程序运行速度不是太慢以至于无法在数千次迭代中进行测试，那么每次迭代的时间复杂度大约是 `O(log n)`，这意味着对于1024个提交，可以在大约10次迭代内找到问题所在。

如果您的程序运行缓慢，尝试将其缩小为一个能够快速展示问题的最小化示例程序。通常，注释掉大量可能不相关的代码段可以解决问题。

如果您想跟踪进度，您可以要求 `git bisect` 可视化剩余的待测提交范围：
```bash
git bisect visualize --oneline
```

6. 清理

现在，将Git仓库恢复到你开始时的状态（可能是 `HEAD`）：
```bash
git bisect reset
```

同时，可能还需要重新安装好的库版本以便向维护人员报告问题。

有时，问题是由于有意打破向后兼容性的API更改引起的，你可能需要阅读项目的文档以了解发生了什么变化。例如，如果你从 `transformers==2.0.0` 升级到 `transformers==3.0.0`，几乎可以肯定你的代码将会出现问题，因为主要版本号的改变通常意味着引入重大的API变更。


7. 可能的问题和解决办法：

   a. 跳过某些提交

如果在某种原因下当前的提交不能被测试，可以使用 `skip` 选项来绕过它们：
```bash
git bisect skip
```
这样 `git bisect` 将继续处理剩下的提交。

这在某些API发生意外变化而导致程序以完全不同的方式失败的情况下特别有用。

您也可能尝试制作一个适应新API的小型变体程序，并使用它来进行测试，但这并不总是容易做到的。

   b. 反转顺序

通常，`git bisect` 期望 `bad` 出现在 `good` 之后。

```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->时间
```

然而，如果 `bad` 发生在 `good` 之前的修订版中，并且你想找到第一个修复先前存在的错误的提交，你可以反转 `good` 和 `bad` 的定义。为了避免混淆逻辑状态，建议使用一组新的状态名称，比如 `fixed` 和 `broken`。以下是实现的方法：

```bash
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```
然后使用：
```bash
git fixed / git broken
```
代替：
```bash
git good / git bad
```

   c. 其他复杂情况

有时候会有其他的复杂情况，比如不同修订版的依赖项不一致，可能导致一个问题。如果某个依赖项的新版本需要 `numpy=1.25`，而另一个则需要 `numpy=1.26`。如果这些包的版本向后兼容，安装最新版本应该能解决问题。但这并非总是如此。因此在重新测试程序之前，有时需要手动安装正确的依赖项。

有时，当存在一系列与其他问题无关的损坏提交时，可能会出现问题。在这种情况下，要么找到一个不包含其他损坏范围的 `good...bad` 提交范围，要么尝试像前面提到的那样 `git bisect skip` 那些其他损坏的提交。
