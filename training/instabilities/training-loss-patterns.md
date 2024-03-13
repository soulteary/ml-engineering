# 对训练损失模式的理解

训练损失图类似于心电图，有良好、不良和需要担忧的情况。通过对许多训练损失轨迹的研究，人们可以发展出一种直觉来解释训练过程中出现的各种损失行为以及如何应对这些情况。

在标题中的“理解”一词中存在一定的误导性，因为很多时候我们并不真正了解为什么会发生某些类型的尖峰。这里的“理解”指的是识别不同的模式。然后，通常我们有技术手段来解决不好的模式并成功地将培训带到终点线。

因此，您将在这里找到一个训练损失模式的画廊，有时带有真正的解释，但更多时候是关于可能发生的事情的有根据的猜测。

请原谅这些快照看起来彼此之间差异很大，它们来自多个来源并在多年内收集。

## 好的、坏的和不寻常的

让我们看看一些好、坏和不寻常的模式。

### 非常失败的训练

在开始BLOOM-176B训练之前，我们对[104B模型](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)进行了多次实验。我们无法弄清楚如何在早期不发散。

![](images/pre-bloom-104B-en-fail.png)

如您所见，尝试了很多次，应用了许多技术（参见[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md)）。我们认为导致失败的两个主要障碍是使用fp16和数据中有大量垃圾。对于BLOOM-176B，我们切换到bf16，使用了更干净的数据，还添加了嵌入层归一化，这使得所有区别。

### 几乎完美的训练

![](images/bloom-176B-success.png)

[BLOOM-176B](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml)的训练具有近乎完美的训练损失轨迹，只有一个尖峰在200步后恢复。

您可以检查[TB](https://huggingface.co/bigscience/tr11-176B-logs/tensorboard)以放大并查看其他图表。

这是一次接近完美的训练，确实投入了大量艰苦的工作来实现这一点。

### “顿悟”时刻

最近，我在进行一些性能测试时，在一个包含8个A100节点的集群上用全球批量为8对从零开始的llama-2-7b进行了微小的批量训练。（使用DeepSpeed ZeRO-3 DP与HF Transformers[Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)实现）

![](images/llama-7b-grokking-no-zoom.png)

这里可以看到掩码令牌预测的快速损失改进，从大约4到2.5仅在480个样本之后，在经历了非常稳定的缓慢改进之后。我的同事[Gautam Mittal](https://github.com/gmittal)称之为“顿悟”（[Grokking](https://en.wikipedia.org/wiki/Grok)）的时刻。仅仅几十个步骤，模型突然普遍化为更好地预测掩码令牌。

正常情况下，当使用较大的批次大小时，不会看到如此戏剧性的改进。

如果放大，它在大约60个每迭代8个样品的步骤内发生了：

![](images/llama-7b-grokking.png)

## 主要的损失尖峰类型

一般来说，有三种类型的损失尖峰：

1. 快速恢复的尖峰
2. 慢速恢复的尖峰
3. 不完全恢复的尖峰

尖峰通常是由于数据口袋问题引起的，要么是因为数据没有正确地随机洗牌，要么是因为数据中没有清理掉网站上的垃圾内容。

虽然人们可能会怀疑触发尖峰的是前一批数据，但如果仔细研究那批数据的內容，很可能是找不到任何异常——这种情况往往是在很多步骤之前就开始出现问题，然后在最不经意的时候发生。此外，要研究这个批次的内容可能不容易，因为在大型全局批量和序列长度非常大的情况下，这个问题可能相当于一本书的大小。

### 快速恢复的尖峰

损失尖峰可以经常发生，只要它们迅速反弹回到离开的地方，训练通常会继续就像什么都没发生过一样：

这是一个来自[pre-BLOOM 13B训练实验](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base)的例子：

![](images/pre-bloom-tr1-13B-glitch-1-2.png)

正如你所看到的，有很多尖峰，其中一些幅度非常大，但是它们都很快恢复了。

### 慢速恢复的尖峰

下面是一个来自[IDEFCIS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md)训练的慢速恢复尖峰示例：

![](images/idefics-80b-tr-190-01-spike-recover-2023-05-30.png)

### 不完全恢复的尖峰

以下是从[pre-BLOOM 104B模型尝试](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)的不完全恢复尖峰的一个例子：

![](images/pre-bloom-tr8-104B-glitch-1.png)

在这里，你可以看到尖峰发生后，损失开始恢复，但它决定不完全恢复，而是开始发散。

### 非尖峰的发散

以下是几个没有通过尖峰就发散的例子：

![](images/pre-bloom-tr8-104B-glitch-5.png)

还有更多的例子：

![](images/pre-bloom-tr8-104B-glitch-7-10.png)

每次重启都会取得一点进展，然后模型就会发散。

这些都是来自[pre-BLOOM 104B模型尝试](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)。

### 与多数据集相关的尖峰

在[IDEFCIS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md)训练期间，我们使用了两种不同类型的混合数据集：

![](images/idefics-80b-tr-190-01-losses-2023-06-04.png)

图例：cm4（高）、平均（中）和pmd（低）

你可以看到损失尖峰有时同时发生在两个数据集上，而在其他时候只有其中一个数据集的损失会尖峰。

这里模型在学习两种不同的数据分布，正如你看到的，它在两者上报告的损失不同，并且尖峰行为也不同。PMD数据集的损失对模型来说比CM4的要容易得多。

## 与恢复训练相关的尖峰

由于硬件崩溃或出于需要回滚到较早检查点的原因而导致的训练中断几乎是不可避免的。如果你的训练软件不能完美地处理恢复以至于模型没有注意到曾经发生过中断，那么可能会遇到各种各样的问题。

重新启动的最复杂的挑战之一是恢复各种RNG状态，恢复DataLoader索引到上次训练停止的位置，以及处理如果你使用复杂DataLoaders时的特定需求。

### 与DataSampler相关的问题

在[IDEFCIS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md)训练期间，我们的DataLoader非常复杂，并且在恢复时图像到文本比率波动会导致损失出现小尖峰，所以我们最终在每次恢复时都有一个小尖峰，然后恢复：

![](images/idefics-80b-tr-190-01-image2text.png)

你可以看到损失和比例图的相关性在这里。因为我们不得不多次恢复，所以我们在训练过程中看到了很多这样的尖峰。

### 重复数据的影响

我正在训练Llama2的一个变体，并遇到了这种超级不寻常的尖峰，它既没有发散也没有恢复，而是切换到了一个新的更高损失的级别：

![](images/ptl-repeat-data-p1.png)

我将时间倒退到奇怪的行为发生之前并重新开始。损失训练在前一段相同的损失水平下进行了一段时间，然后再次尖峰并转移到更高的损失。

![](images/ptl-repeat-data-p2.png)

我以前从未见过这种类型的发散。我花了一些时间思考，然后决定看更大的画面。

截至本文撰写之时，[Wandb](https://wandb.ai/)在处理带有一个或多个恢复数据的图标方面做得不是很好，它会忽略恢复后的新数据直到超过旧数据的步骤数为止。这迫使我们在每个带有rollback的恢复时创建一个新的wandb图，以便显示新的数据。如果我们想查看整个图，我们需要拼接它们，其中包括不再准确的死数据点。所以我做了拼接工作并看到了这个谜题：

![](images/ptl-repeat-data-p3.png)

实际上并没有真正的尖峰出现在两次运行的前面。损失从来没有上升过。在两次恢复训练中，损失都没有被正确报告，因为它遇到了完全相同的数据，然后遇到了未见过的数据并开始了正确的报告。换句话说，它是过度拟合并报告了一个错误的损失。

问题的原因是数据重复，由于它显然记住了其中的一些数据，它可以报告更好的损失。

该问题的原因在于[PyTorch Lightning](https://github.com/lightning-ai/lightning)没有正确处理恢复时的DataSampler自动操作——基本上每次你恢复时，你的数据流都将从头开始。当然，用户需要解决这个问题。你可以改变种子来缓解这种情况，避免完全相同的数据序列，但这仍然留下了重复的数据，这不是任何严肃的训练（或者消融实验，因为您的观察将是无效的，如果他们假设独立同分布的数据）所需要的。

脚注：我与PTL开发者讨论了这个[问题](https://github.com/Lightning-AI/lightning/issues/18780)，他们说他们努力寻找一个通用的解决方案，但没有成功。因此，用户需要自己解决问题。

确保检查您的训练框架文档是否正确处理DataSampler的恢复。确保你没有在训练完成后再发现这个问题，因为你最终可能在计划看到的300B个token中只看过一次的情况下训练了6倍多的50B个token。

在进行几次恢复训练之前，最好先进行一些早期的训练，这样就可以暴露潜在的问题。不过，如果每次恢复训练时数据都被重新混排，你可能就不会发现问题。只有在相同的种子下才会显现出来。

