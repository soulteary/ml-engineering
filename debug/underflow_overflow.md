# 下溢和上溢检测

对于本节，我们将使用[underflow_overflow](./underflow_overflow.py)库。

如果您开始收到`损失= NaN`或模型由于激活或权重中的`inf`或`nan`而表现出其他异常行为，则需要发现第一次发生下溢或上溢的位置以及导致该问题的原因。幸运的是，您可以轻松地通过激活一个特殊模块来实现自动检测。

让我们以`t5-large`模型为例进行演示。

```python
from .underflow_overflow import DebugUnderflowOverflow
from transformers import AutoModel

model = AutoModel.from_pretrained("t5-large")
debug_overflow = DebugUnderflowOverflow(model)
```

[`underflow_overflow.DebugUnderflowOverflow`]在模型的每个前向调用之后立即插入钩子，测试输入和输出变量以及相应的模块的权重。一旦在激活值或权重的至少一项中检测到`inf`或`nan`，程序将断言并打印类似于以下内容的报告（使用`google/mt5-small`在fp16混合精度下捕获）：

```
检测到inf / nan 在批次编号= 0时
最后21个正向帧：
绝对最小值  绝对最大值  元数据
                编码器.块.1.层.1.密集relu密集. dropout Dropout
0.00e+00 2.57e+02 输入[0]
0.00e+00 2.85e+02 输出
[...]
                编码器.块.2.层.0 MT5层自注意力
6.78e-04 3.15e+03 输入[0]
2.65e-04 3.42e+03 输出[0]
无元数据 输出[1]
2.25e-01 1.00e+04 输出[2]
                编码器.块.2.层.1.层归一化 T5层归一化
8.69e-02 4.18e-01 权重
2.65e-04 3.42e+03 输入[0]
1.79e-06 4.65e+00 输出
                编码器.块.2.层.1.密集relu密集. wi_0 线性
2.17e-07 4.50e+00 权重
1.79e-06 4.65e+00 输入[0]
2.68e-06 3.70e+01 输出
                编码器.块.2.层.1.密集relu密集. wi_1 线性
8.08e-07 2.66e+01 权重
1.79e-06 4.65e+00 输入[0]
1.27e-04 2.37e+02 输出
                编码器.块.2.层.1.密集relu密集. dropout Dropout
0.00e+00 8.76e+03 输入[0]
0.00e+00 9.74e+03 输出
                编码器.块.2.层.1.密集relu密集 T5密集门控GeLU密集
1.79e-06 4.65e+00 输入[0]
3.18e-04 6.27e+04 输出
                编码器.块.2.层.1. dropout Dropout
3.18e-04 6.27e+04 输入[0]
0.00e+00       无限 输出
```

示例输出已在中部截断以节省篇幅。

报告的第一行声明了问题发生的批次号（这里“批次编号= 0”意味着问题发生在第一个批次上）。

每一项报告都以完全限定的模块入口开始，如果查看最后的几项：

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

我们可以看到，最后一帧报告了对`Dropout.forward`函数的调用，其中第一项是唯一的输入，第二项是唯一的输出。您可以看到它来自`dropout`属性内部的`DenseReluDense`类的一个实例的调用。我们还可以看到，在第一层的第2个编码器的第一个批次上，输入和输出的绝对最大值分别为6.27e+04。

现在，我们可以将报告与[`models/t5/modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)中的代码匹配起来：

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

现在很容易找到`dropout`调用以及之前的所有调用。

由于检测是在前向挂钩中进行的，因此这些报告是在每次`forward`返回后立即打印的。

回到完整的报告中，为了采取行动并修复问题，我们需要向上回溯几个框架，因为在那里数字开始变得非常大以至于在fp16中发生了上溢。当然，可能有其他的解决方案。例如，如果我们切换到fp32模式，就可以避免这种问题，这样当乘法或加总大数时就不会发生数值上溢条件。

实际上，探测器已经报告了这些问题，因为上述例子中的每一个调用都是一个`nn.Module`，但是假设你有一些直接计算的地方想要监控，这就是如何做到这一点的方法：

```python
from underflow_overflow import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

在这里，我们在`forward`方法中添加了两处这样的检测点，以便跟踪`forwarded_states`上的任何`inf`或`nan`情况。

此外，如果在自己的代码中初始化了调试器，可以调整默认情况下要保存的最大帧数，比如：

```python
from .underflow_overflow import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

## 具体批次的绝对值混洗和最大化值追踪

同一个调试类也可以用于按批次追踪，同时关闭下溢/上溢检测功能。

例如，如果您想监视特定批次的所有`forward`调用中的所有成分的绝对最小值和最大值，并且只想对批次1和3进行追踪。那么你可以像这样实例化这个类：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

然后，批次1和3将被按照与下溢/上溢检测器相同的格式进行追踪。

批次是从零开始的索引。

这对于知道问题在某个特定的批次号码之后开始出现的情况非常有帮助，所以可以直接跳转到那个区域。以下是这样一个配置的样本截断输出：

```
                  *** 开始批次编号=1 ***
绝对最小值  绝对最大值  元数据
                共享嵌入 Embedding
1.01e-06 7.92e+02 权重
0.00e+00 2.47e+04 输入[0]
5.36e-05 7.92e+02 输出
[...]
                解码器 dropout Dropout
1.60e-07 2.27e+01 输入[0]
0.00e+00 2.52e+01 输出
                解码器 T5堆栈
没有张量输出
                LM头 线性
1.01e-06 7.92e+02 权重
0.00e+00 1.11e+00 输入[0]
6.06e-02 8.39e+01 输出
                T5ForConditionalGeneration
没有张量输出

                  *** 开始批次编号=3 ***
绝对最小值  绝对最大值  元数据
                共享嵌入 Embedding
1.01e-06 7.92e+02 权重
0.00e+00 2.78e+04 输入[0]
5.36e-05 7.92e+02 输出
[...]
```

在这里，您会得到大量帧的转储——与模型中的前向调用数量一样多，因此这可能不是您想要的，但有时这比正常的调试器更容易用于调试目的。例如，如果问题从批次150开始发生，那么可以在批次149和150之间快速前进并进行比较，看看在哪里数字开始偏离。

您还可以指定要在其后面停止训练的批次号码，如下所示：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```

