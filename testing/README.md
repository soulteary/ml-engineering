请将以下文本翻译成中文：

# 编写和运行测试

注意：本文件中的一部分内容引用了由Hugging Face提供的[testing_utils.py](testing_utils.py)中的功能，这些功能是我之前在Hugging Face工作时开发的。

本文涵盖了使用`pytest`和`unittest`的功能，并展示了如何将两者结合在一起使用。


## 运行测试

### 运行所有测试

```console
pytest
```
我使用了以下的别名：
```bash
alias pyt='pytest --disable-warnings --instafail -rA'
```

这个别名告诉`pytest`执行以下操作：

- 禁用警告
- `--instafail` 在错误发生时立即显示失败信息，而不是等到最后才显示
- `-rA` 生成一个简短的测试摘要信息

要使用此别名，你需要安装额外的依赖项：
```
pip install pytest-instafail
```


### 获取所有测试的列表

显示测试套件中的所有测试：

```bash
pytest --collect-only -q
```

显示特定测试文件中的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用的别名为：
```bash
alias pytc='pytest --disable-warnings --collect-only -q'
```

### 运行特定的测试模块

要运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定的测试

如果使用的是`unittest`，要运行特定的子测试，你需要知道包含那些测试的`unittest`类的名称。例如，它可能是这样的：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

在这里：

- `tests/test_optimization.py` 是包含测试的文件
- `OptimizationTest` 是包含所需测试函数的类名称
- `test_adam_w` 是具体需要执行的测试函数名称

如果文件中有多个类，你可以选择只运行某个类中的测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

这将运行该类中的所有测试。

如前所述，你可以在控制台中查看`OptimizationTest`类中包含了哪些测试：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

你可以通过关键字表达式来运行特定的测试。

例如，要运行仅包含“adam”关键词的所有测试：

```bash
pytest -k adam tests/test_optimization.py
```

逻辑运算符`and` 和 `or` 可以用来表示是否所有的关键词都需要匹配。`not` 可以用来说明否定。

例如，要运行除包含“adam”关键词以外的所有测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

或者，可以将两个模式结合起来：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，为了同时运行`test_adam_w`和`test_adam_w`这两个测试，可以使用：

```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

这里我们使用了`or`，因为我们希望任意一个关键词都能匹配以包括这两个测试。

如果你想要包含所有满足两个条件的测试，那么应该使用`and`：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 运行已修改的测试

你可以通过使用[pytest-picked](https://github.com/anapaulagomes/pytest-picked)来运行与未提交的文件或当前分支（根据Git）相关的测试。这是一个非常有用的特性，可以帮助你在提交更改之前快速验证它们没有破坏任何东西，因为它不会运行与你未接触的文件相关的测试。

```bash
pip install pytest-picked
```

然后运行：

```bash
pytest --picked
```

所有来自被修改的文件和文件夹内的测试都将被运行。

### 自动重新运行失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)提供了一个非常实用的功能，即检测所有失败的测试，并在你修改源代码后连续重新运行这些失败的测试，直到它们通过为止。这样，当你修复问题时，就不需要手动重启`pytest`。这个过程会一直重复，直到所有测试都通过，这时会再次进行完整的运行。

```bash
pip install pytest-xdist
```

要进入这种模式，运行：

```bash
pytest -f
```

或者：

```bash
pytest --looponfail
```

`looponfailroots`选项允许你指定哪些目录将被监视是否有文件变化。默认情况下，它会检查`transformers`和`tests`目录及其所有内容。如果你想改变这一点，可以通过设置配置选项来实现：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

这将在`setup.cfg`、`pytest.ini`或`tox.ini`文件中起作用。

[pytest-watch](https://github.com/joeyespo/pytest-watch)是实现相同功能的另一种实现。


### 跳过测试模块

如果要运行所有测试模块，但有一些你想排除在外，你可以通过提供一个要运行的测试的明确列表来排除某些模块。例如，要运行除了`test_modeling_*.py`测试之外的所有测试：

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态

在CI构建中或在隔离性很重要的环境中（比如速度优先的环境），缓存应被清空：

```bash
pytest --cache-clear tests
```

### 并行运行测试

正如前面提到的，`make test`使用`pytest-xdist`插件（`-n X`参数，例如`-n 2`用于运行2个并发作业）来并行运行测试。

`pytest-xdist`的`--dist=`选项允许控制测试的分组方式。`--dist=loadfile`会将同一文件中的测试放在同一个进程上。

由于测试顺序不同且不可预测，如果在使用`pytest-xdist`的情况下出现测试失败（这意味着我们有未发现的耦合测试），可以使用[pytest-replay](https://github.com/ESSS/pytest-replay)来重放测试，以便在相同的顺序下运行它们，这有助于减少随机性并可能帮助找到问题的根源。

### 测试顺序和重复

多次重复测试对于检测潜在的相互依赖问题和状态相关bug（如清理工作不正确）是非常有用的。此外，简单的多重重复有助于发现一些因随机性而暴露的问题。


#### 重复测试

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder）：

```bash
pip install pytest-flakefinder
```

然后运行每个测试多次（默认为50次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

脚注：一旦安装了`pytest-flakefinder`，它就会自动启用，因此不需要任何配置更改或命令行选项即可开始使用。

如文档中所解释的那样，每次运行都会打印出使用的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

如果给定的特定序列失败，你可以通过复制粘贴上述种子值来重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

在这种情况下，只有当使用完全相同的测试集（或不加选择的全部测试）时，才能依赖于种子值的确定性行为。一旦你开始手动缩小测试范围，你就不能再依靠种子值来保持一致的行为，而是必须手动列出它们并按确切的失败顺序排列，并指示`pytest`不要随机化它们，而是使用`--random-order-bucket=none`：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要禁用随机排序，即使在`pytest-flakefinder`安装之后，也可以这样做：

```bash
pytest --random-order-bucket=none
```

关于`--random-order-bucket`的不同模式，请参阅其[文档](https://github.com/jbasko/pytest-random-order）。

另一个随机化的替代方案是：[pytest-randomly](https://github.com/pytest-dev/pytest-randomly)。这个模块具有类似的功能/接口，但它不像`pytest-random-order`那样支持不同的分桶模式。它在安装后也会强制自己生效，这可能不是你所期望的。

#### 运行测试时的随机顺序

```bash
pip install pytest-random-order
```

重要提示：一旦安装了`pytest-random-order`，它会在安装后自动启用，这意味着你的测试将会按照随机的顺序运行，而不需要任何配置更改或命令行选项。

如文档中所述，这有助于检测测试之间的隐含依赖关系——其中一组的测试状态可能会影响另一组的结果。每当`pytest-random-order`被激活时，它将打印出使用的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

如果特定的序列失败，你可以通过复制粘贴上述种子值来重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

这将确保测试按与上次失败时相同的顺序运行。但是，一旦你开始手动限制测试的范围，你就不能依赖于种子的确定行为，而是需要显式地列出它们并按确切的顺序排列，并指示`pytest`不要随机化它们，而是使用`--random-order-bucket=none`：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

这将导致所有指定的测试按原样运行，不受随机性的影响。

关于`--random-order-bucket`的不同模式，请参阅其[文档](https://github.com/jbasko/pytest-random-order）。

另一个随机化的替代方案是：[pytest-randomly](https://github.com/pytest-dev/pytest-randomly)。这个模块具有类似的功能/接口，但它不像`pytest-random-order`那样支持不同的分桶模式。它在安装后也会强制自己生效，这可能不是你所期望的。

### 到GPU还是不到GPU

在配备GPU的系统上，可以通过设置环境变量`CUDA_VISIBLE_DEVICES=""`来模拟CPU-only模式：

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者，如果你的系统中有多块GPU，你可以指定哪一块应该被使用。例如，如果有GPU编号为`0`和`1`，你可以运行：

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

这对于在不同GPU之间分配任务很有用。

有些测试必须在CPU-only模式下运行，有些则在GPU或TPU上运行，还有一些则需要在多GPU环境下运行。下面列出了用于设置GPU要求的装饰器：

- `require_torch` - 这个测试将在Torch可用时运行
- `require_torch_gpu` - 作为`require_torch`的扩展，要求至少有一块GPU可用
- `require_torch_multi_gpu` - 作为`require_torch`的扩展，要求至少有两块GPU
- `require_torch_non_multi_gpu` - 作为`require_torch`的扩展，要求最多只有一块GPU
- `require_torch_up_to_2_gpus` - 作为`require_torch`的扩展，要求不超过两块GPU

让我们用一张表格来描述GPU需求的情况：

| n gpus | decorator                       |
|--------+----------------------------------|
| `>= 0` | `@require_torch`                 |
| `>= 1` | `@require_torch_gpu`             |
| `>= 2` | `@require_torch_multi_gpu`       |
| `< 2`  | `@require_torch_non_multi_gpu`   |
| `< 3`  | `@require_torch_up_to_2_gpus`    |

例如，这里是必须在使用两块或多块GPU的环境下运行的一个测试示例：

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

这些装饰器可以被嵌套使用：

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

一些装饰器，如`@parametrized`，会重写测试的名字，因此`@require_*`跳过装饰器需要放在它们的后面才能正常工作。这里有一个正确的用法例子：

```python no-style
from testing_utils import require_torch_gpu
from parameterized import parameterized

@parameterized.expand([
    ("negative", -1.5, -2.0),
    ("integer", 1, 1.0),
    ("large fraction", 1.6, 1),
])
@require_torch_gpu
def test_integration_foo():
```

这与`pytest.mark.parametrize`标记的使用略有不同，后者在`unittests`中不起作用，但在`pytest`测试中有效。

在测试内部：

- 有多少GPU可用：

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 分布式训练

`pytest`本身并不能直接处理分布式训练。如果尝试这样做，子进程实际上并不做正确的事情，而是在循环中运行`pytest`，这是无效的。然而，通过启动一个正常的进程，它可以管理输入输出管道，然后再从中派生出多个工作者。

以下是一些使用它的测试示例：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

要在这些测试中看到效果，你需要至少有两个GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

（`RUN_SLOW`是一个特殊的标志，用于通常绕过慢速测试的Hugging Face Transformers项目。）

### 输出捕获

在测试期间，发送到`stdout`和`stderr`的所有输出都被捕获。如果测试或准备方法失败，相应的捕获输出将与失败堆栈跟踪一起显示。

要禁用输出捕捉并获得正常的`stdout`和`stderr`流，可以使用`-s`或`--capture=no`选项：

```bash
pytest --color=no tests/utils/test_logging.py
```

要将测试结果发送到JUnit格式的输出：

```bash
py.test tests --junitxml=result.xml
```

### 颜色控制

要关闭颜色（例如，黄色在白色背景上不可读）：

```bash
pytest --color=no tests/utils/test_logging.py
```

### 将测试报告发送到在线粘贴板服务

创建每个测试失败的URL：

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这将提交测试运行信息到一个远程粘贴服务，并为每个失败提供一个URL。您可以选择性地选择要发送的测试，例如通过添加`-x`来发送单个特定失败的信息。

创建整个测试会话日志的URL：

```bash
pytest --pastebin=all tests/utils/test_logging.py
```








## 编写测试

大多数时候，如果将`pytest`和`unittest`结合在一个测试套件中是可以很好地工作的。你可以阅读[这里的文档](https://docs.pytest.org/en/stable/unittest.html)了解哪些特性在混合使用这两种框架时得到了支持。需要注意的是，大多数`pytest` fixture都不适用于`unittest`。同样，`parameterized`的参数化功能也不适用，但我们使用`parameterized`模块，它与`unittest`和`pytest`兼容。


### 参数化

经常会有需要多次运行同一个测试的需求，但每次使用不同的参数。这在`unittest`中可以通过`parameterized`模块轻松完成，而在`pytest`中则是通过`pytest.mark.parametrize`标记实现的。


#### Unittest中的参数化

下面的例子演示了如何在`unittest`中使用`parameterized`模块来进行参数化：

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert math.floor(input) == expected
```

现在，这个测试会被运行三次，每次`test_floor`的最后三个参数分别对应参数列表中的相应元素。

你可以通过`-k`过滤器来选择性地运行特定的子测试，即使是在`parameterized`的情况下。例如，你可以运行`negative`和`integer`子测试：

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或者，你可以运行除`negative`子测试外的所有子测试：

```bash
pytest -k "not negative" tests/test_mytest.py
```

除了使用`-k`过滤器外，你还可以找出每个子测试的确切名称，并通过它们的完整名称运行任何一个或所有子测试。

```bash
pytest test_this1.py --collect-only -q
```

这将列出：

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

所以你现在可以运行特定的子测试：

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

就像前面的例子一样。

`parameterized`模块不仅适用于`unittest`，也适用于`pytest`测试。

#### Pytest中的参数化

下面是如何在`pytest`中使用`pytest.mark.parametrize`标记来进行参数化：

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert math.floor(input) == expected
```

与`parameterized`相比，`pytest.mark.parametrize`标记创建的子测试名称略有不同。以下是它们的样子：

```bash
pytest test_this2.py --collect-only -q
```

这将列出：

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

所以你现在可以运行特定的子测试：

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

就像之前的例子一样。



### 文件和目录

在测试中，经常需要知道相对于当前测试文件的位置，但这并不是一件容易的事，因为测试可以从不同的目录调用，也可能位于深度不同的子目录中。`testing_utils.TestCasePlus`类解决了这个问题，它提供了对基本路径的简单访问，以及在这些路径上的便捷操作：

- `pathlib`对象（都已完全解析）：
  - `test_file_path` - 当前的测试文件路径，即`__file__`
  - `test_file_dir` - 包含当前测试文件的目录
  - `tests_dir` - `tests`测试套件的目录
  - `examples_dir` - `examples`测试套件的目录
  - `repo_root_dir` - 仓库的根目录
  - `src_dir` - `src`目录（即`transformers`子目录所在的目录）

- 字符串化的路径 - 与上面的相同，但这些返回的是作为字符串的路径，而不是`pathlib`对象：
  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

要开始使用这些属性，只需让测试继承自`testing_utils.TestCasePlus`。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果你不想使用`pathlib`对象，或者只需要路径作为字符串，你可以总是通过调用`str()`来转换`pathlib`对象，或者使用以`_str`结尾的访问器。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

#### 临时文件和目录

在测试中，创建独特的临时文件和目录非常重要，以确保并行运行的测试不会覆盖彼此的数据。我们还希望在测试结束时自动删除这些临时文件和目录。因此，使用像`tempfile`这样的包来解决这些问题是很重要的。

然而，在调试测试时，你可能需要能够查看临时文件或目录的内容，并且希望能够看到它们的精确路径，而不是每次运行时都有随机化的路径。`testing_utils.TestCasePlus`类在这方面特别有用。它是一个`unittest.TestCase`的子类，所以我们可以很容易地在测试模块中继承它。

以下是它的使用示例：

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

这段代码创建了一个唯一的临时目录，并将`tmp_dir`设置为它的位置。

- 创建一个独特的临时目录：

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir`将包含新创建的临时目录的路径。它将在测试结束后自动删除。

- 创建我自己选择的临时目录，确保它在测试开始时不为空，并且在测试结束后不将其清空。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

这在你想要监控特定的目录并在调试过程中保留其中的数据时非常有用。

- 你甚至可以自定义`before`和`after`参数的行为，从而得到以下行为之一：

  - `before=True`: 临时目录在每次测试开始时会自动清空。
  - `before=False`: 如果临时目录已经存在，现有的文件将不被移除。
  - `after=True`: 临时目录将在每次测试结束后自动删除。
  - `after=False`: 临时目录将在每次测试结束后保持不变。

脚注：为了安全地执行等价于`rm -r`的操作，仅允许子目录存在于项目的检出存储库中，因此意外地不会删除`/tmp`或其他重要的部分文件系统。也就是说，请始终传递以`./`开头的路径。

脚注：每个测试都可以注册多个临时目录，它们都将被自动删除，除非另有请求。


#### 临时sys.path覆盖

如果你需要在导入其他测试的过程中临时覆盖`sys.path`，你可以使用`ExtendSysPath`上下文管理器。例如：

```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试

有两种主要类型的跳过：

- **skip**: 这意味着你预计你的测试只有在某些条件得到满足时才会成功，否则`pytest`应该完全跳过运行这个测试。常见的例子包括在非Windows平台上跳过仅Windows平台支持的测试，或者在缺少外部资源（如数据库）时跳过依赖于它们的测试。
- **xfail**: 这意味着你预期测试会失败，原因可能是尚未实现的功能或尚未解决的bug。当一个预期的失败变成实际的成功（即`pytest.mark.xfail`标记下的测试通过了）时，它被称为`xpass`，并在测试总结中被报告。

这两者的重要区别在于，`skip`不会运行测试，而`xfail`则会。因此，如果代码中的bug会导致一些不良的状态，而这些状态会影响其他的测试，你应该避免使用`xfail`。

#### 实现

- 这里是如何无条件地跳过一个测试：

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

或者通过`pytest`的方式：

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

或者`xfail`的方式：

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

这里是如何基于内部的检查来决定是否跳过测试：

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

或者，你可以跳过整个模块：

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

或者`xfail`的方式：

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 这里是如何基于条件来跳过测试：

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

或者：

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

或者跳过整个模块：

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

更多细节、例子和使用方式见[这里](https://docs.pytest.org/en/latest/skipping.html)。



### 捕获输出

#### 捕获标准输出/错误输出

为了测试函数是否向`stdout`和/或`stderr`写入内容，测试可以访问这些流，使用`pytest`的[capsys系统](https://docs.pytest.org/en/latest/capture.html)。这是如何完成的：

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```

当然，大多数时候，`stderr`是通过异常抛出来产生的，所以在这种情况下，你需要使用`try/except`语句：

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

另一种捕获`stdout`的方法是使用`contextlib.redirect_stdout`上下文管理器：

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

为了方便调试测试问题，`CaptureStdout`、`CaptureStderr`和`CaptureStd`这三个上下文管理器类会自动处理所有这些情况，无论你是否使用`-s`选项运行`pytest`。

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

这里有一个完整的测试示例：

```python
from testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

如果你需要捕获`stderr`，使用`CaptureStderr`类：

```python
from testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

如果你需要同时捕获两个流，使用父级`CaptureStd`类：

```python
from testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

此外，为了便于调试，这些上下文管理器会自动在退出上下文时回放捕获的流。


#### 捕获日志记录流

如果你需要验证日志输出的内容，你可以使用`CaptureLogger`类：

```python
from transformers import logging
from testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### 测试环境变量

如果你想在特定的测试中测试环境变量的影响，你可以使用`mockenv`装饰器：

```python
from testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

在某些情况下，需要调用外部程序，这就需要设置`PYTHONPATH`环境变量以包含多个本地路径。`TestCasePlus`类可以帮助解决这个问题：

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

根据测试文件是在`tests`还是在`examples`测试套件中，`get_env`方法将正确设置`PYTHONPATH`，包括`src`目录，以确保测试是基于当前仓库进行的，同时也考虑到了已经在`os.environ`中设置的任何`PYTHONPATH`值。

这个helper方法创建了一个`os.environ`对象的副本，因此原始环境保持不变。


### 获取可复制的测试结果

在一些场景中，你可能需要去除测试过程中的随机因素，以获得完全确定的结果。为此，你需要固定随机数生成器的种子：

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
tf.random.set_seed(seed)
```

## 调试测试

要使调试器在遇到警告时中断，你可以这样做：

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```


## 创建多个pytest报告的大规模hack

下面是一个我在多年前为更好地理解CI报告中存在的问题所做的对`pytest`的巨大补丁。

要激活它，请在`tests/conftest.py`（如果没有的话，请创建它）中添加以下内容：

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

然后在运行测试套件时，添加`--make-reports=mytests`，如下所示：

```bash
pytest --make-reports=mytests tests
```

这将创建八个单独的报告：
```bash
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

这样，你就可以拥有每个类型报告的独立文件，而不是只有一个`pytest`的标准输出，里面混杂着各种信息。

这个功能原本就在`pytest`中，但我找不到一种简单的方法来提取它，所以我通过[testing_utils.py](testing_utils.py)中的monkey-patching覆盖实现了它。嗯，我曾经问过是否可以将这个贡献作为一个特性加入`pytest`，但我的提议没有被接受。
