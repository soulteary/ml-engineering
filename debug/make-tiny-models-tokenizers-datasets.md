请将以下文本翻译成中文：

# 使用微型模型、标记器和数据集实现更快调试和开发

如果您在调试问题并使用完整大小的模型和标记器进行开发，那么您可能不是以非常高效的方式工作。不仅解决问题变得更加困难，而且程序重启并达到预期状态所需的时间可能会非常长——累积起来，这会严重消耗一个人的动力和工作效率，更不用说解决问题的过程会更慢，甚至根本无法解决。

解决方案很简单：

**除非您正在测试模型的质量，否则始终使用潜在的微型随机模型和标记器。**

此外，大型模型通常需要大量的资源，这些资源通常是昂贵的，并且可以使调试过程变得极其复杂。例如，任何调试器都可以处理单个进程，但如果您的模型太大而不能容纳在一个进程中且需要某种形式的[并行化](../training/model-parallelism)，这将涉及多个进程，大多数调试器要么会出现故障，要么会给您提供不必要的信息。理想的发展环境是一个进程和一个保证适合最便宜的单个小消费者GPU的最小微型模型。即使没有周围的GPU，您也可以免费使用[Google Colab](https://colab.research.google.com/)来进行紧急开发。

因此，更新的ML开发格言变成了：

- 模型越大，最终产品的生成效果越好；
- 模型越小，最终产品训练启动的速度就越快。

脚注：最近的研究表明，更大的并不总是更好，但这足以传达我交流的重要性。

一旦代码正常运行，请切换到真实模型以测试生成的质量。但在这种情况下，仍然首先尝试产生高质量结果的最小模型。只有当您看到生成的大部分是正确的时，才使用最大的模型来验证您的工作是否完美。

## 制作微型模型

重要提示：鉴于它们的流行度和设计良好的简单API，我将讨论HF [`transformers`](https://github.com/huggingface/transformers/)模型。但是，相同的原理可以应用于任何其他模型。

TL;DR：制作一个HF `transformers`微型模型是非常简单的：

1. 从全尺寸模型的配置对象获取配置对象
2. 将隐藏大小和其他一些参数缩减至合理范围
3. 根据缩小后的配置创建新模型
4. 保存该模型即可！

脚注：重要的是要记住，这将生成一个随机的模型，因此不要期望其输出有任何质量。

现在让我们通过实际的代码来看看如何将[“google/mt5-small”](https://huggingface.co/google/mt5-small/tree/main)转换为其对应的微型随机版本。

```
from transformers import MT5Config, MT5ForConditionalGeneration

mname_from = "google/mt5-small"
mname_very_small = "mt5-tiny-random"

config = MT5Config.from_pretrained(mname_from)

config.update(dict(
    d_model=64,
    d_ff=256,
))
print("new config", config)

very_small_model = MT5ForConditionalGeneration(config)
print(f"num of params {very_small_model.num_parameters()}")

very_small_model.save_pretrained(mname_very_small)
```

如您所见，这是非常简单的操作。您甚至可以在巨大的176B参数模型上执行此操作，例如[BLOOM-176B](https://huggingface.co/bigscience/bloom)，因为除了其配置对象之外，您实际上不会加载原始模型中的任何内容。

在修改配置之前，您可以转储原始参数并选择缩减更多维度。例如，减少层数会使模型变得更小，更容易调试。所以这里是如何做进一步的缩减，以便我们可以从[“google/mt5-small”](https://huggingface.co/google/mt5-small/tree/main)开始：

```
config.update(dict(
    vocab_size=keep_items+12,
    d_model=64,
    d_ff=256,
    d_kv=8,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=4,
    relative_attention_num_buckets=32,
))
```

原始[“google/mt5-small”](https://huggingface.co/google/mt5-small/tree/main)模型文件为1.2 GB。使用上述更改后，我们将它减小到了126 MB。

我们还可以通过在保存前将其转换为fp16（或bf16）进一步减小它的体积：

```
very_small_model.half()
very_small_model.save_pretrained(mname_very_small)
```
这样可以将文件大小减半至约64 M。

因此，您可以从这里停止，并且您的程序将启动得快得多。

还有一步可以帮助使模型真正小型化。

到目前为止，我们所缩减的是模型本身，但还没有触及标记器的词汇表大小，而这正是定义我们的词汇量大小的关键因素。

## 制作微型标记器

这个任务根据底层标记器的不同，可以是相对简单的流程，也可以是更加复杂的练习。

下面介绍的食谱来自Hugging Face的几位出色的标记器专家，我对其进行了调整以满足我的需求。

您可能不需要理解它们的工作原理直到实际需要它们为止，因此如果这是您第一次阅读本章，您可以安全地跳过这些部分直接到达[使用微型模型与微型标记器](#making-a-tiny-model-with-a-tiny-tokenizer)。

### Anthony Moi 的版本

[Anthony Moi](https://github.com/n1t0)的标记器缩减脚本：

```
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer

vocab_keep_items = 5000
mname = "microsoft/deberta-base"

tokenizer = AutoTokenizer.from_pretrained(mname, use_fast=True)
assert tokenizer.is_fast, "This only works for fast tokenizers."
tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
vocab = tokenizer_json["model"]["vocab"]
if tokenizer_json["model"]["type"] == "BPE":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
    merges = tokenizer_json["model"]["merges"]
    new_merges = []
    for i in range(len(merges)):
        a, b = merges[i].split()
        new_token = "".join((a, b))
        if a in new_vocab and b in new_vocab and new_token in new_vocab:
            new_merges.append(merges[i])
    tokenizer_json["model"]["merges"] = new_merges
elif tokenizer_json["model"]["type"] == "Unigram":
    new_vocab = vocab[:vocab_keep_items]
elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
else:
    raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
tokenizer_json["model"]["vocab"] = new_vocab
tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
tokenizer.save_pretrained(".")
```

后来我发现GPT-2的特殊标记`""`被巧妙地藏在词汇表的最后，因此它会丢失导致代码崩溃。所以我用黑客方式把它放回了原处：

```
if "gpt2" in mname:
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items-1 }
        new_vocab[""] = vocab_keep_items-1
    else:
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
```

### Lysandre Debut 的版本

[Lysandre Debut](https://github.com/LysandreJik)的基于`train_new_from_iterator`的缩减方法：

```
from transformers import AutoTokenizer

mname = "microsoft/deberta-base" # 或者任何带有快速标记器的检查点。
vocab_keep_items = 5000

tokenizer = AutoTokenizer.from_pretrained(mname)
assert tokenizer.is_fast, "This only works for fast tokenizers."
tokenizer.save_pretrained("big-tokenizer")
# 应该是一个生成文本列表的生成器。
training_corpus = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]
new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_keep_items)
new_tokenizer.save_pretrained("small-tokenizer")
```
但是这个方法需要一个培训语料库，所以我有了一个想法来欺骗系统并在自己的原始词汇表基础上训练新的标记器，从而得到了：

```
from transformers import AutoTokenizer

mname = "microsoft/deberta-base"
vocab_keep_items = 5000

tokenizer = AutoTokenizer.from_pretrained(mname)
assert tokenizer.is_fast, "This only works for fast tokenizers."
vocab = tokenizer.get_vocab()
training_corpus = [ vocab.keys() ] # 应该是一个生成器，生成文本列表。
new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_keep_items)
new_tokenizer.save_pretrained("small-tokenizer")
```

几乎完美，只是现在它没有任何关于每个单词/字符频率的信息（大多数标记器就是这样计算他们的词汇量的）。如果我们确实需要这个信息，我们可以通过让每个键出现`len(vocab) - ID times`来解决这个问题，即：

```
training_corpus = [ (k for i in range(vocab_len-v)) for k,v in vocab.items() ]
```
这将大大增加脚本完成时间。

但对于微型模型的测试目的来说，频率并不重要。

### 手动编辑标记器文件的方法

某些标记器可以通过直接编辑文件来实现词汇量缩减，例如，让我们将Llama2的标记器缩减到3k项：

```
# Shrink the orig vocab to keep things small (just enough to tokenize any word, so letters+symbols)
# ElectraTokenizerFast is fully defined by a tokenizer.json, which contains the vocab and the ids,
# so we just need to truncate it wisely
import subprocess
import shlex
from transformers import LlamaTokenizerFast

mname = "meta-llama/Llama-2-7b-hf"
vocab_keep_items = 3000

tokenizer_fast = LlamaTokenizerFast.from_pretrained(mname)
tmp_dir = f"/tmp/{mname}"
tokenizer_fast.save_pretrained(tmp_dir)
# resize tokenizer.json (vocab.txt will be automatically resized on save_pretrained)
# perl  -0777 -pi -e 's|(2999).*|$1},"merges": []}|$msg' tokenizer.json # 0-indexed, so vocab_keep_items-1!
closing_pat = '},"merges": []}}'
cmd = (f"perl -0777 -pi -e 's|({vocab_keep_items-1}).*|$1{closing_pat}|$msg' {tmp_dir}/tokenizer.json")
#print(f"Running:\n{cmd}")
result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
# reload with modified tokenizer
tokenizer_fast_tiny = LlamaTokenizerFast.from_pretrained(tmp_dir)
tokenizer_fast_tiny.save_pretrained(".")
```
请记得，结果只适用于功能性测试，而不是质量工作。

这里是完整的[make_tiny_model.py](https://huggingface.co/stas/tiny-random-llama-2/blob/main/make_tiny_model.py)脚本示例，其中包括了模型和标记器的缩减。

### SentencePiece 词汇量缩减

首先克隆SentencePiece到父目录中：
```
git clone https://github.com/google/sentencepiece
```
然后进行缩减：
```
# workaround for fast tokenizer protobuf issue, and it's much faster too!
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import XLMRobertaTokenizerFast

mname = "xlm-roberta-base"

# Shrink the orig vocab to keep things small
vocab_keep_items = 5000
tmp_dir = f"/tmp/{mname}"
vocab_orig_path = f"{tmp_dir}/sentencepiece.bpe.model" # 这个名称可能会有所不同
vocab_short_path = f"{tmp_dir}/spiece-short.model"
# HACK: need the sentencepiece source to get sentencepiece_model_pb2, as it doesn't get installed
sys.path.append("../sentencepiece/python/src/sentencepiece")
import sentencepiece_model_pb2 as model
tokenizer_orig = XLMRobertaTokenizerFast.from_pretrained(mname)
tokenizer_orig.save_pretrained(tmp_dir)
with open(vocab_orig_path, 'rb') as f: data = f.read()
# adapted from https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/
m = model.ModelProto()
m.ParseFromString(data)
print(f"Shrinking vocab from original {len(m.pieces)} dict items")
for i in range(len(m.pieces) - vocab_keep_items): _ = m.pieces.pop()
print(f"new dict {len(m.pieces)}")
with open(vocab_short_path, 'wb') as f: f.write(m.SerializeToString())
m = None

tokenizer_fast_tiny = XLMRobertaTokenizerFast(vocab_file=vocab_short_path)
tokenizer_fast_tiny.save_pretrained(".")
```

### 制作具有微型标记器的微型模型

现在我们已经能够将词汇量缩减到标记器允许的最大最小值，也就是说，我们需要足够的令牌来覆盖目标字母表和特殊字符，通常3-5k个令牌就足够了。有时你可以让它变得更小，毕竟原始ASCII字符集中只有128个字符。

如果我们继续前面的MT5代码片段并添加标记器缩减代码，我们会得到这个脚本[mt5-make-tiny-model.py](https://huggingface.co/stas/mt5-tiny-random/blob/main/mt5-make-tiny-model.py)，并且在运行之后，我们的最终模型文件真的非常小——仅3.34 MB！正如您看到的，脚本还包含代码来验证模型是否能够与修改后的标记器一起工作。结果将是垃圾，但意图是对新的模型和标记器进行功能性测试。

这里还有一个例子[fsmt-make-super-tiny-model.py](https://huggingface.co/stas/tiny-wmt19-en-ru/blob/main/fsmt-make-super-tiny-model.py)，在这里您可以看到我是如何从头开始创建全新的微型词汇表的。

我也建议始终将与模型相关的所有构建脚本与模型存储在一起，以便您能快速修复问题或制作类似的模型版本。

同样值得注意的是，由于HF `transformers`需要微型模型用于他们的测试，你很可能会在他们内部测试仓库下找到已经存在的对应于每种架构的微型模型，大部分来自于
https://huggingface.co/hf-internal-testing （尽管他们没有包括制作这些模型的代码，但现在您可以根据这些笔记推断出其中的奥秘）。

另一个提示：如果您需要稍微不同的微型模型，您也可以从一个已有的微型模型开始并对其进行适应。既然它是随机的，重新调整大小主要是关于获得合适的维度。例如，如果找到的微型模型有2层但您需要8层，只需按照较大的维度重新保存它即可。

## 制作微型数据集

类似于模型和标记器，拥有一个方便使用的微型数据集对于加快开发的启动速度也非常有用。虽然这对质量测试没有帮助，但它非常适合于快速启动您的程序。

脚注：使用微型数据集的影响不会像使用微型模型那样显著，如果您使用预索引的Arrow文件格式的数据集，因为它们已经是超级快的。但是，假设您希望迭代器在一个epoch内完成10步。相反编写代码来截断数据集，您可以使用一个小型的数据集代替。

这个过程制作微型数据集比解释起来更为复杂，因为它取决于原始数据集的构建者，而这些构建者的做法可能是截然不同的。然而，概念仍然是相当简单的：

1. 克隆整个数据集Git仓库
2. 将原始数据的tarball替换为一个仅包含少量样本的小型tarball
3. 保存即可！

以下是一些例子：

- [stas/oscar-en-10k](https://huggingface.co/datasets/stas/oscar-en-10k/blob/main/oscar-en-10k.py)
- [stas/c4-en-10k](https://huggingface.co/datasets/stas/c4-en-10k/blob/main/c4-en-10k.py)
- [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k/blob/main/openwebtext-10k.py)

在上述所有案例中，我从原始tarball中提取了前10k条记录，再次打包成tarball，使用了较小的tarball，并完成了数据集repo的建设。其余的构建脚本基本保持不变。

还有一些合成数据集的例子，其中我没有简单地缩减原始tarball，而是解包了它，手动选择了代表性的示例，然后编写了脚本来基于那些少数的代表性样例构建任意长度的新数据集：
- [stas/general-pmd-synthetic-testing](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/blob/main/general-pmd-synthetic-testing.py) 和相应的[解包器](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/blob/main/general-pmd-ds-unpack.py)
- [stas/cm4-synthetic-testing](https://huggingface.co/datasets/stas/cm4-synthetic-testing/blob/main/cm4-synthetic-testing.py) ——以及相应的[解包器](https://huggingface.co/datasets/stas/cm4-synthetic-testing/blob/main/m4-ds-unpack.py)

在这些案例中，解包器是将每个复杂的多记录样本展开到各自的子目录中，以便现在您可以轻松地进行调整。您可以为图像添加或删除它们，缩短文本记录等。您还会注意到我在缩减大型图像的同时保留比例，使其变为32x32的大小，因此我又一次应用了重要的原则——在所有不影响目标代码库要求的维度上做到微型化。

主要脚本使用那个结构来构建任何所需长度的数据集。

这里是为[stas/general-pmd-synthetic-testing](https://huggingface.co/datasets/stas/general-pmd-synthetic-testing/)部署这些脚本的一些说明：

```
# prep dataset repo
https://huggingface.co/new-dataset => stas/general-pmd-synthetic-testing
git clone https://huggingface.co/datasets/stas/general-pmd-synthetic-testing
cd general-pmd-synthetic-testing

# select a few seed records so there is some longer and shorter text, records with images and without,
# a few variations of each type
rm -rf data
python general-pmd-ds-unpack.py --dataset_name_or_path \
general_pmd/image/localized_narratives__ADE20k/train/00000-00002 --ids 1-10 --target_path data

cd data

# shrink to 32x32 max, keeping ratio
mogrify -format jpg -resize 32x32\> */*jpg

# adjust one record to have no image and no text
cd 1
rm image.jpg text.txt
touch image.null text.null
cd -

cd ..

# create tarball
tar -cvzf data.tar.gz data

# complete the dataset repo
echo "This dataset is designed to be used in testing. It's derived from general-pmd/localized_narratives__ADE20k \
dataset" >> README.md

# test dataset
cd ..
datasets-cli test general-pmd-synthetic-testing/general-pmd-synthetic-testing.py --all_configs
```

我也强烈建议始终将构建脚本的副本与数据集相关联，以便您能快速修复问题或制作类似版本的数据集。

就像微型模型一样，您也会发现许多微型数据集位于https://huggingface.co/hf-internal-testing之下。

## 结论

虽然在机器学习领域我们有数据集、模型和标记器这三个要素，每一个都能被制作成微型化的形式，从而实现超快的开发和低资源要求，但如果您来自不同的行业，您可以将这里的思想扩展到您特定领域的工件/有效载荷。

## 本章节中所有脚本的中文备份

如果在阅读本文档时，链接到的外部脚本不可访问或在Hugging Face Hub上遇到问题，这里提供了[本地对这些脚本的最新备份](./tiny-scripts/)。

注意给自我：为了更新这些链接在本章中引用的所有文件的本地备份，运行以下命令：
```
perl -lne 'while (/(https.*?.py)\)/g) { $x=$1; $x=~s/blob/raw/; print qq[wget $x] }' make-tiny-models.md
```

