# 从零开始重新训练HF Hub模型，使用微调示例

Hugging Face的Transformers提供了非常棒的微调示例 https://github.com/huggingface/transformers/tree/main/examples/pytorch，几乎涵盖了所有模式，并且这些例子可以直接运行。

但是，如果你想从零开始重新训练而不是微调呢？

这里有一个简单的技巧来实现这一点。

我们将使用`facebook/opt-1.3b`并计划使用bf16训练方案作为这里的示例：

```bash
cat << EOF > prep-bf16.py
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModel.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOF
```

现在运行：

```bash
python prep-bf16.py
```

这将创建一个文件夹：`opt-1.3b-bf16`，其中包含您需要的一切来从头开始训练模型。换句话说，你有一个预训练样式的模型，除了它的初始化已经完成之外，还没有进行任何培训。

调整上面的脚本以使用`torch.float16`或`torch.float32`如果您打算改用它们的话。

现在你可以像往常一样继续微调这个保存的模型：

```bash
python -m torch.distributed.run \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

关键条目是：
```bash
--model_name_or_path opt-1.3b-bf16
```

其中`opt-1.3b-bf16`是你刚刚在上一步中生成的本地目录。

其他超参数有时可以在论文或随附模型的文档中找到。

简而言之，这种食谱允许您使用微调示例来重新训练[HF中心](https://huggingface.co/models)上可用的任何模型。