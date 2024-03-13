# 禁用 NVLink 基准测试

让我们比较在少量维基文本上训练的 GPT-2 语言模型训练。

结果如下：

| NVlink | 时间 |
| -----  | ---: |
| 是     | 101秒 |
| 否     | 131秒 |

你可以看到，使用 NVLink 的 GPU 可以完成训练大约快23%。 在第二个基准测试中，我们使用 `NCCL_P2P_DISABLE=1` 来告诉GPU不要使用 NVLink，而是改用 PCIe。

我们将使用[HF Transformer示例](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/examples/pytorch/language-modeling/run_clm.py)。

这里是完整的基准代码和输出：

```bash
# 有 NVLink 的 DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# 没有 NVLink 的 DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2个TITAN RTX 24GB + NVLink（每个GPU有2个NVLink）
软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`