# 资源

## 有用的汇编

- @StellaAthena 在 GitHub 上创建了名为“Common LLM Settings”的电子表格（链接：https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0）。这个电子表格对于即将开始新的 LLM 训练的人来说是一个非常有价值的资源，因为它列出了已知的 LLM 训练情况。

- 我几年前开始整理有关模型在何种数据类型上进行训练的信息（链接：https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671）。虽然目前只包含少数几个模型的信息，但如果您正在研究数据类型的影响，这些信息仍然有用。我使用这些信息来尝试编写一个自动检测模型是否在 bfloat16、fp16 或 fp32 中预训练的脚本，以及相关的浮点数属性比较研究（链接：https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb 和 https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb）。

## 公开可用的 LLM/VLM 训练日志

LLM/VLM 的训练日志和记录是学习如何处理训练不稳定性和选择良好超参数的最佳来源之一。如果您知道任何未在此列表中的公共 LLM/VLM 训练日志，请告诉我或在 PR 中添加它们。谢谢！

以下列表按年份分组，无特定顺序。

### 2021

- 大科学项目（BigScience）在 BLOOM 发布前的 108B 训练实验（2021 年）（链接：https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md 和 https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide）。备份链接：https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md 和 https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide。

### 2022

- 大科学项目的 BLOOM-176B 训练（2022 年）（链接：https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md 和 https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/）。备份链接：https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md 和 https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/。

- Meta 的 OPT-175B 训练（2022 年）（链接：https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles）。视频介绍：https://www.youtube.com/watch?v=p9IxoSkvZ-M。备份链接：https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles。

- THUDM 的 GLM-130B 训练（2022 年）（英文版链接：https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md；中文版链接：https://github.com/THUDM/GLM-130B/blob/main/logs/main-log.md）。备份链接：https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md 和 https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md。

### 2023

- Hugging Face 的 IDEFICS-80B 多模态训练（Flamingo 复现）（2023 年）（学习日志：https://github.com/huggingface/m4-logs/blob/master/memos/README.md；训练纪事：https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md）。备分链接：https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md 和 https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md。

- BloombergGPT 50B LLM，见论文《BloombergGPT: A Large Language Model for Finance》（2023 年）（论文链接：https://arxiv.org/abs/2303.17564）。

