# 检查点

- [`torch-checkpoint-convert-to-bf16`](./torch-checkpoint-convert-to-bf16) - 将现有的fp32 PyTorch检查点转换为bf16。如果找到[SafeTensors](https://github.com/huggingface/safetensors/)，也会对其进行转换。应该很容易适应其他类似的情况。

- `torch-checkpoint-shrink.py` - 此脚本修复了由于某种原因存储的张量的大小大于保存时其视图大小的检查点。它克隆当前视图并重新保存它们，仅使用当前视图的存储。

