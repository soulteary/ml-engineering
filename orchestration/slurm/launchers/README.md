# 在SLURM中使用单节点和多节点启动器
以下是完全的SLURM脚本，它们展示了如何将各种启动器与使用`torch.distributed`（但应很容易适应其他分布式环境）的软件集成。
- [`torchrun`启动器](torchrun-launcher.slurm) - 用于[PyTorch分布式](https://github.com/pytorch/pytorch)。
- [`加速`启动器](accelerate-launcher.slurm) - 用于[HF加速](https://github.com/huggingface/accelerate)。
- [`闪电`启动器](lightning-launcher.slurm) - 用于[闪电](https://lightning.ai/)（“PyTorch Lightning”和“Lightning Fabric”）。
- [`srun`启动器](srun-launcher.slurm) - 用于本机SLURM启动器 - 在这里我们必须手动设置`torch.distributed`期望的环境变量。
所有这些脚本都使用[torch-distributed-gpu-测试.py](../../../debug/torch-distributed-gpu-test.py)作为演示脚本来进行GPU上的分布式测试，你可以通过以下命令将其复制到当前目录：
```bash
cp ../../../debug/torch-distributed-gpu-test.py .
```
假设你克隆了此存储库。但是，你可以用任何你需要的东西替换它。