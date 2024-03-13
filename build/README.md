# 书本构建

重要提示：这仍然是一个正在进行中的工作——它基本上可以运行，但需要一些样式表的工作来使pdf看起来更美观。预计在几周内完成。

这份文件假设您正在项目的根目录下工作。

## 安装要求

1. 安装用于书籍构建的Python包
   ```bash
   pip install -r build/requirements.txt
   ```

2. 下载[Prince XML](https://www.princexml.com/download/)的免费版本。它被用来构建这本电子书的PDF版本。

## 构建HTML

```bash
make html
```

## 构建PDF

```bash
make pdf
```

它会首先构建html目标，然后使用它来构建pdf版本。

## 检查链接和锚点

要验证所有本地链接和带有锚点的链接是否有效，请运行：
```bash
make check-links-local
```

如果也想同时检查外部链接
```bash
make check-links-all
```
请谨慎使用后者以避免因频繁访问服务器而被封禁。

## 将MD文件或文件夹移动并调整相对链接

例如：`slurm` => `orchestration/slurm`
```bash
src=slurm
dst=orchestration/slum

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local
```

## 缩放图像

当包含的图像过大时，将它们缩小一点：

```bash
mogrify -format png -resize 1024x1024\> *png
```