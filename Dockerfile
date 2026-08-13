# 基于官方 TIS 镜像
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# 使用清华源一次性安装所有依赖
# 默认的 torch 2.1.2 已经自带 CUDA 支持，不需要加 +cu121 后缀
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.1.2 \
    triton==2.1.0 \
    ninja \
    numpy \
    sympy

# 设置工作目录
WORKDIR /opt/tritonserver