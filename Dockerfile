FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.1.2 \
    triton==2.1.0 \
    ninja \
    numpy \
    sympy

WORKDIR /opt/tritonserver
