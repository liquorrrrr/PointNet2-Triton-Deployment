# PointNet2-Triton-Deployment
This project is based on the classic 3D point cloud network PointNet2, and uses OpenAI Triton language and NVIDIA Triton Server to achieve stable and reliable cloud deployment. It aims to address the pain points of "difficult operator compilation" and "slow model deployment" in the industrial application of 3D point cloud processing.
### 📊 算子性能超越 CUDA 基线

![算子性能对比图](assets/op_performance.png) 
*(图：OpenAI Triton 算子与原生 CUDA 算子的 Nsight Compute 底层耗时对比)*

**核心优化动机：** 
为了打造一条极速的云端部署路线，我们必须彻底避开 C++/CUDA 的复杂编译链以及繁琐的 TensorRT 算子插件编写。
如上图所示，我们使用 OpenAI Triton 重写了 PointNet++ 的核心非标算子（Farthest Point Sampling 与 Ball Query）。测试证明，**纯 Python 语境下的 Triton 算子性能，成功追平甚至超越了原作者手写的 CUDA 算子基线。**

---

### 🌩️ TIS 云端部署与极限压测

基于上述 Triton 优化算子，我们放弃了传统的 ONNX/TensorRT 路线，直接使用 NVIDIA TIS 的 **Python Backend** 实现了 100% Python 环境的原生挂载部署。

为了验证该链路的工业级可靠性，我们在单张 NVIDIA A10 显卡上，针对 80 并发进行了长达 5 分钟的高压稳态压测。在 TIS 的动态批处理（Dynamic Batching）加持下，服务表现极其稳定，完美榨干了物理算力：

**表 1：吞吐量与长尾延迟指标**

**表 2：硬件资源压榨指标 (Nsys/Ncu 监控)**

### 🚀 快速复现 (How to Run)

**前置要求：** 宿主机已安装 Docker 及支持 GPU 穿透的 [NVIDIA Container Toolkit](<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" title="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" target="_blank"><img src="/images/ext/file.png" alt="" style="width: 32px; height: 32px; vertical-align: middle;"></a>)。

**1. 构建定制镜像**
由于 TIS 官方镜像缺少 Triton 编译环境，需先通过 Dockerfile 补全依赖：
```bash
docker build -t custom_tis_pointnet:v1 .
