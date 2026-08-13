# PointNet2-Triton-Deployment
This project is based on the classic 3D point cloud network PointNet2, and uses OpenAI Triton language and NVIDIA Triton Server to achieve stable and reliable cloud deployment. It aims to address the pain points of "difficult operator compilation" and "slow model deployment" in the industrial application of 3D point cloud processing.
### 📊 The performance of the operator exceeds the CUDA baseline.

![算子性能对比图](assets/op_performance.png) 
*(Figure: Comparison of underlying execution time between OpenAI Triton FPS operator and native CUDA operator in Nsight Compute)*

![算子性能对比图](assets/op_performance.png) 
*(Figure: Comparison of underlying execution time between OpenAI Triton BQ operator and native CUDA operator in Nsight Compute)*

**Core optimization motivation：** 
In order to create a fast cloud deployment route, we completely avoided the complex compilation chain of C++/CUDA and the cumbersome TensorRT operator plugin writing. We rewrote the core non-standard operators (Farthest Point Sampling and Ball Query) of PointNet++ using OpenAI Triton. As shown in the above figure, the throughput and time consumption of our FPS and BQ operators under different Batch sizes and input point cloud numbers N are superior to those of the CUDA operators. Specific test details can be found in assets/1.pdf. The test results prove that the performance of Triton operators in a pure Python environment has successfully matched and even surpassed the baseline of the CUDA operators written by the original author. **

---

### 🌩️ TIS Cloud Deployment and Extreme Load Testing

Based on the above Triton optimization operator, we abandoned the traditional ONNX/TensorRT approach and directly used NVIDIA TIS's **Python Backend** to achieve native mounting deployment in a 100% Python environment.

To verify the industrial-grade reliability of this link, we concurrently ran two instance models on a single NVIDIA A10 graphics card. First, we conducted a 5-second short-term stress test for 10-100 concurrent instances, and the results are shown in Table 1; then, in a scenario where the business SLA was designed with a client delay of less than 1.5 seconds, we conducted a 5-minute long-term high-pressure concurrent performance stress test for 80 concurrent instances, and the results are shown in Table 2. Through the Prometheus Metrics probe, we captured the real-time hardware status of TIS. The results showed: - **GPU Utilization (Utilization): Stabilized at 97%**, perfectly extracting the computing power of the A10, - **Power Usage Threshold (Power Usage): Maintained at 144.9W / 150W extreme power consumption**, the graphics processing unit (SM) was in an extremely dense and fully loaded working area. - **Long-term Failure Rate (Failure Rate): Maintained at 0 failures**. The dynamic concatenation mechanism (Avg Batch Size ≈ 24) operated smoothly, without OOM or request avalanche accumulation. This verified the high availability of this deployment solution.

**Table 1: Short-term Stepwise Concurrent Stress Testing**

| Concurrency | throughput | latency |
| :---: | :---: | :---: |
| 10 | 106.87 infer/sec | 93313 usec |
| 20 | 106.321 infer/sec | 187028 usec |
| 30 | 105.766 infer/sec | 283234 usec |
| 40 | 103.926 infer/sec | 376484 usec |
| 50 | 105.764 infer/sec | 476418 usec |
| 60 | 102.928 infer/sec | 572095 usec |
| 70 | 104.419 infer/sec | 663141 usec |
| **80** | **104.871 infer/sec** | **758578 usec** |
| 90 | 104.986 infer/sec | 854484 usec |
| 100 | 105.43 infer/sec | 947724 usec |

**Table 2: Long-term High Voltage Concurrent Extreme Performance Stress Test**

| Client Metric | Value | Server Metric | Value |
| :--- | :--- | :--- | :--- |
| **Request count** | 113175 | **Inference count** | 113175 |
| **Throughput** | 104.78 infer/sec | **Execution count** | 4255 |
| **Avg latency** | 763287 usec (standard deviation 10079 usec) | **Successful request count** | 113175 |
| **p50 latency** | 762714 usec | | |
| **p90 latency** | 857837 usec | | |
| **p95 latency** | 899918 usec | **Avg request latency** | 759977 usec (overhead 447 usec + queue 217111 usec + compute input 1406 usec + compute infer 537271 usec + compute output 3740 usec) |
| **p99 latency** | 986845 usec | | |
| **Avg HTTP time** | 763279 usec (send/recv 1076 usec + response wait 762203 usec) | | |

### 🚀 How to Run

**Prerequisites:** 
The host machine must have Docker installed along with the [NVIDIA Container Toolkit](<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" title="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" target="_blank"><img src="/images/ext/file.png" alt="" style="width: 32px; height: 32px; vertical-align: middle;"></a>) to enable GPU passthrough.

**1. Build the Custom Image**
Since the official TIS image lacks the Triton compilation environment, we first resolve dependencies via the Dockerfile:
```bash
docker build -t custom_tis_pointnet:v1 .
```

**2. Start the Service**
```bash
docker run --gpus all --rm --net=host --privileged \
  -v $(pwd)/model_repository:/models \
  custom_tis_pointnet:v1 \
  tritonserver --model-repository=/models
```

**3. Run Concurrent Stress Test**
```bash
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
  perf_analyzer -m pointnet2_onnx_alternative -u localhost:8000 \
  --concurrency-range 80:80 --shape points:8192,6 -p 300000
```

---

### 🙏 Acknowledgements

Special thanks to [erikwijmans](https://github.com/erikwijmans/Pointnet2_PyTorch) for the outstanding open-source project `Pointnet2_PyTorch`. The PyTorch network architecture and the baseline CUDA operators used for comparison in this project are highly inspired by and referenced from this repository.
