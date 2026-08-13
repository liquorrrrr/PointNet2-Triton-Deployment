# Docker Startup Command
 docker run --gpus all \
  --rm \
  --name my_triton_server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /root/model_repository:/models \
  pointnet2-tis:latest \
  tritonserver --model-repository=/models

# Stress Testing Command (Another Docker acts as the client to send requests)
# Short-term 
root@iZbp12hp1ex5fa5lz3fug6Z:~# docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
  perf_analyzer -m pointnet2_onnx_alternative \
  -u localhost:8000 \
  --concurrency-range 10:100:10 \
  --shape points:8192,6 \
  -p 5000 # Test duration
# Long-term
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
  perf_analyzer -m pointnet2_onnx_alternative \
  -u localhost:8000 \
  --concurrency-range 80:80 \
  --shape points:8192,6 \
  -p 300000 # Test duration

# 8002 Port for Monitoring GPU Usage Rate
curl localhost:8002/metrics
