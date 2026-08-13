# docker启动命令
 docker run --gpus all \
  --rm \
  --name my_triton_server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /root/model_repository:/models \
  pointnet2-tis:latest \
  tritonserver --model-repository=/models

# 8002端口抓GPU使用率等情况
curl localhost:8002/metrics

# 压测命令（另一个docker作为客户端发请求）
# 短压
root@iZbp12hp1ex5fa5lz3fug6Z:~# docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
  perf_analyzer -m pointnet2_onnx_alternative \
  -u localhost:8000 \
  --concurrency-range 10:100:10 \
  --shape points:8192,6 \
  -p 5000
# 长压
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk \
  perf_analyzer -m pointnet2_onnx_alternative \
  -u localhost:8000 \
  --concurrency-range 80:80 \
  --shape points:8192,6 \
  -p 300000