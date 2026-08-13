import json
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils

# 从你的 models 文件夹导入网络
from models.pointnet2_sem_seg import get_model 

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 开启 cuDNN 自动基准测试 (针对卷积层的极致图优化)
        torch.backends.cudnn.benchmark = True
        
        # 2. 实例化网络 (纯随机权重，不加载模型，直接用于测速)
        self.model = get_model(num_classes=5, normal_channel=True).to(self.device)
        self.model.eval()
        
        ## ==========================================
        # 3. 开启 PyTorch 2.0 核心图优化
        # ==========================================
        #print("Enabling torch.compile graph optimization...")
        # 动态组批模式下，由于 Batch Size 是动态变化的：
        # 我们使用默认编译模式。如果使用 "reduce-overhead" (即 CUDA Graphs)，
        # 一旦 Batch 发生改变会重新生成 Graph 导致卡顿，因此使用默认模式最稳妥。
        #self.model = torch.compile(self.model)
        #print("Graph optimization configured!")

        # ==========================================
        # 4. 关键步骤：JIT 预热 (Warmup Compile)
        # ==========================================
        # 我们在 config.pbtxt 中配置了 preferred_batch_size: [4, 8, 16, 32]
        # 在这里提前用假数据让 PyTorch 把这些 Batch Size 的执行图编译好，防止线上超时！
        print("Starting model warmup compile for dynamic batching...")
        warmup_batches = [4, 8, 16, 32]
        for b in warmup_batches:
            print(f"Warmup compiling for Batch Size: {b} ...")
            # 假定输入的点数是 8192，通道是 6
            dummy_input = torch.randn(b, 6, 8192, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        print("Model warmup compile finished! Server is ready.")

    def execute(self, requests):
        responses = []
        for request in requests:
            # 1. 解析输入
            points_tensor = pb_utils.get_input_tensor_by_name(request, "points")

            # 使用 DLPack 直接把 Triton 显存指针包装成 PyTorch Tensor，完全避免 CPU 拷贝
            points_torch = from_dlpack(points_tensor.to_dlpack()).to(self.device)
            
            # 2. 维度对齐 [Batch, N, 6] -> [Batch, 6, N]
            points_torch = points_torch.transpose(1, 2).contiguous()
            
            # 3. 前向推理 (这里会自动调用你的 Triton 算子，并在外围执行图优化)
            with torch.no_grad():
                pred_out, _ = self.model(points_torch) 
            
            # 4. 包装输出 (返回格式为 [Batch, N, 5])-同样使用 DLPack 零拷贝，直接把 PyTorch 显存结果还给 Triton
            # 注意：to_dlpack 之前必须确保 Tensor 是连续内存 (contiguous)
            pred_out_contiguous = pred_out.contiguous()
            out_tensor = pb_utils.Tensor.from_dlpack(
                "segmentation_output", 
                to_dlpack(pred_out_contiguous)
            )
            
            # 5. 返回给客户端
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
            
        return responses

    def finalize(self):
        print('Cleaning up PointNet2 model...')