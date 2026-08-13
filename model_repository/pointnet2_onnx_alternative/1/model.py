import json
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from models.pointnet2_sem_seg import get_model 

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        self.model = get_model(num_classes=5, normal_channel=True).to(self.device)
        self.model.eval()
        
        print("Starting model warmup compile for dynamic batching...")
        warmup_batches = [4, 8, 16, 32]
        for b in warmup_batches:
            print(f"Warmup compiling for Batch Size: {b} ...")
            dummy_input = torch.randn(b, 6, 8192, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        print("Model warmup compile finished! Server is ready.")

    def execute(self, requests):
        responses = []
        for request in requests:
            points_tensor = pb_utils.get_input_tensor_by_name(request, "points")
            points_torch = from_dlpack(points_tensor.to_dlpack()).to(self.device)
            points_torch = points_torch.transpose(1, 2).contiguous()
            
            with torch.no_grad():
                pred_out, _ = self.model(points_torch) 
   
            pred_out_contiguous = pred_out.contiguous()
            out_tensor = pb_utils.Tensor.from_dlpack(
                "segmentation_output", 
                to_dlpack(pred_out_contiguous)
            )
            
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
            
        return responses

    def finalize(self):
        print('Cleaning up PointNet2 model...')
