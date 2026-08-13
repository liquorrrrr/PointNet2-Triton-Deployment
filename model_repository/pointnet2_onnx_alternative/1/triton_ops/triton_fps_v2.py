import torch
import triton
import triton.language as tl

@triton.jit
def _fps_v2_kernel(
    xyz_ptr, temp_ptr, idx_ptr,
    B, N, M,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    
    xyz_batch = xyz_ptr + batch_idx * N * 3
    temp_batch = temp_ptr + batch_idx * N
    idx_batch = idx_ptr + batch_idx * M

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 1. 循环外一次性 Load 全局显存
    x = tl.load(xyz_batch + cols * 3 + 0, mask=mask, other=0.0)
    y = tl.load(xyz_batch + cols * 3 + 1, mask=mask, other=0.0)
    z = tl.load(xyz_batch + cols * 3 + 2, mask=mask, other=0.0)

    # 2. 初始化局部最短距离，同样不需要 temp_batch 频繁写回
    dist_local = tl.full((BLOCK_SIZE,), 1e10, dtype=tl.float32)

    farthest_idx = 0
    tl.store(idx_batch, farthest_idx)


    for step in range(1, M):
        # 3. 从局部已加载的 x,y,z 中提取最远点的坐标，而非使用 tl.load 读全局显存
        # 注：Triton 允许我们使用 sum + mask 或多播形式读取标量，例如：
        f_x = tl.sum(tl.where(cols == farthest_idx, x, 0.0), axis=0)
        f_y = tl.sum(tl.where(cols == farthest_idx, y, 0.0), axis=0)
        f_z = tl.sum(tl.where(cols == farthest_idx, z, 0.0), axis=0)

        dx = x - f_x
        dy = y - f_y
        dz = z - f_z
        dist = dx * dx + dy * dy + dz * dz

        dist_local = tl.minimum(dist_local, dist)

        search_dist = tl.where(mask, dist_local, -1e10)
        farthest_idx = tl.argmax(search_dist, axis=0)

        tl.store(idx_batch + step, farthest_idx)

def triton_fps_v2(xyz, npoint):
    assert xyz.is_cuda, "Input must be on GPU"
    xyz = xyz.contiguous() # Triton 强要求显存连续
    B, N, C = xyz.shape
    device = xyz.device

    idx = torch.zeros((B, npoint), dtype=torch.int32, device=device)
    temp_dist = torch.empty((B, N), dtype=torch.float32, device=device)

    # 动态计算大于 N 的最小 2 的幂次方 (如 N=8192 则 BLOCK_SIZE=8192)
    BLOCK_SIZE = 1 << (N - 1).bit_length()

    # num_warps=16 (512线程) 极限压榨并发度掩盖内存延迟
    _fps_v2_kernel[(B,)](
        xyz, temp_dist, idx,
        B, N, npoint,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16 
    )
    return idx.long()