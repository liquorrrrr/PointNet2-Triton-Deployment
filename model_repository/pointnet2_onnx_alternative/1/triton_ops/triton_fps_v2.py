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

    x = tl.load(xyz_batch + cols * 3 + 0, mask=mask, other=0.0)
    y = tl.load(xyz_batch + cols * 3 + 1, mask=mask, other=0.0)
    z = tl.load(xyz_batch + cols * 3 + 2, mask=mask, other=0.0)

    dist_local = tl.full((BLOCK_SIZE,), 1e10, dtype=tl.float32)

    farthest_idx = 0
    tl.store(idx_batch, farthest_idx)


    for step in range(1, M):
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
    xyz = xyz.contiguous() 
    B, N, C = xyz.shape
    device = xyz.device

    idx = torch.zeros((B, npoint), dtype=torch.int32, device=device)
    temp_dist = torch.empty((B, N), dtype=torch.float32, device=device)
    BLOCK_SIZE = 1 << (N - 1).bit_length()

    _fps_v2_kernel[(B,)](
        xyz, temp_dist, idx,
        B, N, npoint,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16 
    )
    return idx.long()
