import torch
import triton
import triton.language as tl

@triton.jit
def _ball_query_matrix_kernel(
    idx_ptr,          
    xyz_ptr,          
    new_xyz_ptr,      
    B, N, M, radius_sq, nsample: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)

    xyz_batch = xyz_ptr + batch_idx * N * 3
    new_xyz_batch = new_xyz_ptr + batch_idx * M * 3
    idx_batch = idx_ptr + batch_idx * M * nsample

    m_offsets = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    q_x = tl.load(new_xyz_batch + m_offsets * 3 + 0, mask=m_mask, other=0.0)
    q_y = tl.load(new_xyz_batch + m_offsets * 3 + 1, mask=m_mask, other=0.0)
    q_z = tl.load(new_xyz_batch + m_offsets * 3 + 2, mask=m_mask, other=0.0)

    found_cnts = tl.zeros((BLOCK_M,), dtype=tl.int32)
    
    for n_block_start in range(0, N, BLOCK_N):
        n_offsets = n_block_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        p_x = tl.load(xyz_batch + n_offsets * 3 + 0, mask=n_mask, other=0.0)
        p_y = tl.load(xyz_batch + n_offsets * 3 + 1, mask=n_mask, other=0.0)
        p_z = tl.load(xyz_batch + n_offsets * 3 + 2, mask=n_mask, other=0.0)

        dx = q_x[:, None] - p_x[None, :]
        dy = q_y[:, None] - p_y[None, :]
        dz = q_z[:, None] - p_z[None, :]
        dist_sq = dx * dx + dy * dy + dz * dz

        valid_mask = (dist_sq < radius_sq) & m_mask[:, None] & n_mask[None, :]
        valid_ints = tl.where(valid_mask, 1, 0)
        chunk_counts = tl.sum(valid_ints, axis=1) 
        local_pos = tl.cumsum(valid_ints, axis=1) - 1 
        write_pos = found_cnts[:, None] + local_pos # [BLOCK_M, BLOCK_N]
        write_mask = valid_mask & (write_pos < nsample)
        out_offsets = m_offsets[:, None] * nsample + write_pos
        write_vals = tl.broadcast_to(n_offsets[None, :], (BLOCK_M, BLOCK_N))
        tl.store(idx_batch + out_offsets, write_vals, mask=write_mask)
        found_cnts += chunk_counts

def triton_bq_v1(xyz, new_xyz, radius, nsample):
    assert xyz.is_cuda and new_xyz.is_cuda
    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape
    device = xyz.device

    idx = torch.zeros((B, M, nsample), dtype=torch.int32, device=device)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (B, triton.cdiv(M, BLOCK_M))

    _ball_query_matrix_kernel[grid](
        idx, xyz, new_xyz,
        B, N, M, radius * radius, nsample,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8  
    )
    return idx.long()
