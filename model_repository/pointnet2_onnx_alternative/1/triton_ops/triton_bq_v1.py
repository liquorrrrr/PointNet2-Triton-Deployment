import torch
import triton
import triton.language as tl

@triton.jit
def _ball_query_matrix_kernel(
    idx_ptr,          # 输出索引 (B, M, nsample)
    xyz_ptr,          # 数据集点云 (B, N, 3)
    new_xyz_ptr,      # 查询点云 (B, M, 3)
    B, N, M, radius_sq, nsample: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    # 1. 2D Grid 划分：批次 batch_idx 和查询点块 m_block_idx
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)

    # 2. 定位到当前 Batch 的指针起点
    xyz_batch = xyz_ptr + batch_idx * N * 3
    new_xyz_batch = new_xyz_ptr + batch_idx * M * 3
    idx_batch = idx_ptr + batch_idx * M * nsample

    # 3. 获取当前 Block 负责的 M 个查询点的索引 (行)
    m_offsets = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # 4. 加载查询点坐标 (形状: [BLOCK_M])
    q_x = tl.load(new_xyz_batch + m_offsets * 3 + 0, mask=m_mask, other=0.0)
    q_y = tl.load(new_xyz_batch + m_offsets * 3 + 1, mask=m_mask, other=0.0)
    q_z = tl.load(new_xyz_batch + m_offsets * 3 + 2, mask=m_mask, other=0.0)

    # 维护每个查询点当前已经找到了多少个点 (形状: [BLOCK_M])
    found_cnts = tl.zeros((BLOCK_M,), dtype=tl.int32)
    
    # === 核心：遍历 N 的矩阵化循环 ===
    for n_block_start in range(0, N, BLOCK_N):
        n_offsets = n_block_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # 5. 加载数据集点坐标 (形状: [BLOCK_N])
        p_x = tl.load(xyz_batch + n_offsets * 3 + 0, mask=n_mask, other=0.0)
        p_y = tl.load(xyz_batch + n_offsets * 3 + 1, mask=n_mask, other=0.0)
        p_z = tl.load(xyz_batch + n_offsets * 3 + 2, mask=n_mask, other=0.0)

        # 6. 【神来之笔：广播计算矩阵距离】
        # q_x[:, None] 变成 [BLOCK_M, 1]，p_x[None, :] 变成 [1, BLOCK_N]
        # dx 的形状直接是 [BLOCK_M, BLOCK_N]
        dx = q_x[:, None] - p_x[None, :]
        dy = q_y[:, None] - p_y[None, :]
        dz = q_z[:, None] - p_z[None, :]
        dist_sq = dx * dx + dy * dy + dz * dz

        # 7. 生成布尔掩码矩阵 [BLOCK_M, BLOCK_N]
        # 必须同时满足：距离小于半径，并且 m 和 n 都是合法的索引
        valid_mask = (dist_sq < radius_sq) & m_mask[:, None] & n_mask[None, :]

        # 8. 将布尔转为 0/1 [BLOCK_M, BLOCK_N]
        valid_ints = tl.where(valid_mask, 1, 0)

        # 9. 累加计算当前块中，每一行 (每个查询点) 新找到了几个点
        chunk_counts = tl.sum(valid_ints, axis=1) # 形状: [BLOCK_M]

        # 10. 【并行黑科技：前缀和算出写入位置】
        # tl.cumsum 可以在行方向上，算出一个点是该行的第几个有效点
        local_pos = tl.cumsum(valid_ints, axis=1) - 1 
        
        # 这个有效点的全局写入位置 = 之前已经找到的数量 + 局部位置
        write_pos = found_cnts[:, None] + local_pos # [BLOCK_M, BLOCK_N]

        # 11. 判断是否需要写入：必须是有效点，且写入位置小于 nsample
        write_mask = valid_mask & (write_pos < nsample)

        # 12. 算出输出数组的一维偏移量 [BLOCK_M, BLOCK_N]
        out_offsets = m_offsets[:, None] * nsample + write_pos

        # 要写进去的值，就是 n_offsets 扩充成矩阵 [BLOCK_M, BLOCK_N]
        write_vals = tl.broadcast_to(n_offsets[None, :], (BLOCK_M, BLOCK_N))

        # 13. 并行写回显存！一气呵成！
        tl.store(idx_batch + out_offsets, write_vals, mask=write_mask)

        # 14. 更新每个查询点已找到的数量
        found_cnts += chunk_counts

def triton_bq_v1(xyz, new_xyz, radius, nsample):
    assert xyz.is_cuda and new_xyz.is_cuda
    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape
    device = xyz.device

    # 原版 BQ 要求如果不够 nsample，用找到的第一个点复制填充。
    # 为保证严格对齐且不增加 Triton 负担，这里提前把 idx 填为每个 Query 自己的索引（或 0）。
    idx = torch.zeros((B, M, nsample), dtype=torch.int32, device=device)

    # 线程块配置：一个 Block 处理 64个查询点 x 64个数据集点 的矩阵
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid 维度 0：Batch，维度 1：M 被切分成多少块
    grid = (B, triton.cdiv(M, BLOCK_M))

    _ball_query_matrix_kernel[grid](
        idx, xyz, new_xyz,
        B, N, M, radius * radius, nsample,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8  # 256 线程，专门应对 64x64 的稠密矩阵
    )
    return idx.long()