import math

import torch
import triton
import triton.language as tl


@triton.jit
def bib_gqa_ref_qkv_score_kernel(
        q_tensor, q_bs, q_H, q_h,
        k_tensor, k_D, k_H, k_h,
        v_tensor, v_D, v_H, v_h,
        deno_tensor, deno_bs, deno_H, deno_dim,  # 2dim, max and sum
        nume_tensor, nume_bs, nume_H, nume_h,
        block_to_request, b2r_bs,
        block_to_start, b2s_bs,
        block_to_length, b2l_bs,
        block_to_chunk, b2c_bs,
        sm_scale: tl.constexpr,
        HEAD_NUM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
):
    '''
    the b2l is the block to length in the block, which is the used length of the block
    '''
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch = tl.load(block_to_request + block_idx * b2r_bs)
    cur_start = tl.load(block_to_start + block_idx * b2s_bs)
    cur_length = tl.load(block_to_length + block_idx * b2l_bs)
    cur_chunk = tl.load(block_to_chunk + block_idx * b2c_bs)

    head_off = tl.arange(0, HEAD_DIM)
    chunk_off = tl.arange(0, CHUNK_SIZE)

    q_off = cur_batch * q_bs + head_idx * q_H + head_off * q_h
    q_vec = tl.load(q_tensor + q_off)  # [head_dim]

    k_len = cur_chunk * CHUNK_SIZE + chunk_off
    k_off = (k_len[:, None] + cur_start) * k_D + head_idx * k_H + head_off[None, :] * k_h
    v_off = (k_len[:, None] + cur_start) * v_D + head_idx * v_H + head_off[None, :] * v_h
    k_mask = k_len < cur_length
    k_mat = tl.load(k_tensor + k_off, mask=k_mask[:, None], other=0.0)
    score = tl.sum(q_vec[None, :] * k_mat, axis=1)
    score = tl.where(k_mask, score * sm_scale, -1e9)

    v_mat = tl.load(v_tensor + v_off, mask=k_mask[:, None], other=0.0)  # [chunk_size, head_dim]
    s_max = tl.max(score, axis=0)
    s_exp = tl.exp(score - s_max)
    s_exp_sum = tl.sum(s_exp, axis=0)

    s_exp_v = s_exp[:, None] * v_mat
    s_exp_v_sum = tl.sum(s_exp_v, axis=0)

    # save s_max, s_exp_sum, s_exp_v
    # shape (1, ), (1, ), (head_dim, )
    tl.store(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 0 * deno_dim, s_max)
    tl.store(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 1 * deno_dim, s_exp_sum)
    tl.store(nume_tensor + block_idx * nume_bs + head_idx * nume_H + head_off * nume_h, s_exp_v_sum)


@triton.jit
def bib_gqa_ref_request_max_reduce(
        request_to_block, r2b_bs, r2b_blk,
        deno_tensor, deno_bs, deno_H, deno_dim,  # 2dim, max and sum
        nume_tensor, nume_bs, nume_H, nume_h,
        out_tensor, out_bs, out_H, out_h,
        BATCH_SIZE: tl.constexpr,
        CHUNK_NUM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROUND_CHUNK_NUM: tl.constexpr
):
    request_id = tl.program_id(0)
    head_idx = tl.program_id(1)

    if request_id >= BATCH_SIZE:
        return

    chunk_off = tl.arange(0, ROUND_CHUNK_NUM)
    head_off = tl.arange(0, HEAD_DIM)
    block_idx = tl.load(request_to_block + request_id * r2b_bs + r2b_blk * chunk_off, mask=chunk_off < CHUNK_NUM,
                        other=-1)
    block_mask = block_idx != -1

    s_max = tl.load(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 0 * deno_dim, mask=block_mask,
                    other=-1e9)  # (CN, )
    s_exp_sum = tl.load(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 1 * deno_dim, mask=block_mask,
                        other=0)  # (CN, )
    s_exp_v = tl.load(nume_tensor + block_idx[:, None] * nume_bs + head_idx * nume_H + head_off[None, :] * nume_h,
                      mask=block_mask[:, None], other=0.)  # (CN, h)

    s_g_max = tl.max(s_max, axis=0)  # 1
    rescale = tl.exp(s_max - s_g_max)  # CN
    s_exp_sum = s_exp_sum * rescale  # CN
    s_exp_sum = tl.sum(s_exp_sum)  # 1

    s_exp_v = s_exp_v * rescale[:, None]  # CN, h
    s_exp_v = tl.sum(s_exp_v, 0)
    s_exp_v = s_exp_v / s_exp_sum

    tl.store(out_tensor + request_id * out_bs + head_idx * out_H + head_off * out_h, s_exp_v)


@torch.no_grad()
def bib_decoding(
        q_tensor,
        k_tensor,
        v_tensor,
        score_tensor,
        request_to_block,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale,
        deno_tensor=None,
        nume_tensor=None,
        CHUNK_SIZE=64,
        debug=None,
):
    r'''
    step 1:
        deno: get max scale and exp sum of each block
        nume: get the inner product of scale and value, (reduce on the Length dim)
    step 2:
        sync all deno
            deno: get global max, and rescale the exp sum
    '''
    block_num = block_to_request.shape[0]
    chunk_num = request_to_block.shape[1]
    batch_size = q_tensor.shape[0]
    head_num = q_tensor.shape[1]
    head_dim = q_tensor.shape[2]

    if not deno_tensor:
        deno_tensor = torch.zeros((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
    if not nume_tensor:
        nume_tensor = torch.zeros((block_num, head_num, head_dim), dtype=torch.float32, device=q_tensor.device)

    grid = lambda META: (block_num, head_num)

    bib_gqa_ref_qkv_score_kernel[grid](
        q_tensor, q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2),
        k_tensor, k_tensor.stride(0), k_tensor.stride(1), k_tensor.stride(2),
        v_tensor, v_tensor.stride(0), v_tensor.stride(1), v_tensor.stride(2),
        deno_tensor, deno_tensor.stride(0), deno_tensor.stride(1), deno_tensor.stride(2),
        nume_tensor, nume_tensor.stride(0), nume_tensor.stride(1), nume_tensor.stride(2),
        block_to_request, block_to_request.stride(0),
        block_to_start, block_to_start.stride(0),
        block_to_length, block_to_length.stride(0),
        block_to_chunk, block_to_chunk.stride(0),
        sm_scale=sm_scale,
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        CHUNK_SIZE=CHUNK_SIZE,
    )

    grid = lambda META: (2 ** (math.ceil(math.log2(batch_size))), head_num)
    bib_gqa_ref_request_max_reduce[grid](
        request_to_block, request_to_block.stride(0), request_to_block.stride(1),
        deno_tensor, deno_tensor.stride(0), deno_tensor.stride(1), deno_tensor.stride(2),
        nume_tensor, nume_tensor.stride(0), nume_tensor.stride(1), nume_tensor.stride(2),
        score_tensor, score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2),
        BATCH_SIZE=batch_size,
        CHUNK_NUM=chunk_num,
        HEAD_DIM=head_dim,
        ROUND_CHUNK_NUM=2 ** (math.ceil(math.log2(chunk_num)))
    )


def bib_decoding_torch(
        q_tensor,
        k_tensor,
        v_tensor,
        k_length,
        k_start,
        sm_scale,
):
    bs, H, h = q_tensor.shape
    max_L = k_length.max()
    sm_mat = torch.zeros((bs, H, max_L), dtype=torch.float16, device=torch.device('cuda:0'))

    for bidx in range(bs):
        q_vec = q_tensor[bidx]  # H, h
        start = k_start[bidx]
        dur = k_length[bidx]
        k_vec = k_tensor[start: start + dur]  # L, H, h

        q_vec = q_vec.reshape(H, h, 1)
        k_vec = k_vec.permute(1, 0, 2).contiguous()  # H, L, h

        sm_mat[bidx, :, :dur] = torch.matmul(k_vec, q_vec).reshape(H, dur)  # H, L
        sm_mat[bidx, :, dur:] = torch.tensor([float('-inf')]).to(torch.half)

    sm_mat *= sm_scale
    sm_mat = sm_mat.softmax(dim=-1)

    attn = torch.zeros((bs, H, h), dtype=torch.float16, device=torch.device('cuda:0'))
    for bidx in range(bs):
        sm = sm_mat[bidx]  # H, L
        sm = sm.permute(1, 0).contiguous().reshape(-1, H, 1)

        start = k_start[bidx]
        dur = k_length[bidx]
        v_vec = v_tensor[start: start + dur]  # L, H, h

        score = sm[:dur] * v_vec
        score = score.sum(dim=0)  # H, h
        attn[bidx] = score

    return attn
