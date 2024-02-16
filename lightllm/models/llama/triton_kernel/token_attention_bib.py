import math

import numpy as np
import torch
import triton
import triton.language as tl

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


@triton.jit
def _bib_qkv_kernel(
        Q, K, V, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
        Block_to_batch, Block_to_start,
        S_max, S_exp_sum, S_exp_v_sum,
        stride_req_to_tokens_b, stride_req_to_tokens_s,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_smax_blk, stride_smax_h,
        stride_s_exp_sum_blk, stride_s_exp_sum_h,
        stride_s_exp_v_sum_blk, stride_s_exp_v_sum_h, stride_s_exp_v_sum_d,
        kv_group_num,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr
):
    cur_block = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch = tl.load(Block_to_batch + cur_block)
    start_n = tl.load(Block_to_start + cur_block)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q)  # shape = (BLOCK_DMODEL)
    offs_n_new = cur_batch_start_index + offs_n
    k_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new,
        mask=offs_n_new < cur_batch_end_index, other=0)
    off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None,
                                                                    :] * stride_kd  # shape = (BLOCK_N, BLOCK_DMODEL)
    k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)

    att_value = tl.sum(q[None, :] * k, 1)  # shape = [BLOCK_N, 1]
    att_value = tl.where(offs_n_new < cur_batch_end_index, att_value * sm_scale, -1e9)  # shape = (BLOCK_N, 1)

    off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    v = tl.load(V + off_v, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
    s_max = tl.max(att_value, axis=0)
    s_exp = tl.exp(att_value - s_max)
    s_exp_sum = tl.sum(s_exp, axis=0)

    s_exp_v_sum = tl.sum(s_exp[:, None] * v, axis=0)

    s_max_off = cur_block * stride_smax_blk + cur_head * stride_smax_h
    s_exp_sum_off = cur_block * stride_s_exp_sum_blk + cur_head * stride_s_exp_sum_h
    s_exp_v_sum_off = cur_block * stride_s_exp_v_sum_blk + cur_head * stride_s_exp_v_sum_h + offs_d * stride_s_exp_v_sum_d

    tl.store(S_max + s_max_off, s_max)
    tl.store(S_exp_sum + s_exp_sum_off, s_exp_sum)
    tl.store(S_exp_v_sum + s_exp_v_sum_off, s_exp_v_sum)


@triton.jit
def _bib_reduce_kernel(
        Req_to_block,
        S_max, S_exp_sum, S_exp_v_sum,
        Attn_out,
        stride_req_to_block_b, stride_req_to_block_s,
        stride_smax_blk, stride_smax_h,
        stride_s_exp_sum_blk, stride_s_exp_sum_h,
        stride_s_exp_v_sum_blk, stride_s_exp_v_sum_h, stride_s_exp_v_sum_d,
        stride_att_out_blk, stride_att_out_h, stride_att_out_d,
        BATCH_SIZE: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        TOTAL_N_BLOCK: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    if cur_batch > BATCH_SIZE:
        return

    chunk_off = tl.arange(0, TOTAL_N_BLOCK)
    dim_off = tl.arange(0, BLOCK_DMODEL)

    block_idx = tl.load(Req_to_block + cur_batch * stride_req_to_block_b + chunk_off * stride_req_to_block_s,
                        mask=chunk_off < TOTAL_N_BLOCK, other=-1)
    block_mask = block_idx != -1

    s_max = tl.load(S_max + block_idx * stride_smax_blk + cur_head * stride_smax_h,
                    mask=block_mask, other=-1e9)
    s_exp_sum = tl.load(S_exp_sum + block_idx * stride_s_exp_sum_blk + cur_head * stride_s_exp_sum_h,
                        mask=block_mask, other=0)
    s_exp_v = tl.load(
        S_exp_v_sum + block_idx[:, None] * stride_s_exp_v_sum_blk + cur_head * stride_s_exp_v_sum_h + dim_off[None,
                                                                                                      :] * stride_s_exp_v_sum_d,
        mask=block_mask[:, None], other=0)

    s_g_max = tl.max(s_max, axis=0)  # 1
    rescale = tl.exp(s_max - s_g_max)  # CN
    s_exp_sum = s_exp_sum * rescale  # CN
    s_exp_sum = tl.sum(s_exp_sum)  # 1

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    acc += s_exp_v * rescale[:, None]  # CN, h
    acc = tl.sum(acc, 0)
    s_exp_v = acc / s_exp_sum
    s_exp_v = s_exp_v.to(tl.float16)

    tl.store(Attn_out + cur_batch * stride_att_out_blk + cur_head * stride_att_out_h + dim_off * stride_att_out_d,
             s_exp_v)


@torch.no_grad()
def token_attention_bib(
        q, k, v, attn_out, infer_state: LlamaInferStateInfo, bib_chunk_size,
):
    if attn_out is None:
        attn_out = torch.empty_like(q)

    infer_state.init_bib_extra_state(q, bib_chunk_size)
    BLOCK_N = 64
    hidden = q.shape[-1]
    sm_scale = 1.0 / (hidden ** 0.5)

    Req_to_tokens = infer_state.req_manager.req_to_token_indexs
    Req_to_block = infer_state.req_to_block
    B_req_idx = infer_state.b_req_idx
    B_seq_len = infer_state.b_seq_len
    Block_to_batch = infer_state.block_to_batch
    Block_to_start = infer_state.block_to_start
    s_max, s_exp_sum, s_exp_v_sum = infer_state.s_max, infer_state.s_exp_sum, infer_state.s_exp_v_sum
    batch = B_req_idx.shape[0]
    num_block = Block_to_batch.shape[0]
    max_blocks = (B_seq_len.max() / bib_chunk_size).ceil().int().item()
    head = q.shape[1]

    kv_group_num = q.shape[1] // v.shape[1]
    return token_attention_bib_raw(
        q, k, v, sm_scale,
        Req_to_tokens, Req_to_block, B_req_idx, B_seq_len,
        Block_to_batch, Block_to_start,
        s_max, s_exp_sum, s_exp_v_sum,
        attn_out,
        batch,
        kv_group_num, head,
        BLOCK_N, hidden, max_blocks, num_block
    )


def token_attention_bib_raw(
        q, k, v, sm_scale,
        Req_to_tokens, Req_to_block, B_req_idx, B_seq_len,
        Block_to_batch, Block_to_start,
        s_max, s_exp_sum, s_exp_v_sum,
        attn_out,
        batch,
        kv_group_num, head,
        BLOCK_N, hidden, max_blocks, num_block
):
    grid = (num_block, head)

    _bib_qkv_kernel[grid](
        q, k, v, sm_scale, Req_to_tokens, B_req_idx, B_seq_len,
        Block_to_batch, Block_to_start,
        s_max, s_exp_sum, s_exp_v_sum,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s_max.stride(0), s_max.stride(1),
        s_exp_sum.stride(0), s_exp_sum.stride(1),
        s_exp_v_sum.stride(0), s_exp_v_sum.stride(1), s_exp_v_sum.stride(2),
        kv_group_num,
        BLOCK_DMODEL=hidden,
        BLOCK_N=BLOCK_N,
    )

    grid = (batch, head)
    _bib_reduce_kernel[grid](
        Req_to_block,
        s_max, s_exp_sum, s_exp_v_sum,
        attn_out,
        Req_to_block.stride(0), Req_to_block.stride(1),
        s_max.stride(0), s_max.stride(1),
        s_exp_sum.stride(0), s_exp_sum.stride(1),
        s_exp_v_sum.stride(0), s_exp_v_sum.stride(1), s_exp_v_sum.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        BATCH_SIZE=batch,
        BLOCK_DMODEL=hidden,
        TOTAL_N_BLOCK=max_blocks,
    )

    return attn_out


def align_attention_inter(lengths):
    batch_size = len(lengths)
    total_len = sum(lengths)
    lengths = torch.tensor(lengths, dtype=torch.long)
    H, h = 32, 128
    chunk = 32
    q = torch.randn(batch_size, H, h).cuda()
    k = torch.randn(total_len * 5, H, h).cuda()
    v = torch.randn(total_len * 5, H, h).cuda()

    starts = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]).cuda()
    rand_idx = torch.randperm(total_len * 5)[:total_len].cuda()
    max_length = max(lengths)
    used_k = k[rand_idx]
    used_v = v[rand_idx]

    qk_at = []
    for i in range(batch_size):
        qq = q[i].view(H, 1, h)  # shape (H, h)
        kk = used_k[starts[i]:starts[i] + lengths[i]]  # shape (lengths[i], H, h)
        vv = used_v[starts[i]:starts[i] + lengths[i]]  # shape (lengths[i], H, h)
        qk = torch.einsum('Hlh,LHh->HLl', qq, kk)
        qk = qk / math.sqrt(h)
        qk_at.append(qk.sum((1, 2)))

    qk_at = torch.stack(qk_at)

    req_to_token_indexs = torch.zeros(batch_size, max_length, dtype=torch.long).cuda() - 1
    for i in range(batch_size):
        req_to_token_indexs[i, :lengths[i]] = rand_idx[starts[i]:starts[i] + lengths[i]]

    blocks = []

    blocks = (lengths / chunk).ceil().int()
    total_blocks = blocks.sum().item()
    max_blocks = blocks.max().item()
    block_to_request = np.zeros((total_blocks,), dtype=np.int32)
    block_to_start = np.zeros((total_blocks,), dtype=np.int32)
    request_to_block = np.zeros((batch_size, max_blocks), dtype=np.int32) - 1

    k_length = lengths.tolist()
    k_start = starts.tolist()
    _arange = np.arange(total_blocks)
    block_idx = 0
    for req_idx, (leng, start) in enumerate(zip(k_length, k_start)):
        block_num = math.ceil(leng / chunk)
        block_to_start[block_idx:block_idx + block_num] = _arange[:block_num]
        block_to_request[block_idx:block_idx + block_num] = req_idx
        request_to_block[req_idx, :block_num] = _arange[block_idx: block_idx + block_num]
        block_idx += block_num

    # block_to_length = torch.from_numpy(block_to_length).cuda()
    block_to_start = torch.from_numpy(block_to_start).cuda()
    # block_to_chunk = torch.from_numpy(block_to_chunk).cuda()
    block_to_request = torch.from_numpy(block_to_request).cuda()
    request_to_block = torch.from_numpy(request_to_block).cuda()

    attn_out_triton = torch.zeros(batch_size, H, h).cuda()
    BLOCK_N = chunk
    hidden = q.shape[-1]
    sm_scale = 1.0 / (hidden ** 0.5)

    num_block = block_to_request.shape[0]
    kv_group_num = q.shape[1] // v.shape[1]
    head = H
    grid = (num_block, head)
    torch.cuda.synchronize()

    s_max_block = []
    s_exp_sum_block = []
    s_exp_v_sum_block = []
    for i in range(len(lengths)):
        qq = q[i].view(H, 1, h)  # shape (H, 1, h)
        token_idx = req_to_token_indexs[i, :lengths[i]]
        kk = k[token_idx]  # shape (lengths[i], H, h)
        kk = kk.permute(1, 2, 0)  # shape (H, h, lengths[i])
        qk = torch.bmm(qq, kk)  # shape (H, 1, lengths[i])
        qk = qk / math.sqrt(h)
        qk = qk.squeeze(1)  # shape (H, lengths[i])
        vv = v[token_idx]  # shape (lengths[i], H, h)

        for j in range((lengths[i] + BLOCK_N - 1) // BLOCK_N):
            start = j * BLOCK_N
            end = min((j + 1) * BLOCK_N, lengths[i])
            qk_blk = qk[:, start:end]
            s_max = qk_blk.max(1).values  # shape (H,)
            s_exp = (qk_blk - s_max.unsqueeze(1)).exp()  # shape (H, end - start)
            s_exp_sum = s_exp.sum(1)  # shape (H,)
            vv_blk = vv[start:end]  # shape (end - start, H, h)
            vv_blk = vv_blk.permute(1, 0, 2)  # shape (H, end - start, h)
            s_exp_v_sum = torch.bmm(s_exp.unsqueeze(1), vv_blk).squeeze(1)  # shape (H, h)

            s_max_block.append(s_max)
            s_exp_sum_block.append(s_exp_sum)
            s_exp_v_sum_block.append(s_exp_v_sum)

    att_out_ref = []
    for i in range(len(lengths)):
        qq = q[i].view(H, 1, h)
        token_idx = req_to_token_indexs[i, :lengths[i]]
        kk = k[token_idx]
        vv = v[token_idx]
        qk = torch.einsum('Hlh,LHh->HLl', qq, kk)
        qk = qk / math.sqrt(h)
        score = torch.softmax(qk, dim=1)
        att_out_ref.append(torch.einsum('HLl,LHh->Hlh', score, vv).view(H, h))
    att_out_ref = torch.stack(att_out_ref)

    s_max_block = torch.stack(s_max_block)
    s_exp_sum_block = torch.stack(s_exp_sum_block)
    s_exp_v_sum_block = torch.stack(s_exp_v_sum_block)

    Smax = torch.zeros(num_block, H).cuda()
    Sexp_sum = torch.zeros(num_block, H).cuda()
    Sexp_v_sum = torch.zeros(num_block, H, h).cuda()

    assert s_max_block.shape == Smax.shape
    assert s_exp_sum_block.shape == Sexp_sum.shape
    assert s_exp_v_sum_block.shape == Sexp_v_sum.shape, f'{s_exp_v_sum_block.shape} {Sexp_v_sum.shape}'

    q, k, v = q.half(), k.half(), v.half()
    attn_out_triton = attn_out_triton.half()

    attn_out_triton = token_attention_bib_raw(
        q, k, v, sm_scale,
        req_to_token_indexs, request_to_block, torch.arange(batch_size).cuda(), torch.tensor(lengths).cuda(),
        block_to_request, block_to_start,
        Smax, Sexp_sum, Sexp_v_sum,
        attn_out_triton,
        batch=batch_size,
        kv_group_num=kv_group_num, head=head,
        BLOCK_N=BLOCK_N, hidden=hidden, max_blocks=max_blocks, num_block=num_block
    )

    # _bib_qkv_kernel[grid](
    #     q, k, v, sm_scale, req_to_token_indexs, torch.arange(batch_size).cuda(), lengths.cuda(),
    #     block_to_request, block_to_start,
    #     Smax, Sexp_sum, Sexp_v_sum,
    #     req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
    #     q.stride(0), q.stride(1), q.stride(2),
    #     k.stride(0), k.stride(1), k.stride(2),
    #     v.stride(0), v.stride(1), v.stride(2),
    #     Smax.stride(0), Smax.stride(1),
    #     Sexp_sum.stride(0), Sexp_sum.stride(1),
    #     Sexp_v_sum.stride(0), Sexp_v_sum.stride(1), Sexp_v_sum.stride(2),
    #     kv_group_num,
    #     BLOCK_DMODEL=hidden,
    #     BLOCK_N=BLOCK_N,
    # )
    # torch.cuda.synchronize()
    # # print(s_max_block - Smax)
    # # print(s_exp_sum_block - Sexp_sum)
    # # print(s_exp_v_sum_block - Sexp_v_sum)
    #
    # grid = (batch_size, head)
    # _bib_reduce_kernel[grid](
    #     request_to_block,
    #     Smax, Sexp_sum, Sexp_v_sum,
    #     attn_out_triton,
    #     request_to_block.stride(0), request_to_block.stride(1),
    #     Smax.stride(0), Smax.stride(1),
    #     Sexp_sum.stride(0), Sexp_sum.stride(1),
    #     Sexp_v_sum.stride(0), Sexp_v_sum.stride(1), Sexp_v_sum.stride(2),
    #     attn_out_triton.stride(0), attn_out_triton.stride(1), attn_out_triton.stride(2),
    #     BATCH_SIZE=batch_size,
    #     BLOCK_DMODEL=hidden,
    #     TOTAL_N_BLOCK=max_blocks,
    # )

    torch.cuda.synchronize()
    print(f'max error = {(attn_out_triton - att_out_ref).abs().max()}')
    cos_sim = torch.nn.functional.cosine_similarity(attn_out_triton.view(-1), att_out_ref.view(-1), dim=0)
    print(f'cosine similarity = {cos_sim.item()}')


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    align_attention_inter([7])
