import math

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel_bib(KV, Cos, Sin, block_to_request, block_to_start, block_to_length, block_to_chunk, stride_kvbs,
                       stride_kvh, stride_kvd, stride_cosbs, stride_cosseq, stride_cosd, stride_sinbs, stride_sinseq,
                       stride_sind, H,
                       # N_CTX 代表要计算的上下文长度
                       B, BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr, ):
    cur_blk_index = tl.program_id(0)
    cur_head_index = tl.program_id(1)
    if cur_blk_index >= B:
        return

    cur_batch = tl.load(block_to_request + cur_blk_index)
    cur_start = tl.load(block_to_start + cur_blk_index)
    cur_length = tl.load(block_to_length + cur_blk_index)
    cur_chunk = tl.load(block_to_chunk + cur_blk_index)

    cur_seq_off = cur_chunk * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    cur_seq_range = cur_start + cur_seq_off
    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)

    seq_mask = cur_seq_off[:, None, None] < cur_length
    head_mask = cur_head_range[None, :, None] < H
    mask = seq_mask & head_mask

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)

    off_kv0 = cur_seq_range[:, None, None] * stride_kvbs + cur_head_range[None, :, None] * stride_kvh + dim_range0[None,
                                                                                                        None,
                                                                                                        :] * stride_kvd
    off_kv1 = cur_seq_range[:, None, None] * stride_kvbs + cur_head_range[None, :, None] * stride_kvh + dim_range1[None,
                                                                                                        None,
                                                                                                        :] * stride_kvd

    off_dimcos_sin = cur_batch * stride_cosbs + cur_seq_off[:, None, None] * stride_cosseq + dim_range0[None, None,
                                                                                             :] * stride_cosd

    kv0 = tl.load(KV + off_kv0, mask=mask, other=0.)
    kv1 = tl.load(KV + off_kv1, mask=mask, other=0.)
    cos = tl.load(Cos + off_dimcos_sin, mask=seq_mask, other=0.)
    sin = tl.load(Sin + off_dimcos_sin, mask=seq_mask, other=0.)

    out0 = kv0 * cos - kv1 * sin
    out1 = kv0 * sin + kv1 * cos

    tl.store(KV + off_kv0, out0, mask=mask)
    tl.store(KV + off_kv1, out1, mask=mask)

    return


@torch.no_grad()
def rotary_emb_fwd_bib(kv, cos, sin, block_to_request, block_to_start, block_to_length, block_to_chunk, block_num,
                       chunk_size=64):
    _, head_num, head_dim = kv.shape
    BLOCK_NUM = block_num
    BLOCK_HEAD = 4
    BLOCK_SEQ = chunk_size
    grid = ((BLOCK_NUM), triton.cdiv(head_num, BLOCK_HEAD))
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    _rotary_kernel_bib[grid](kv, cos, sin, block_to_request, block_to_start, block_to_length, block_to_chunk,
                             kv.stride(0), kv.stride(1), kv.stride(2), cos.stride(0), cos.stride(1), cos.stride(2),
                             sin.stride(0),
                             sin.stride(1), sin.stride(2), head_num, BLOCK_NUM, BLOCK_HEAD=BLOCK_HEAD,
                             BLOCK_SEQ=BLOCK_SEQ,
                             BLOCK_DMODEL=head_dim, num_warps=num_warps, num_stages=1, )
    return


def torch_rotary_emb(x_data, cos, sin, x_start, x_len):
    y = torch.zeros_like(x_data)
    for c, s, start, leng in zip(cos, sin, x_start, x_len):
        x0 = x_data[start: start + leng, :, 0: x_data.shape[-1] // 2]
        x1 = x_data[start: start + leng, :, x_data.shape[-1] // 2: x_data.shape[-1]]
        c = c[:leng].view(leng, 1, -1)
        s = s[:leng].view(leng, 1, -1)
        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c
        y[start: start + leng, :, 0: x_data.shape[-1] // 2] = o0
        y[start: start + leng, :, x_data.shape[-1] // 2: x_data.shape[-1]] = o1
    return y


def align_rotary_emb(bs, max_len, H, d, bib, dtype=torch.half):
    # create data
    max_len = min(4 * bib, max_len)
    x_len = torch.randint(5, max_len, (bs,))
    x_data = torch.rand(x_len.sum() + bib * bs, H, d)
    cur_start = 0

    x_start = []
    for i in range(bs):
        x_start.append(cur_start)
        cur_end = cur_start + x_len[i]
        x_data[cur_end: cur_end + bib] = 0.
        cur_start = cur_end + bib

    print(x_start, x_len)
    blocks = (x_len / bib).ceil().int()
    total_blocks = blocks.sum().item()
    max_blocks = blocks.max().item()
    block_to_length = np.zeros((total_blocks,), dtype=np.int32)
    block_to_request = np.zeros((total_blocks,), dtype=np.int32)
    block_to_start = np.zeros((total_blocks,), dtype=np.int32)
    block_to_chunk = np.zeros((total_blocks,), dtype=np.int32)
    request_to_block = np.zeros((bs, max_blocks), dtype=np.int32) - 1

    k_length = x_len.tolist()
    k_start = x_start
    _arange = np.arange(total_blocks)
    block_idx = 0
    for req_idx, (leng, start) in enumerate(zip(k_length, k_start)):
        block_num = math.ceil(leng / bib)
        block_to_length[block_idx:block_idx + block_num] = leng
        block_to_start[block_idx:block_idx + block_num] = start
        block_to_chunk[block_idx:block_idx + block_num] = _arange[:block_num]
        block_to_request[block_idx:block_idx + block_num] = req_idx
        request_to_block[req_idx, :block_num] = _arange[block_idx: block_idx + block_num]
        block_idx += block_num

    block_to_length = torch.from_numpy(block_to_length).cuda()
    block_to_start = torch.from_numpy(block_to_start).cuda()
    block_to_chunk = torch.from_numpy(block_to_chunk).cuda()
    block_to_request = torch.from_numpy(block_to_request).cuda()
    request_to_block = torch.from_numpy(request_to_block).cuda()

    x_data = x_data.cuda()

    cos_shape = (bs, max(k_length), d // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device='cuda')
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device='cuda')
    # forward pass
    y_torch = torch_rotary_emb(x_data, cos, sin, x_start, x_len)
    rotary_emb_fwd_bib(x_data, cos, sin, block_to_request, block_to_start, block_to_length, block_to_chunk,
                       block_num=len(block_to_request), chunk_size=bib)
    y_ref = x_data

    # compare
    print("type:", y_torch.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_torch - y_ref)))
    assert torch.allclose(y_torch, y_ref, atol=1e-2, rtol=0)


if __name__ == '__main__':
    # align_rotary_emb(bs=5, max_len=1024, H=16, d=128, bib=64)
    align_rotary_emb(bs=1, max_len=20, H=16, d=128, bib=32)
