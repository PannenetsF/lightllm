import triton
import triton.language as tl
import torch
import bib_binding

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.token_attention_bib import \
    token_attention_bib_raw

@triton.jit(do_not_specialize=[19])
def _bib_reduce_kernel(
        Req_to_block,
        S_max, S_exp_sum, S_exp_v_sum,
        Attn_out,
        stride_req_to_block_b, stride_req_to_block_s,
        stride_smax_blk, stride_smax_h,
        stride_s_exp_sum_blk, stride_s_exp_sum_h,
        stride_s_exp_v_sum_blk, stride_s_exp_v_sum_h, stride_s_exp_v_sum_d,
        stride_att_out_blk, stride_att_out_h, stride_att_out_d,
        # BATCH_SIZE: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        TOTAL_N_BLOCK_ROUND: tl.constexpr,
        TOTAL_N_BLOCK,

):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    chunk_off = tl.arange(0, TOTAL_N_BLOCK_ROUND)
    dim_off = tl.arange(0, BLOCK_DMODEL)

    block_idx = tl.load(
        Req_to_block + cur_batch * stride_req_to_block_b + chunk_off * stride_req_to_block_s,
        mask=chunk_off < TOTAL_N_BLOCK, other=-1)
    block_mask = block_idx != -1

    s_max = tl.load(
        S_max + block_idx * stride_smax_blk + cur_head * stride_smax_h,
        mask=block_mask, other=-1e9)
    s_exp_sum = tl.load(
        S_exp_sum + block_idx * stride_s_exp_sum_blk + cur_head * stride_s_exp_sum_h,
        mask=block_mask, other=0)
    s_exp_v = tl.load(
        S_exp_v_sum + block_idx[:,
                      None] * stride_s_exp_v_sum_blk + cur_head * stride_s_exp_v_sum_h + dim_off[
                                                                                         None,
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

    tl.store(
        Attn_out + cur_batch * stride_att_out_blk + cur_head * stride_att_out_h + dim_off * stride_att_out_d,
        s_exp_v)

@torch.no_grad()
def token_attention_bib_cuda(
        q, k, v, attn_out, infer_state: LlamaInferStateInfo, bib_chunk_size,
):
    # if attn_out is None:
    #     attn_out = torch.zeros_like(q)
    # else:
    #     attn_out.zero_()
    if attn_out is None:
        attn_out = torch.empty_like(q)

    infer_state.init_bib_extra_state(q, k, bib_chunk_size)
    CHUNK_SIZE = bib_chunk_size
    sm_scale = infer_state.sm_scale
    Req_to_tokens = infer_state.req_manager.req_to_token_indexs
    Req_to_block = infer_state.req_to_block
    B_req_idx = infer_state.b_req_idx
    B_seq_len = infer_state.b_seq_len
    Block_to_batch = infer_state.block_to_batch
    Block_to_start = infer_state.block_to_start
    s_max, s_exp_sum, s_exp_v = infer_state.s_max, infer_state.s_exp_sum, infer_state.s_exp_v_sum

    batch = infer_state.batch_size
    num_block = infer_state.num_block
    max_blocks = infer_state.max_blocks
    max_blocks_round = infer_state.max_blocks_round
    head = infer_state.head
    hidden = infer_state.hidden

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=attn_out,
        batch_seq_len=B_seq_len,
        batch_to_req=B_req_idx,
        block_to_batch=Block_to_batch,
        block_to_start=Block_to_start,
        req_to_tokens=Req_to_tokens,
        batch_to_blocks=Req_to_block,
        score_max=s_max,
        score_sum=s_exp_sum,
        score_expv=s_exp_v,
        sm_scale=float(sm_scale),
        chunk_size=int(CHUNK_SIZE),
        num_warps=int(CHUNK_SIZE // 32),
    )

    bib_binding.bib_decoding(**kwargs)

    grid = (batch, head)
    _bib_reduce_kernel[grid](
        Req_to_block,
        s_max, s_exp_sum, s_exp_v,
        attn_out,
        Req_to_block.stride(0), Req_to_block.stride(1),
        s_max.stride(0), s_max.stride(1),
        s_exp_sum.stride(0), s_exp_sum.stride(1),
        s_exp_v.stride(0), s_exp_v.stride(1), s_exp_v.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        BLOCK_DMODEL=hidden,
        TOTAL_N_BLOCK_ROUND=max_blocks_round,
        TOTAL_N_BLOCK=max_blocks,
    )

    return attn_out
