import math
import os.path

import numpy as np
import torch
import triton

from lightllm.models.llama.triton_kernel.token_attention_bib import \
    token_attention_bib_raw
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import \
    token_att_fwd
from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import \
    token_softmax_reducev_fwd


def _token_decode_attention_normal(q, k, v, out, att_m_tensor,
                                   req_to_token_indexs,
                                   b_req_idx,
                                   b_start_loc,
                                   b_seq_len,
                                   max_len_in_batch,
                                   other_kv_index,
                                   total_token_num,
                                   batch_size,
                                   head_num, head_dim):
    batch_size = batch_size
    calcu_shape1 = (batch_size, head_num, head_dim)

    token_att_fwd(q.view(calcu_shape1),
                  k,
                  att_m_tensor,
                  req_to_token_indexs,
                  b_req_idx,
                  b_start_loc,
                  b_seq_len,
                  max_len_in_batch)

    token_softmax_reducev_fwd(att_m_tensor,
                              v,
                              out.view(calcu_shape1),
                              req_to_token_indexs,
                              b_req_idx,
                              b_start_loc,
                              b_seq_len,
                              other_kv_index)
    return out


def prepare_token_attention_normal(lengths, head_num, head_dim, chunk_size):
    # q, k, v,
    # req_to_token_indexs,
    # b_req_idx,
    # b_start_loc,
    # b_seq_len,
    # max_len_in_batch,
    # other_kv_index,
    # total_token_num,
    # batch_size,
    # head_num, head_dim
    batch_size = len(lengths)
    max_len_in_batch = max(lengths)
    total_token_num = sum(lengths)

    req_to_token_indexs = torch.zeros((batch_size, max_len_in_batch),
                                      dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((batch_size,),
                              dtype=torch.int32, device="cuda")
    cnt = 0
    for idx in range(batch_size):
        req_to_token_indexs[idx, :lengths[idx]] = torch.arange(cnt,
                                                               cnt + lengths[
                                                                   idx], ).cuda()
        b_start_loc[idx] = cnt
        cnt += lengths[idx]
    b_req_idx = torch.arange(batch_size, ).cuda()
    b_seq_len = lengths.clone().detach().cuda()
    q = torch.rand((batch_size, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    att_m_tensor = torch.empty((head_num, total_token_num),
                               dtype=q.dtype, device="cuda")
    k = torch.rand((total_token_num, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    v = torch.rand((total_token_num, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    other_kv_index = 0
    return (q, k, v, out, att_m_tensor,
            req_to_token_indexs,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            other_kv_index,
            total_token_num,
            batch_size,
            head_num, head_dim)


def prepare_token_attention_bib(lengths, head_num, head_dim, chunk_size):
    # q, k, v, sm_scale,
    # Req_to_tokens, Req_to_block, B_req_idx, B_seq_len,
    # Block_to_batch, Block_to_start,
    # s_max, s_exp_sum, s_exp_v_sum,
    # attn_out,
    # batch,
    # kv_group_num, head,
    # BLOCK_N, hidden, max_blocks, num_block
    batch_size = len(lengths)
    max_len_in_batch = max(lengths)
    total_token_num = sum(lengths)
    req_to_token_indexs = torch.zeros((batch_size, max_len_in_batch),
                                      dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((batch_size,),
                              dtype=torch.int32, device="cuda")
    cnt = 0
    for idx in range(batch_size):
        req_to_token_indexs[idx, :lengths[idx]] = torch.arange(cnt,
                                                               cnt + lengths[
                                                                   idx], ).cuda()
        b_start_loc[idx] = cnt
        cnt += lengths[idx]
    b_req_idx = torch.arange(batch_size, ).cuda()
    b_seq_len = lengths.clone().detach().cuda()
    q = torch.rand((batch_size, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    k = torch.rand((total_token_num, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    v = torch.rand((total_token_num, head_num, head_dim),
                   dtype=torch.float16, device="cuda")
    sm_scale = 1.0
    blocks = (lengths / chunk_size).ceil().int()
    total_blocks = blocks.sum().item()
    max_blocks = blocks.max().item()
    block_to_batch = np.zeros((total_blocks,), dtype=np.int32)
    block_to_start = np.zeros((total_blocks,), dtype=np.int32)
    req_to_block = np.zeros((batch_size, max_blocks), dtype=np.int32) - 1

    _arange = np.arange(total_blocks)
    block_idx = 0
    for req_idx, (leng, start) in enumerate(
            zip(b_seq_len.tolist(), b_start_loc.tolist())):
        block_num = math.ceil(leng / chunk_size)
        block_to_start[block_idx:block_idx + block_num] = _arange[:block_num]
        block_to_batch[block_idx:block_idx + block_num] = req_idx
        req_to_block[req_idx, :block_num] = _arange[
                                            block_idx: block_idx + block_num]
        block_idx += block_num
    block_to_start = torch.from_numpy(block_to_start).cuda()
    block_to_batch = torch.from_numpy(block_to_batch).cuda()
    req_to_block = torch.from_numpy(req_to_block).cuda()

    s_max = torch.zeros((total_token_num, head_num),
                        dtype=torch.float32, device="cuda")
    s_exp_sum = torch.zeros((total_token_num, head_num),
                            dtype=torch.float32, device="cuda")
    s_exp_v_sum = torch.zeros((total_token_num, head_num, head_dim),
                              dtype=torch.float32, device="cuda")
    attn_out = torch.zeros((batch_size, head_num, head_dim),
                           dtype=torch.float16, device="cuda")
    kv_group_num = 1
    head = head_num
    BLOCK_N = chunk_size
    hidden = head_dim
    max_blocks = 1
    num_block = 1
    return (
        q, k, v, sm_scale, req_to_token_indexs, req_to_block, b_req_idx,
        b_seq_len,
        block_to_batch, block_to_start, s_max, s_exp_sum, s_exp_v_sum,
        attn_out, batch_size, kv_group_num, head, BLOCK_N, hidden, max_blocks,
        num_block)


def benchmark(avg_leng, std_leng, max_leng, batch_size, head_num, head_dim,
              chunk_size, sweep_config={}):
    sweep_term = sweep_config.get('term', 'batch_size')
    sweep_start = sweep_config.get('start', 1)
    sweep_end = sweep_config.get('end', 1)
    sweep_step = sweep_config.get('step', 1)
    sweep_log = sweep_config.get('log', False)

    sweep_numbers = np.linspace(sweep_start, sweep_end, sweep_step)
    base_config = dict(
        avg_leng=avg_leng,
        std_leng=std_leng,
        max_leng=max_leng,
        batch_size=batch_size,
        head_num=head_num,
        head_dim=head_dim,
        chunk_size=chunk_size,
    )

    base_config.pop(sweep_term)

    # lengths = np.random.normal(avg_leng, std_leng, batch_size).astype(np.int32)
    # lengths = np.clip(lengths, 1, max_leng)
    # normal_args = prepare_token_attention_normal(lengths, head_num, head_dim)
    # bib_args = prepare_token_attention_bib(lengths, head_num, head_dim, chunk_size)
    def _bench(base_config):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['val'],
                x_vals=sweep_numbers,
                x_log=sweep_log,
                line_arg='provider',
                line_vals=['light', 'bib'],
                line_names=['light', 'bib'],
                # line_vals=['bib'],
                # line_names=['bib'],
                ylabel='latency',
                plot_name=f'{sweep_term}',
                args={"args": base_config},
            ))
        def benchmark(val, provider, args):
            print(f'benching {sweep_term} = {val}, provide = {provider}')
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'light':
                prepare_fn = prepare_token_attention_normal
                call_fn = _token_decode_attention_normal
            elif provider == 'bib':
                prepare_fn = prepare_token_attention_bib
                call_fn = token_attention_bib_raw
            else:
                raise NotImplementedError
            kwargs = {}
            for key in args:
                kwargs[key] = args[key]
            kwargs[sweep_term] = val
            avg_leng = kwargs.pop('avg_leng')
            std_leng = kwargs.pop('std_leng')
            max_leng = kwargs.pop('max_leng')
            batch_size = int(kwargs.pop('batch_size'))
            lengths = np.random.normal(avg_leng, std_leng, batch_size).astype(
                np.int32)
            print(lengths)
            # let the min lengths to be max
            min_idx = np.argmin(lengths)
            lengths[min_idx] = max_leng
            lengths = np.clip(lengths, 1, max_leng)
            lengths = torch.from_numpy(lengths)
            kwargs['lengths'] = lengths
            args = prepare_fn(**kwargs)
            args = [arg.cuda().contiguous() if isinstance(arg,
                                                          torch.Tensor) else arg
                    for arg in args]
            p50, p20, p80 = triton.testing.do_bench(lambda: call_fn(*args),
                                                    quantiles=quantiles, fast_flush=True)
            return p50, p20, p80

        return benchmark

    bm = _bench(base_config)
    # turn sweep config to name
    sweep_name = f'{sweep_term}_{sweep_start}_{sweep_end}_{sweep_step}_{sweep_log}'
    print(f'running {sweep_name}')
    if os.path.exists(sweep_name):
        print(f'{sweep_name} exists, skip')
    else:
        os.makedirs(sweep_name, exist_ok=True)
    print(f'real path is {os.path.abspath(sweep_name)}')
    bm.run(show_plots=False, print_data=False, save_path=sweep_name)


benchmark(avg_leng=400, std_leng=0, max_leng=4000, batch_size=4, head_num=32,
          head_dim=128, chunk_size=32,
          sweep_config={'term': 'avg_leng', 'start': 100, 'end': 4000, 'step': 10,
                        'log': True})
# benchmark(avg_leng=400, std_leng=100, max_leng=4000, batch_size=1, head_num=32,
#           head_dim=128, chunk_size=32,
#           sweep_config={'term': 'batch_size', 'start': 1, 'end': 128, 'step': 10,
#                         'log': False})
#
