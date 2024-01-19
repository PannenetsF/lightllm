import math

import numpy as np
import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class LlamaInferStateInfoBiB(LlamaInferStateInfo):
    def __init__(self, bib_size: int, chunk_size: int):
        super().__init__()
        self.bib_size = bib_size
        self.chunk_size = chunk_size

        self.chunk_num = None
        self.request_to_block = None
        self.block_to_length = None
        self.block_to_start = None
        self.block_to_chunk = None
        self.block_to_request = None
        self.block_num = None
        self.deno_tensor = None
        self.nume_tensor = None
        self.b_q_start_loc = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            max_len = self.b_seq_len.cpu().numpy().max()
            bs = self.b_seq_len.cpu().shape[0]
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, max_len)
                                                            for i in range(bs)], axis=0)).cuda()
            self.position_cos_kv = torch.index_select(model._cos_cached, 0, position_ids).view(bs, max_len, -1)
            self.position_sin_kv = torch.index_select(model._sin_cached, 0, position_ids).view(bs, max_len, -1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            self.position_cos_kv = self.position_cos
            self.position_sin_kv = self.position_sin
        return

    def init_bib_state(self, model, input_ids: torch.Tensor, b_req_idx: torch.Tensor, b_start_loc: torch.Tensor,
                       b_seq_len: torch.Tensor):
        self.init_some_extra_state(model, input_ids)
        k_length = b_seq_len
        k_start = b_start_loc
        batch_size = self.batch_size
        chunk_size = self.chunk_size
        chunk_num = math.floor((k_length.max() + chunk_size - 1) / chunk_size)

        bib_off = torch.arange(0, batch_size) * self.bib_size
        b_q_start_loc = b_start_loc - bib_off.cuda()

        blocks = (k_length / chunk_size).ceil().int()
        total_blocks = blocks.sum().item()
        max_blocks = blocks.max().item()
        block_to_length = np.zeros((total_blocks,), dtype=np.int32)
        block_to_request = np.zeros((total_blocks,), dtype=np.int32)
        block_to_start = np.zeros((total_blocks,), dtype=np.int32)
        block_to_chunk = np.zeros((total_blocks,), dtype=np.int32)
        request_to_block = np.zeros((batch_size, max_blocks), dtype=np.int32) - 1

        k_length = k_length.tolist()
        k_start = k_start.tolist()
        _arange = np.arange(total_blocks)
        block_idx = 0
        for req_idx, (leng, start) in enumerate(zip(k_length, k_start)):
            block_num = math.ceil(leng / chunk_size)
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

        block_num = len(block_to_request)

        self.request_to_block = request_to_block
        self.block_to_length = block_to_length
        self.block_to_start = block_to_start
        self.block_to_chunk = block_to_chunk
        self.block_to_request = block_to_request
        self.block_num = block_num
        self.b_q_start_loc = b_q_start_loc
        self.chunk_num = chunk_num
        # self.deno_tensor = deno_tensor
        # self.nume_tensor = nume_tensor

    def init_inter_tensor(self, q_tensor):
        if self.deno_tensor is not None:
            return
        block_num = self.block_num
        _, head_num, head_dim = q_tensor.shape
        self.deno_tensor = torch.empty((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
        self.nume_tensor = torch.empty((block_num, head_num, head_dim), dtype=torch.float32, device=q_tensor.device)
