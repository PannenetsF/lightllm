import logging
import math
import torch
import numpy as np
import triton

logger = logging.getLogger(__name__)

from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager

class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None

        self.req_to_block = None
        self.block_to_batch = None
        self.block_to_start = None
        self.s_max = None
        self.s_exp_sum = None
        self.s_exp_v_sum = None
        self.hidden = None
        self.sm_scale = None
        self.head = None

    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            # b_loc[0, max_len_in_batch - 1].item()
        return

    def init_bib_extra_state(self, q, k, chunk_size):
        if self.req_to_block is not None:
            return
        else:
            # logger.debug(f'init bib with bib={chunk_size} batch_size={q.shape[0]}')
            k_length = self.b_seq_len
            k_start = self.b_start_loc
            batch_size, H, h = q.shape
            chunk_size = chunk_size
            chunk_num = math.floor((k_length.max() + chunk_size - 1) / chunk_size)

            blocks = (k_length / chunk_size).ceil().int()
            total_blocks = blocks.sum().item()
            max_blocks = blocks.max().item()
            max_blocks_round = triton.next_power_of_2(max_blocks)
            block_to_request = np.zeros((total_blocks,), dtype=np.int32)
            block_to_start = np.zeros((total_blocks,), dtype=np.int32)
            used_blk = max(1024, max_blocks_round)
            request_to_block = np.zeros((batch_size, used_blk), dtype=np.int32) - 1

            _arange = np.arange(total_blocks)
            block_idx = 0
            for req_idx, (leng, start) in enumerate(zip(k_length.tolist(), k_start.tolist())):
                block_num = math.ceil(leng / chunk_size)
                block_to_start[block_idx:block_idx + block_num] = _arange[:block_num]
                block_to_request[block_idx:block_idx + block_num] = req_idx
                request_to_block[req_idx, :block_num] = _arange[block_idx: block_idx + block_num]
                block_idx += block_num

            block_to_start = torch.from_numpy(block_to_start).cuda()
            block_to_request = torch.from_numpy(block_to_request).cuda()
            request_to_block = torch.from_numpy(request_to_block).cuda()

            self.kv_group_num = q.shape[1] // k.shape[1]

            self.hidden = h
            self.head = H
            self.sm_scale = 1.0 / (h ** 0.5)
            self.batch_size = batch_size
            self.num_block = total_blocks
            self.max_blocks = max_blocks
            self.max_blocks_round = max_blocks_round

            self.req_to_block = request_to_block
            self.block_to_batch = block_to_request
            self.block_to_start = block_to_start

            pot_total_blocks = triton.next_power_of_2(total_blocks)
            self.s_max = torch.empty((pot_total_blocks, H), dtype=torch.float32, device=q.device)
            self.s_exp_sum = torch.empty((pot_total_blocks, H), dtype=torch.float32, device=q.device)
            self.s_exp_v_sum = torch.empty((pot_total_blocks, H, h), dtype=torch.float32, device=q.device)
            return
