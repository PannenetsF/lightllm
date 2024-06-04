import os

import torch
from transformers.utils.generic import np

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import \
    destindex_copy_kv as scatter_kv
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import \
    srcindex_copy_kv as gather_kv
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class CPUMemoryManager:
    def __init__(self, mem_manager: MemoryManager, size: int, mov_buf_size: int):
        self.mem_manager = mem_manager
        # note that we do not have to record ref for cpu kv 
        # as they are not currently used in the model
        self.cpu_mem_state = torch.zeros((size,), dtype=torch.int32, device="cpu")
        self.cpu_can_use_mem_size = size
        
        nccl_port = os.environ.get("_NCCL_PORT_", None)
        assert nccl_port is not None
        self.shared_layer_indicators = SharedArray(f"{str(nccl_port)}_cpu_mem_manger_layer_indicators", (mem_manager.layer_num,), dtype=np.int32)
        self._init_buffers(size, mov_buf_size)

    def _init_buffers(self, pool_size, buf_size):
        dtype = self.mem_manager.kv_buffer[0].dtype
        _, head_num, head_dim = self.mem_manager.kv_buffer[0].shape
        layer_num = self.mem_manager.layer_num
        device_cpu = torch.device("cpu")
        device_cuda = self.mem_manager.kv_buffer[0].device
        pool_shape = (pool_size, head_num, head_dim)

        buf_shape = (buf_size, head_num, head_dim)
        self.layer_num = layer_num
        self.buf_size = buf_size
        self.cpu_kv_pool = [
            torch.empty(pool_shape, dtype=dtype, device=device_cpu)
            for _ in range(layer_num)
        ]
        self.cpu_kv_buf = torch.empty(buf_shape, dtype=dtype, device=device_cpu)
        self.gpu_kv_buf = torch.empty(buf_shape, dtype=dtype, device=device_cuda)


    def alloc(self, need_size: int):
        if need_size > self.cpu_can_use_mem_size:
            logger.warn(f"warn no enough cache need_size {need_size} left_size {self.cpu_can_use_mem_size}")
            return None
        can_use_index = torch.nonzero(self.cpu_mem_state == 0).view(-1)
        select_index = can_use_index[0:need_size]
        self.cpu_mem_state[select_index] = 1
        self.cpu_can_use_mem_size -= need_size
        return select_index
    
    def free(
        self, free_index: torch.Tensor
        ):
        self.cpu_mem_state[free_index] = 0
        self.cpu_can_use_mem_size += free_index.shape[0]

    def _clear_indicators(self):
        self.shared_layer_indicators.arr.fill(0)

    def offload(self, src_idx):
        # from gpu to cpu
        size = src_idx.shape[0]
        dst_idx = self.alloc(size)
        self._clear_indicators()
        for i in range(self.layer_num):
            for j in range((size + self.buf_size - 1) // self.buf_size):
                start = j * self.buf_size
                end = min((j + 1) * self.buf_size, size)
                # destindex_copy_kv(self.mem_manager.kv_buffer[i], src_idx[start:end], self.gpu_kv_buf[:end - start])
                gather_kv(self.mem_manager.kv_buffer[i], src_idx[start:end], self.gpu_kv_buf[:end - start])
                self.cpu_kv_buf[:end - start].copy_(self.gpu_kv_buf[:end - start])
                src_shape = self.cpu_kv_buf[:end - start].shape
                dst_shape = self.cpu_kv_pool[i][dst_idx[start:end]].shape
                # print(f'src_shape: {src_shape}, dst_shape: {dst_shape}, pool_size: {self.cpu_kv_pool[i].shape}, idx" {dst_idx}, slice: {dst_idx[start:end]}, j={j}')
                self.cpu_kv_pool[i][dst_idx[:end - start]] = self.cpu_kv_buf[:end - start]
                err = (self.cpu_kv_pool[i][dst_idx[:end - start]].clone().detach() - self.mem_manager.kv_buffer[i][src_idx[start:end]].clone().detach().cpu()).abs().sum()
            self.shared_layer_indicators.arr[i] = 1
        return dst_idx 

    def resume(self, src_idx, dst_idx):
        # from cpu to gpu 
        size = src_idx.shape[0]
        self._clear_indicators()
        for i in range(self.layer_num):
            for j in range((size + self.buf_size - 1) // self.buf_size):
                start = j * self.buf_size
                end = min((j + 1) * self.buf_size, size)
                self.cpu_kv_buf[:end - start].copy_(self.cpu_kv_pool[i][src_idx[start:end]])
                self.gpu_kv_buf[:end - start].copy_(self.cpu_kv_buf[:end - start])
                scatter_kv(self.gpu_kv_buf[:end - start], dst_idx[start:end], self.mem_manager.kv_buffer[i])
            self.shared_layer_indicators.arr[i] = 1
        self.free(src_idx)


